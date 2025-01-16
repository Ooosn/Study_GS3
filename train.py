#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

#training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.unfreeze_iterations, args.debug_from)
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, unfreeze_iterations, debug_from):
    first_iter = 0

    # 准备输出和日志记录器，更新模型路径
    tb_writer = prepare_output_and_logger(dataset)

    # 建立高斯对象，场景对象
    gaussians = GaussianModel(dataset.sh_degree, use_MBRDF=dataset.use_nerual_phasefunc, basis_asg_num=dataset.basis_asg_num, \
                            hidden_feature_size=dataset.phasefunc_hidden_size, hidden_feature_layer=dataset.phasefunc_hidden_layers, \
                            phase_frequency=dataset.phasefunc_frequency, neural_material_size=dataset.neural_material_size,
                            maximum_gs=dataset.maximum_gs)

    scene = Scene(dataset, gaussians, opt=opt, shuffle=True)
    
    # 初始化模型的训练设置
    gaussians.training_setup(opt)

    # 如果提供了检查点路径，则加载模型和训练状态
    # 检查点通常包含模型参数和训练迭代次数
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    # 设置背景颜色，1,1,1,1,0,0,0表示白色背景，0,0,0,0,0,0,0表示黑色背景
    bg_color = [1, 1, 1, 1, 0, 0, 0] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0]
    # 将 bg_color 转换为 PyTorch 的张量，并将其分配到 GPU（device="cuda"）以加速后续的计算。
    # 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 新建事件，用于记录迭代开始和结束的时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 初始化一些训练过程中需要使用的变量
    prune_visibility = False    # 可见性修剪，是否剔除不可见的点，可以释放内存，提高内存使用效率
    viewpoint_stack = None    # 存储相机视点（viewpoints）的堆栈，在训练过程中，会从堆栈中弹出一个相机视点，用于渲染图像，从而完成对所有视角的遍历。
    opt_test = False    # 当前是否处于优化测试模式
    opt_test_ready = False    # 是否准备好进行优化测试
    """
    指数移动平均（EMA）损失，用于记录训练过程中的损失值，只是为了平滑损失曲线，更好地可视化训练过程，不参与模型的训练。
    ema_loss_for_log = α * 当前损失 + (1 - α) * ema_loss_for_log 类似于梯度更新中的一阶动量。
    """
    ema_loss_for_log = 0.0    # 初始值为 0.0，表示尚未计算损失值
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")    #使用 tqdm 创建一个可视化的进度条，用于显示训练的进度
    first_iter += 1    # 迭代次数加一，开始下一次迭代
    """
    相位函数（Phase Function） 是一个来自图形学和物理学的概念，主要用于描述光与介质相互作用时，光的散射方向分布。
    本文中，即为神经相位函数（Neural Phase Function），是一个用于描述材质表面散射特性的神经网络模型。
    """
    """
    ASG（Anisotropic Spherical Gaussians, 各向异性球面高斯）
    本文中，ASG 作为镜面反射的模型，用于描述材质表面的镜面反射特性。
    """
    # 用于记录是否冻结了相位函数
    # 根据当前迭代次数，决定是否冻结相位函数
    phase_func_freezed = False
    asg_freezed = True
    if first_iter < unfreeze_iterations:
        gaussians.neural_phasefunc.freeze()
        phase_func_freezed = True
        
    # initialize parallel GPU stream 多流并行
    # 有时会出现错误，可以尝试关闭，改为串行，torch.cuda.current_stream().synchronize()
    light_stream = torch.cuda.current_stream()
    calc_stream = torch.cuda.current_stream()
    
    """开始训练"""
    # 每次迭代，都会从视点堆栈中选择一个视点，然后渲染图像，计算损失，更新模型参数，并不是每次计算全部视点的损失
    for iteration in range(first_iter, opt.iterations + 1):    #左闭右开区间，因此加1
        iter_start.record()    # 记录迭代开始的时间

        # update lr of asg
        gaussians.update_learning_rate(iteration, \
                                        asg_freeze_step=opt.asg_lr_freeze_step, \
                                        local_q_freeze_step=opt.local_q_lr_freeze_step, \
                                        freeze_phasefunc_steps=opt.freeze_phasefunc_steps)
        # opt camera or point light
        if scene.optimizing:
            scene.update_lr(iteration, \
                            freez_train_cam=opt.train_cam_freeze_step, \
                            freez_train_pl=opt.train_pl_freeze_step, \
                            cam_opt=dataset.cam_opt, \
                            pl_opt=dataset.pl_opt)
            
        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 采取神经网络后，不再使用球谐函数，因此不再需要逐级增加球谐函数的阶数
        if iteration % 1000 == 0:
            if not dataset.use_nerual_phasefunc:
                gaussians.oneupSHdegree()
                
        if iteration <= opt.asg_freeze_step:
            gaussians.asg_func.asg_scales.requires_grad_(False)
            gaussians.asg_func.asg_rotation.requires_grad_(False)
        # else if iteration > opt.asg_freeze_step and asg_freezed:
        # 后续的迭代中，asg_freezed 可能已为 False，所以不需要重新设置
        elif asg_freezed:
            asg_freezed = False
            gaussians.asg_func.asg_scales.requires_grad_(True)
            gaussians.asg_func.asg_rotation.requires_grad_(True)
            print("set ansio param requires_grad: ", gaussians.asg_func.asg_scales.requires_grad)
        
        # Pick a random Camera
        # 如果视点堆栈为空，则根据 opt_test_ready 和 opt_test 的值，进行轮番选择训练视点和测试视点
        if not viewpoint_stack:
            # only do pose opt for test sets
            if opt_test_ready and scene.optimizing:
                opt_test = True
                # 重新填装测试视点堆栈
                viewpoint_stack = scene.getTestCameras().copy()
                opt_test_ready = False
            else:
                opt_test = False
                # 重新填装训练视点堆栈
                viewpoint_stack = scene.getTrainCameras().copy()
                opt_test_ready = True

        # 为当前迭代选择一个视点
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        """！！！开始渲染"""
        # debug用，一般不需要
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 选择背景颜色，如果opt输出为随机背景，则随机选择一个背景颜色，否则使用固定背景颜色
        bg = torch.rand((7), device="cuda") if opt.random_background else background
        
        # precompute shading frames and ASG frames
        local_axises = gaussians.get_local_axis # (K, 3, 3)
        asg_scales = gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, 2)
        asg_axises = gaussians.asg_func.get_asg_axis    # (basis_asg_num, 3, 3)

        # only opt with diffuse term at the beginning for a stable training process
        if iteration < opt.spcular_freeze_step + opt.fit_linear_step:
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, pipe, bg, fix_labert=True, is_train=prune_visibility)
        else:
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, pipe, bg, is_train=prune_visibility)
        
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
        
        if iteration <= unfreeze_iterations:
            image = image
        else:
            image = image * shadow + other_effects

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
        if dataset.hdr:
            if iteration <= opt.spcular_freeze_step:
                gt_image = torch.pow(gt_image, 1./2.2)
            elif iteration < opt.spcular_freeze_step + opt.fit_linear_step//2:
                gamma = 1.1 * float(opt.spcular_freeze_step + opt.fit_linear_step - iteration + 1) / float(opt.fit_linear_step // 2 + 1)
                gt_image = torch.pow(gt_image, 1./gamma)
        else:
            image = torch.clip(image, 0.0, 1.0)

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 反向传播，计算各个参数的梯度
        loss.backward()

        iter_end.record()

        # torch.no_grad() 用于关闭梯度计算，加速模型的预测和推理过程
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), \
                testing_iterations, scene, render, (pipe, background), gamma=2.2 if dataset.hdr else 1.0)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if opt_test and scene.optimizing:
            # 只是用于优化场景光源和相机信息，因为这些信息是未知的，需要通过优化来估计
            # 不会用于直接优化高斯模型
                if iteration < opt.iterations:
                    """"
                    在相机初始化时，cam_params 和 pl_params 都是 torch.nn.Parameter，默认开启了梯度计算（requires_grad=True）。
                    因此，当 viewpoint_cam 中的每个视角（即相机对象）参与计算时，这些参数会被自动追踪并记录到计算图中。
                    因为根据 torch 以及之前我们自己定义的函数，传递的都是地址，所以可以实现有效的一一对应的更新。

                    由于 PyTorch 的机制，我们在初始化时传递的是参数的引用（内存地址），因此这些参数在计算图中的位置与原始定义的位置一一对应。
                    换句话说，无论是在计算还是更新过程中，操作的始终是这个对象所包含的实际参数，确保了一一对应的关系。

                    优化器通过初始化时绑定的参数引用（内存地址），能够直接访问这些参数的 .grad 属性。
                    在调用 optimizer.step() 时，优化器根据计算出的梯度和设置的学习率更新参数值，实现了有效的一一对应更新。
                    """
                    scene.optimizer.step()
                    scene.optimizer.zero_grad(set_to_none = True)
                    # do not optimize the scene
                    gaussians.optimizer.zero_grad(set_to_none = True)
            else:
                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1], render_pkg["out_weight"])

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                        if gaussians.get_xyz.shape[0] > gaussians.maximum_gs * 0.95:
                            prune_visibility = True
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                else:
                    prune_visibility = False

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    # opt the camera pose
                    if scene.optimizing:
                        scene.optimizer.step()
                        scene.optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)
                
            if phase_func_freezed and iteration >= unfreeze_iterations:
                gaussians.neural_phasefunc.unfreeze()
                phase_func_freezed = False

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if iteration == opt.spcular_freeze_step:
                gaussians.neural_phasefunc.freeze()
                gaussians.neural_material.requires_grad_(False)
            
            if iteration == opt.spcular_freeze_step + opt.fit_linear_step:
                gaussians.neural_phasefunc.unfreeze()
                gaussians.neural_material.requires_grad_(True)

        # update cam and light
        if scene.optimizing:
            viewpoint_cam.update("SO3xR3")
    

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gamma=1.0):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        light_stream = torch.cuda.Stream()
        calc_stream = torch.cuda.Stream()
        local_axises = scene.gaussians.get_local_axis # (K, 3, 3)
        asg_scales = scene.gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, sg_num, 2)
        asg_axises = scene.gaussians.asg_func.get_asg_axis    # (basis_asg_num, sg_num, 3, 3)
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, *renderArgs)
                    mimage, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
                    image = torch.clamp(mimage * shadow + other_effects, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None].pow(1./gamma), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None].pow(1./gamma), global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--unfreeze_iterations", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    # 令 args 包含 parser 中定义的所有参数的键和值
    args = parser.parse_args(sys.argv[1:])  # argument vector， 第一位是文件名，所以从第二位开始解析
    args.save_iterations.append(args.iterations)    # 将最后一个迭代次数加入保存迭代次数列表
    
    # Prepare training
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)  #设置输出流是否静默

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  #如果命令行参数中包含 --detect_anomaly，则根据 stong_true 设置为True，将进行异常检测
    
    # Start training
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.unfreeze_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
