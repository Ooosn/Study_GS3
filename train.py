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

    # 初始化高斯实例
    gaussians = GaussianModel(dataset.sh_degree, use_MBRDF=dataset.use_nerual_phasefunc, basis_asg_num=dataset.basis_asg_num, \
                            hidden_feature_size=dataset.phasefunc_hidden_size, hidden_feature_layer=dataset.phasefunc_hidden_layers, \
                            phase_frequency=dataset.phasefunc_frequency, neural_material_size=dataset.neural_material_size,
                            maximum_gs=dataset.maximum_gs, asg_channel_num=dataset.asg_channel_num)

    # 根据 训练args，优化args 以及 初始高斯 建立场景实例，最终的高斯实例
    """
        1. 加载场景数据集，无论是否加载旧模型都需要
            对于非加载旧模型，则为重新训练，需要创建新的模型文件夹，并将点云文件拷贝到新建的模型文件夹下，准备训练集的相机信息，将相机信息写入到 cameras.json 文件中
            而对于加载旧模型，该相机信息已经存在，所以不需要再次创建
        2. 加载高斯模型，如果加载旧模型则直接从点云中加载高斯模型，否则从场景信息的点云信息中创建高斯模型
        3. 根据命令行参数判断是否需要添加优化相机参数和光源参数
    """
    scene = Scene(dataset, gaussians, opt=opt, shuffle=True)
    
    # 初始化参数设置
    gaussians.training_setup(opt)

    # 如果提供了检查点路径，则加载检查点的高斯参数
    # 如果为 True ，会覆盖上面的  gaussians.training_setup(opt) 
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
    
    dddd = 0

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



        """！开始渲染"""
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
        if iteration < opt.spcular_freeze_step + opt.fit_linear_step:   # 只考虑漫反射，不考虑镜面反射，刚开始训练时，先优化漫反射，赋予每个高斯点一个基础颜色
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, pipe, bg, fix_labert=True, is_train=prune_visibility)
        else:    # 开始考虑镜面反射             
            render_pkg = render(viewpoint_cam, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, pipe, bg, is_train=prune_visibility)
        
        # 此外，取出各个高斯点云坐标、可见性、半径，用于后续修剪
        viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
        if render_pkg["out_weight"].sum() > 0:
            print("################################")
            print(render_pkg["out_weight"])
            print("################################")
        # 如果迭代次数小于 unfreeze_iterations，则不考虑阴影和次要效果，此时 shadow 和 other_effects 都是 0
        if iteration <= unfreeze_iterations:
            image = image
        else:
            image = image * shadow + other_effects



        """！！Loss部分"""
        # 获取真实图片数据
        gt_image = viewpoint_cam.original_image.cuda()
        # 根据图片格式，进行调整
        if dataset.hdr:
            if iteration <= opt.spcular_freeze_step:
                gt_image = torch.pow(gt_image, 1./2.2)
            elif iteration < opt.spcular_freeze_step + opt.fit_linear_step//2:
                gamma = 1.1 * float(opt.spcular_freeze_step + opt.fit_linear_step - iteration + 1) / float(opt.fit_linear_step // 2 + 1)
                gt_image = torch.pow(gt_image, 1./gamma)
        else:
            image = torch.clip(image, 0.0, 1.0)

        Ll1 = l1_loss(image, gt_image)      # lamda_dssim 默认 0.2
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 反向传播，计算各个参数的梯度
        # 尚未更新参数，等待后续挑选更新
        loss.backward()
        iter_end.record() # 记录迭代结束的时间



        """！！参数更新部分"""
        # torch.no_grad() 防止污染计算图，加快计算速度
        with torch.no_grad():
            # Progress bar，平滑损失曲线
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


            # 对于测试集：：：：：  
            # 只用于优化场景光源和相机信息，因为这些信息可能是未知的，需要通过优化来估计    
            if opt_test and scene.optimizing:  
                """这里是否有一个漏洞，当 scene.optimizing 为 false 时，测试集也会进入后续的 else 分支，但是本文中可能默认 optimizing 为 true ，阴差阳错"""
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
                    # 更新相机参数和光源参数
                    scene.optimizer.step()
                    if iteration % 1000 == 0:
                        print(scene.optimizer.param_groups[0]['params'][0].grad)
                        print(scene.optimizer.param_groups[0]['params'][1].grad)
                    # 梯度清零
                    scene.optimizer.zero_grad(set_to_none = True)
                    # 梯度清零
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    """
                    1. set_to_none=True
                        .grad 直接变成 None。
                        更省显存，因为 PyTorch 不会存 0 矩阵，而是完全释放梯度张量。
                        计算时 会跳过 None 梯度的参数，不执行 += 操作
                    2. set_to_none=False (default)
                        梯度 .grad 变成全 0 矩阵。
                        依然占用显存，但 PyTorch 计算梯度时 不会跳过这些参数，仍然会进行 += 操作（梯度累积时有用）。
                    """
                    
                    if False:
                        if iteration % 100 == 0:
                            dddd += 1
                            print("--------------------------------")
                            print(dddd)
                            print(viewpoint_cam.cam_pose_adj)
                            print(viewpoint_cam.pl_adj)
                            print(viewpoint_cam.R)
                            print(viewpoint_cam.T)
                            print(viewpoint_cam.pl_pos)
                            print("--------------------------------")


    

            # 进入训练集：：：：：
            # 1) 对于非测试集，即训练集进入正常阶段，先进行高斯密集化，高斯修剪以及场景参数优化:
            else:

                # 第一步：：Densification, 高斯复制或分裂，when opt.density_from_iter < iteration < opt.densify_until_iter
                # 如果迭代次数小于高斯密集化迭代次数，则进行高斯密集化
                if iteration < opt.densify_until_iter:
                    # 更新每个可见高斯点在2D投影上的最大半径
                    # visibility_filter: 标记哪些点是可见的
                    # max_radii2D: 记录每个点在屏幕空间的最大半径
                    """
                    Bonus:
                        布尔索引：
                            布尔索引是一种用于选择数组中满足特定条件的元素的索引方式。
                            布尔索引通过一个布尔数组来选择数组中的元素，该布尔数组与原数组形状相同，其中的元素为True或False，表示是否选择该元素。
                            布尔索引通常用于根据某些条件过滤数组中的元素，或者根据条件对数组进行操作。
                    """
                    # 通过布尔索引更新当前视角下，可见高斯点在2D投影上的最大半径，其布尔索引由 render_pkg 中的 visibility_filter 提供
                    # radii：当前视角下，可见高斯点的半径（最长轴）
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    """
                    # 获得各个高斯点云坐标、可见性、半径
                    viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    # 获得渲染结果
                    image, shadow, other_effects = render_pkg["render"], render_pkg["shadow"], render_pkg["other_effects"]
                    """
                    # 记录高斯密集化统计信息
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1], render_pkg["out_weight"])

                    # 如果迭代次数大于高斯密集化开始迭代次数，并且迭代次数是高斯密集化迭代间隔的倍数，则进行高斯密集化
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        # size_threshold 是高斯密集化过程中，高斯点尺寸阈值
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # 进行高斯密集化(复制和分裂)，传入的最大梯度，最小透明度，场景范围（相机视锥），尺寸阈值
                        # 最小透明度和尺寸阈值在这里直接调节
                        # 共涉及 高斯密集化，透明度修剪，尺寸修剪，权重修剪，四种对高斯点的修改
                        if False:
                            if iteration % 1000 == 0:
                                print("################################")
                                print(gaussians.xyz_gradient_accum)
                                print(gaussians.denom)
                            print("################################")
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                        # 判断高斯点数量是否超过最大高斯点数量的95%，如果超过，则进行高斯修剪，剔除掉不可见的点
                        if gaussians.get_xyz.shape[0] > gaussians.maximum_gs * 0.95:
                            prune_visibility = True
                    
                    # 如果迭代次数是透明度重置迭代次数的倍数，或者在白色背景且迭代次数等于高斯密集化开始迭代次数，则进行透明度重置
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # 已完成高斯密集化阶段，后续不再进行高斯密集化，高斯点不会发生变多的情况，因此 prune_visibility 设置为 False
                else:
                    prune_visibility = False

                # 2) 完成高斯修剪阶段，再开始更新参数:
                # Optimizer step
                if iteration < opt.iterations:  # 判断是否处于优化阶段
                    gaussians.optimizer.step()
                    # opt the camera pose
                    if scene.optimizing:    # 判断是否开启相机优化
                        scene.optimizer.step()
                        scene.optimizer.zero_grad(set_to_none = True)
                    gaussians.optimizer.zero_grad(set_to_none = True)   # 梯度清零

            """
            1. 默认冻结 高斯点相位函数，初期训练高斯点场景空间为主，并只简单优化漫反射 kd (阴影和次要效果为 0)
            2. 抵达 unfreeze_iterations 时，解冻 高斯点相位函数，开始优化 阴影，材质，次要效果等
            3. 抵达 spcular_freeze_step 时，冻结 高斯点相位函数，再次只优化漫反射 kd
            4. 抵达spcular_freeze_step + fit_linear_step 之后，解冻 高斯点相位函数，并同时开始同时优化镜面反射 ks
            """
            # 初期默认冻结 高斯点相位函数（负责处理光照相关的相位特性（如材质、阴影、次要效果等））
            # 判断是否解冻 高斯点相位函数，解冻后该函数失效，由下列代码接管
            if phase_func_freezed and iteration >= unfreeze_iterations:
                gaussians.neural_phasefunc.unfreeze()
                phase_func_freezed = False

            # 抵达 spcular_freeze_step 时：开始优化相变函数，但依然只考虑 漫反射
            if iteration == opt.spcular_freeze_step:
                gaussians.neural_phasefunc.freeze()
                gaussians.neural_material.requires_grad_(False)

            # spcular_freeze_step + fit_linear_step 之后：开始同时优化 镜面反射 和 漫反射
            if iteration == opt.spcular_freeze_step + opt.fit_linear_step:
                gaussians.neural_phasefunc.unfreeze()
                gaussians.neural_material.requires_grad_(True)

            # 判断是否保存模型
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


        # 更新相机和光源参数：（之前只是更新了 adj 参数，尚未对相机和光源进行直接更新)，注意这里没有设置 torch.no_grad()
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
