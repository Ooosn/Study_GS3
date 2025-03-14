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


import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.neural_phase_function import Neural_phase
from scene.mixture_ASG import Mixture_of_ASG
from utils.system_utils import searchForMaxIteration
from rich import print
from rich.panel import Panel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, gamma, write_image=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    local_axises = gaussians.get_local_axis         # (K, 3, 3)
    asg_scales = gaussians.asg_func.get_asg_lam_miu # (basis_asg_num, 2, channel_num)
    asg_axises = gaussians.asg_func.get_asg_axis    # (basis_asg_num, 3, 3)
        
    light_stream = torch.cuda.Stream()
    calc_stream = torch.cuda.Stream()
    render_shadow = None
    render_other_effects = None
    render_base = None
    rendering = None
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, light_stream, calc_stream, local_axises, asg_scales, asg_axises, pipeline, background)
        render_shadow = render_pkg["shadow"]
        render_other_effects = render_pkg["other_effects"]
        render_base = render_pkg["render"]
        rendering = render_pkg["render"]* render_pkg["shadow"] + render_pkg["other_effects"]
        # render_pkg["render"] 
        # * render_pkg["shadow"]   # 存在一些内存泄漏
        # + render_pkg["other_effects"]
        gt = view.original_image[0:3, :, :]
        
        if True:
            if gamma:
                gt = torch.pow(gt, 1/2.2)
                rendering = torch.pow(rendering, 1/2.2)
            os.makedirs(os.path.join(render_path, 'shadow'), exist_ok=True)
            os.makedirs(os.path.join(render_path, 'other_effects'), exist_ok=True)
            os.makedirs(os.path.join(render_path, 'base'), exist_ok=True)
            if render_shadow is not None:
                if not os.path.exists(os.path.join(render_path, 'shadow', '{0:05d}'.format(idx) + ".png")):
                    torchvision.utils.save_image(render_shadow, os.path.join(render_path, 'shadow', '{0:05d}'.format(idx) + ".png"))
            if render_other_effects is not None:
                torchvision.utils.save_image(render_other_effects, os.path.join(render_path, 'other_effects', '{0:05d}'.format(idx) + ".png"))
            if render_base is not None:
                torchvision.utils.save_image(render_base, os.path.join(render_path, 'base', '{0:05d}'.format(idx) + ".png"))
            if rendering is not None:
                torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, 
                iteration : int, 
                pipeline : PipelineParams, 
                skip_train : bool, 
                skip_test : bool, 
                opt_pose: bool, 
                gamma: bool,
                valid: bool,
                write_image: bool):
    dataset.data_device = "cpu"

    if opt_pose:
        dataset.source_path = os.path.join(dataset.model_path, f'point_cloud/iteration_{iteration}')

    with torch.no_grad():

        # load Gaussians attributes, establish scene
        gaussians = GaussianModel(dataset.sh_degree, dataset.use_nerual_phasefunc, basis_asg_num=dataset.basis_asg_num, asg_channel_num=dataset.asg_channel_num)
        
        # 创建场景实例
        # iteration 为 -1 则加载最新的模型，否则加载指定迭代次数的模型，如果不存在则创建新的模型文件夹，但这里作为渲染，iteration 肯定要的哇
        # valid 渲染验证集，skip_train 不渲染训练集，skip_test 不渲染测试集
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, valid=valid, skip_train=skip_train, skip_test=skip_test)
        
        """
        本项目中: scene 中储存的 model 也就是 ply 文件，只存储了每个高斯的属性，不包括每个高斯的神经网络参数，以及公用的参数
            因此需要先从 ply 文件中加载高斯模型参数，然后从 _model_path = os.path.join(dataset.model_path, f"chkpnt{iteration}.pth")
                中提取神经网络参数和一些共用的参数（比如 asg 基函数参数）
        这里的 model_path 是 模型文件夹，包括两个子模型，一个是 point_cloud; 一个是 chkpnt (torch.save(gaussians.capture()) 得到的文件)
        point_cloud 中存储了高斯模型参数，chkpnt 中存储了优化器状态以及神经网络参数和一些共用的参数
        """
        if dataset.use_nerual_phasefunc:
            if iteration == -1:
                iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
            _model_path = os.path.join(dataset.model_path, f"chkpnt{iteration}.pth")
            if os.path.exists(_model_path):
                (model_params, first_iter) = torch.load(_model_path)
                # load ASG parameters
                gaussians.asg_func = Mixture_of_ASG(dataset.basis_asg_num, dataset.asg_channel_num)
                gaussians.asg_func.asg_sigma = model_params[8]
                gaussians.asg_func.asg_rotation = model_params[9]
                gaussians.asg_func.asg_scales = model_params[10]
                # load MLP parameters
                gaussians.neural_phasefunc = Neural_phase(hidden_feature_size=dataset.phasefunc_hidden_size, \
                                        hidden_feature_layers=dataset.phasefunc_hidden_layers, \
                                        frequency=dataset.phasefunc_frequency, \
                                        neural_material_size=dataset.neural_material_size).to(device="cuda")
                gaussians.neural_phasefunc.load_state_dict(model_params[14])
                gaussians.neural_phasefunc.eval()
            else:
                raise Exception(f"Could not find : {_model_path}")

        bg_color = [1, 1, 1, 1, 0, 0, 0] if dataset.white_background else [0, 0, 0, 0, 0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if valid:
            print("valid")
            render_set(dataset.model_path, "valid", scene.loaded_iter, scene.getValidCameras(), 
                       gaussians, pipeline, background, gamma, write_image=True)
        
        if not skip_train:
            print("train")
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                       gaussians, pipeline, background, gamma, write_image=True)

        if not skip_test:
            print("test")
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                       gaussians, pipeline, background, gamma, write_image=True)
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--load_iteration", default=-1, type=int)   # -1 代表加载最新的模型
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gamma", action="store_true", default=False)
    parser.add_argument("--opt_pose", action="store_true", default=False)
    parser.add_argument("--valid", action="store_true", default=False)
    parser.add_argument("--write_image", action="store_true", default=False)
    # 加载训练模型所使用的参数
    args = get_combined_args(parser)
    args.wang_debug = False

    args_info = f"""
    model_args: {vars(model.extract(args))}
    load_iteration: {args.load_iteration}
    skip_train: {args.skip_train}
    skip_test: {args.skip_test}
    opt_pose: {args.opt_pose}
    gamma: {args.gamma}
    valid: {args.valid}
    """
    
    print(Panel(args_info, title="Arguments", expand=False))
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.load_iteration, pipeline.extract(args), \
                args.skip_train, args.skip_test, args.opt_pose, args.gamma, args.valid, args.write_image)