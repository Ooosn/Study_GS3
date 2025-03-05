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
import random
import json
from utils.system_utils import searchForMaxIteration
from utils.general_utils import  get_expon_lr_func
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.graphics_utils import fov2focal

class Scene:       
    
    """
    主要分为四个功能：
        1. 加载场景数据集，无论是否加载旧模型都需要
        对于非加载旧模型，则为重新训练，需要创建新的模型文件夹，并将点云文件拷贝到新建的模型文件夹下，准备训练集的相机信息，将相机信息写入到 cameras.json 文件中
        而对于加载旧模型，该相机信息已经存在，所以不需要再次创建
        2. 加载高斯模型，如果加载旧模型则直接从点云中加载高斯模型，否则从场景信息的点云信息中创建高斯模型
        3. 优化相机参数和光源参数，根据命令行参数判断是否需要
        4. 储存高斯模型以及，如果优化了的话，相机参数，
        
    重要的属性：
        self.gaussians : GaussianModel 加载的高斯模型
        self.train_cameras : dict   加载的训练集相机信息
        self.test_cameras : dict    加载的测试集相机信息
        以及可能被优化的相机和光源参数，储存于内存或文件中，在后面训练和测试的时候会用到
    """

    gaussians : GaussianModel
                          
    def __init__(self, 
                 args : ModelParams,  # 传入模型参数，这里的args其实就是前文的 dataset
                 gaussians : GaussianModel, 
                 opt=None, # opt ： OptimizationParams 类型，优化参数
                 load_iteration=None, 
                 shuffle=True, 
                 resolution_scales=[1.0], 
                 valid=False):
        """b
        args: Parameters for the model, including paths and configurations.
        gaussians: The Gaussian model for the scene.
        opt: Optional optimization settings.
        load_iteration: Specifies a specific iteration to load, or -1 for the latest.
        shuffle: Whether to shuffle the training and testing cameras.
        resolution_scales: List of scales for resolution during training. Default is [1.0]
        :param path: Path to Blender scene main folder.
        """
        # 路径信息已经被更新/新建  tb_writer = prepare_output_and_logger(dataset)，在这里的 args 就是 train.py 中传递的 dataset
        self.model_path = args.model_path

        self.loaded_iter = None
        self.gaussians = gaussians

        # 判断是否需要加载已训练的模型
        # -1 代表加载最新的模型
        # 其他数字代表加载指定迭代次数的模型
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        # 无论是否加载已训练的模型，都需要加载场景数据集
        # 仅判断是否存在transforms_train.json文件，如果存在则认为是Blender数据集，随后在sceneLoadTypeCallbacks中调用Blender函数加载训练和测试集的相机信息
        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
        
            # scene_info 获得的是一个 SceneInfo 对象，包含了场景的所有信息，比如训练集和测试集的相机信息、点云信息、图片信息等
            """            SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)   # 点云信息地址，不存在则随机初始化
            """
            if not args.wang_debug:
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.view_num, \
                                                           valid=valid, extension=".exr" if args.hdr else ".png")
        else:
            assert False, "Could not recognize scene type!"

        # 场景信息部分
        # 如果不加载已训练的模型，则进行初始化
        # 1. 将点云文件拷贝到新建的模型文件夹下。
        # 2. 准备训练签的相机信息，将相机信息写入到 cameras.json 文件中
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:

                dest_file.write(src_file.read())

            json_cams = []
            camlist = []

            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 按照缩放比例分类，默认只有原始分辨率
        # 并且在此过程中，返回 相机 nn.module 对象 列表 ———— 特别是（每个 相机 对象中有）:
        #                 self.cam_pose_adj = torch.nn.Parameter(torch.zeros((1, 6), requires_grad=True).cuda())
        #                 self.pl_adj = torch.nn.Parameter(torch.zeros((1, 3), requires_grad=True).cuda())
        # 因此后面可以直接将这两个参数传入 优化器 torch.optim.Adam
        # 将之前的 scene_info 中的相机信息，按照缩放比例调整参数，返回 Camera 对象列表
        """
            Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, cx=cam_cx, cy=cam_cy,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  pl_pos=cam_info.pl_pos, pl_intensity=cam_info.pl_intensity,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  is_hdr=args.hdr, image_path=cam_info.image_path)
        """
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        
        # 高斯部分
        # 如果需要加载已训练的模型，则直接从点云中加载高斯模型
        if self.loaded_iter:                                        # os.path.join 可输入多个参数，将多个参数拼接成一个路径
            self.gaussians.load_ply(os.path.join(self.model_path,   # 模型路径 + 子文件夹 point_cloud + iteration_ + 迭代次数 子文件夹 + point_cloud.ply 文件
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        # 如果不加载已训练的模型，则从场景信息的点云信息中创建高斯模型（默认是创建随机点云，如果没有其他渠道获得场景对应的点云信息）
        else: 
            # create_from_pcd 中，会初始化神经网络 self.neural_phasefunc = Neural_phase(
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # 优化部分
        # 判断是否需要优化相机参数和光源参数
        # 如果cam_opt或pl_opt为True，则进行优化，默认为True #怀疑这里的逻辑有问题，应该是优化的时候才设置为True，在后续的运行中，他们也做了区分，但实际上这里是中为true
        if args.cam_opt or args.pl_opt:
            self.optimizing = True
        else:
            self.optimizing = False

        self.save_scale = 1.0

        if opt is not None:     #opt应该是肯定存在的，除非出了问题，没太懂这里的条件判断

            # optimizer for camera and light
            # get_expon_lr_func: 根据迭代次数，衰减步数以及延迟系数生成一个指数衰减的学习率函数
            # 后面直接使用 self.cam_scheduler_args(iteration) 来获取当前迭代次数下的学习率
            # 全局学习率调整
            self.cam_scheduler_args = get_expon_lr_func(lr_init=opt.opt_cam_lr_init,
                                                    lr_final=opt.opt_cam_lr_final,
                                                    lr_delay_steps=opt.opt_cam_lr_delay_step,
                                                    lr_delay_mult=opt.opt_cam_lr_delay_mult,
                                                    max_steps=opt.opt_cam_lr_max_steps)
            
            self.pl_scheduler_args = get_expon_lr_func(lr_init=opt.opt_pl_lr_init,
                                                    lr_final=opt.opt_pl_lr_final,
                                                    lr_delay_steps=opt.opt_pl_lr_delay_step,
                                                    lr_delay_mult=opt.opt_pl_lr_delay_mult,
                                                    max_steps=opt.opt_pl_lr_max_steps)
            
            cam_params = []
            pl_params = []

            # 保存时的分辨率，默认为1.0
            self.save_scale = resolution_scales[0]

            # 按照缩放比例遍历所有的相机，只将相机参数和光源参数添加到 cam_params 和 pl_params 中，不包括其他参数
            # camera对象的 cam_pose_adj 和 pl_adj 属性是一个 grad 为 true 的 tensor，用于保存相机参数和光源参数
            # extend 这里传递地址，extend 适合迭代时添加元素，更简洁。
            for scale in resolution_scales:
                cam_params.extend([self.train_cameras[scale][i].cam_pose_adj for i in range(len(self.train_cameras[scale]))])
                cam_params.extend([self.test_cameras[scale][i].cam_pose_adj for i in range(len(self.test_cameras[scale]))])
                pl_params.extend([self.train_cameras[scale][i].pl_adj for i in range(len(self.train_cameras[scale]))])
                pl_params.extend([self.test_cameras[scale][i].pl_adj for i in range(len(self.test_cameras[scale]))])

            # 默认为true
            # Adam 局部学习率调整
            if self.optimizing:
                # torch.optim.Adam 返回一个参数组对象，对象内维护了一个参数组列表，可以通过 self.optimizer.param_groups 访问，列表每个元素是一个参数组，用字典表示，其包含了优化器的参数
                # 此时学习率为0，后面会根据迭代次数更新学习率
                # - lr: 学习率（这里初始设为0.0）
                # - name: 自定义参数组名称（可选，默认为None）
                #   添加name是为了后续能够通过名字方便地找到并更新特定参数组的设置
                self.optimizer = torch.optim.Adam(
                    [
                        {"params": cam_params, "lr" : 0.0, "name": "cam_adj"},
                        {"params": pl_params, "lr" : 0.0, "name": "pl_adj"}
                    ],
                    lr = 0,
                    eps=1e-15
                )
            else:
                self.optimizer = None

    # 根据迭代次数解冻/update lr rate 
    # self.cam_scheduler_args(iteration) 返回当前迭代次数下的学习率
    def update_lr(self, iteration, freez_train_cam, freez_train_pl, cam_opt, pl_opt):
        # 是否优化，默认为true
        if self.optimizing:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "cam_adj":
                    # 如果不优化相机参数或者迭代次数小于解冻迭代次数，则学习率为0
                    if not cam_opt or iteration < freez_train_cam:
                        param_group['lr'] = 0
                    else:
                        lr = self.cam_scheduler_args(iteration)
                        param_group['lr'] = lr
                elif param_group["name"] == "pl_adj":
                    if not pl_opt or iteration < freez_train_pl:
                        param_group['lr'] = 0
                    else:
                        lr = self.pl_scheduler_args(iteration)
                        param_group['lr'] = lr
    

    # 保存高斯模型和相机参数（因为可能优化了）
    def save(self, iteration):

        # 保存高斯场景模型
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

        # 如果设置了优化，则保存相机参数到于 output 文件夹中的 transforms_train.json和transforms_test.json文件中
        if self.optimizing:
            cx = self.train_cameras[self.save_scale][0].cx
            cy = self.train_cameras[self.save_scale][0].cy
            fx = fov2focal(self.train_cameras[self.save_scale][0].FoVx, self.train_cameras[self.save_scale][0].image_width)
            fy = fov2focal(self.train_cameras[self.save_scale][0].FoVy, self.train_cameras[self.save_scale][0].image_height)
            intrinsics = [cx, cy, fx, fy]

            
            # 默认第一个相机的内参作为全局内参，方便使用，后面的相机只保存相对于第一个相机的相对位置
            # 但是如果使用不同相机，则依然需要存储各个相机的内参，因此此处采用了两种内参存储方式。

            # camera and point light in train set
            cam_new = {'camera_intrinsics': intrinsics, "frames": []}   #第一种方式，存储全局内参
            for i in range(len(self.train_cameras[self.save_scale])):
                camnow = self.train_cameras[self.save_scale][i]
                R, T, pl_pos = camnow.get("SO3xR3")
                focalx = fov2focal(camnow.FoVx, camnow.image_width)
                focaly = fov2focal(camnow.FoVy, camnow.image_height)
                cam_new["frames"].append(
                {
                    "file_path": camnow.image_name,
                    "img_path": camnow.image_path,
                    "R_opt": R.tolist(),
                    "T_opt": T.tolist(),
                    "pl_pos": pl_pos[0].tolist(),
                    "camera_intrinsics": [camnow.cx, camnow.cy, focalx, focaly],    #第二种方式，存储相对内参,考虑到可能使用不同相机
                })
            with open(os.path.join(point_cloud_path, "transforms_train.json"), "w") as outfile:
                json.dump(cam_new, outfile)
        
            # camera and light in test set
            cam_new = {'camera_intrinsics': intrinsics, "frames": []}
            for i in range(len(self.test_cameras[self.save_scale])):
                camnow = self.test_cameras[self.save_scale][i]
                R, T, pl_pos = camnow.get("SO3xR3")
                focalx = fov2focal(camnow.FoVx, camnow.image_width)
                focaly = fov2focal(camnow.FoVy, camnow.image_height)
                cam_new["frames"].append(
                {
                    "file_path": camnow.image_name,
                    "img_path": camnow.image_path,
                    "R_opt": R.tolist(),
                    "T_opt": T.tolist(),
                    "pl_pos": pl_pos[0].tolist(),
                    "camera_intrinsics": [camnow.cx, camnow.cy, focalx, focaly],
                })
            with open(os.path.join(point_cloud_path, "transforms_test.json"), "w") as outfile:
                json.dump(cam_new, outfile)

    """    
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
    """
    # 返回指定 resolution scale 下的训练集相机信息            
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    # 返回指定 resolution scale 下的测试集相机信息
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]