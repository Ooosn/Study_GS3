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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):        #检查参数是否支持简写，即使用'-'+首字母作为简写
                shorthand = True
                key = key[1:]              #去掉下划线
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:   #如果参数是布尔类型，则添加一个store_true的action，此外由于key已经规范化，所以可以直接使用key[0:1]
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()   #创建一个空对象，用于存储修改后的参数
        for arg in vars(args).items():  #vars(args) 返回的是 args 的属性和它们的值组成的字典    #items() 以列表返回可遍历的(键, 值) 元组数组
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):   #vars(self) 返回的是 self 的全部属性和它们的值组成的字典,因此检测当前属性名是否在 self 对象的属性中
                setattr(group, arg[0], arg[1])  #super().__setattr__ 只能操作当前对象（self）的属性 # 用 arg[1] 覆盖 self 各个属性的原始值
        return group #返回修改后的对象

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.hdr = False
        self.use_nerual_phasefunc = True
        # input images upper limit
        self.view_num = 2000
        # MLP parameter
        self.phasefunc_hidden_size = 32
        self.phasefunc_hidden_layers = 3
        # encoding frequency
        self.phasefunc_frequency = 4
        # latent size
        self.neural_material_size = 6
        # basis angular Gaussian num
        self.basis_asg_num = 8
        # optimize cam and pl or not
        self.cam_opt= True
        self.pl_opt= True
        # maximum gaussian number
        self.maximum_gs = 550_000


        """
        mine
        """
        # save time for debug
        self.wang_debug = False
        self.asg_channel_num = 1

        super().__init__(parser, "Loading Parameters", sentinel)

    # 提取出经过命令行参数修改后的参数
    def extract(self, args):
        g = super().extract(args)

        # os.path.abspath() 的参数为空字符串 ""，它会返回当前工作目录的绝对路径。
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.non_trans = 0
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        
        self.kd_lr = 0.01
        self.ks_lr = 0.01
        
        # only use diffuse term
        self.spcular_freeze_step = 9_000
        # gradually change to linear if trained under hdr mode (for a stable initialization)
        self.fit_linear_step = 7_000
        # ASG is initialized as aniso and freezed for a quick convergence
        self.asg_freeze_step = 22000
        
        # ASG lr
        self.asg_lr_freeze_step = 40_000
        self.asg_lr_init = 0.01
        self.asg_lr_final = 0.0001
        self.asg_lr_delay_mult = 0.01
        self.asg_lr_max_steps = 50_000
        
        # local frame lr
        self.local_q_lr_freeze_step = 40_000
        self.local_q_lr_init = 0.01
        self.local_q_lr_final = 0.0001
        self.local_q_lr_delay_mult = 0.01
        self.local_q_lr_max_steps = 50_000
        
        # latent and MLP lr
        self.freeze_phasefunc_steps = 50_000
        self.neural_phasefunc_lr_init = 0.001
        self.neural_phasefunc_lr_final = 0.00001
        self.neural_phasefunc_lr_delay_mult = 0.01
        self.neural_phasefunc_lr_max_steps = 50_000
        
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3_000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False

        # cam and pl opt lr:
        self.train_cam_freeze_step = 5_000
        self.opt_cam_lr_init = 0.001
        self.opt_cam_lr_final = 0.00001
        self.opt_cam_lr_delay_step = 20_000
        self.opt_cam_lr_delay_mult = 0.2
        self.opt_cam_lr_max_steps = 80_000

        self.train_pl_freeze_step = 15000
        self.opt_pl_lr_init = 0.001
        self.opt_pl_lr_final = 0.00005
        self.opt_pl_lr_delay_step = 30_000
        self.opt_pl_lr_delay_mult = 0.1
        self.opt_pl_lr_max_steps = 80_000

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
