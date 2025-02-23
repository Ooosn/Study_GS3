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
import torch.nn.functional as F
import numpy as np
import math

from gsplat import rasterization
from utils.graphics_utils import fov2focal
from diff_gaussian_rasterization_light import GaussianRasterizationSettings as GaussianRasterizationSettings_light
from diff_gaussian_rasterization_light import  GaussianRasterizer as GaussianRasterizer_light
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import getProjectionMatrix, look_at

def render(viewpoint_camera, 
           gau : GaussianModel, 
           light_stream, 
           calc_stream, 
           local_axises, 
           asg_scales, 
           asg_axises, 
           pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0,  # 高斯点的缩放因子，默认为 1.0，用于二次调整整体的高斯点大小
           override_color = None,   # 覆盖学习到的颜色，默认为 None，即使用学习到的颜色
           fix_labert = False,  # 是否只考虑漫反射，根据当前迭代次数来决定
           inten_scale = 1.0,   # 颜色强度缩放，但是感觉被弃用了
           is_train = False,    # 根据 prune_visibility
           simplify = False,):  # 简化代码，没啥用
    
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # 根据当前 相机信息 计算 光栅化参数:
    # 计算相机的原始焦距: f = \frac{W}{2 \tan(\frac{\text{FoV}}{2})}
    # Set up rasterization configuration
    if simplify:
        fx_origin = fov2focal(viewpoint_camera.FoVx, viewpoint_camera.image_width)
        fy_origin = fov2focal(viewpoint_camera.FoVy, viewpoint_camera.image_height)
    else:    
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        # calculate the fov and projmatrix of light
        fx_origin = viewpoint_camera.image_width / (2. * tanfovx)
        fy_origin = viewpoint_camera.image_height / (2. * tanfovy)

    # calculate the fov for shadow splatting:
    # 计算光源和相机的距离比 f_scale_ratio
    light_position = viewpoint_camera.pl_pos[0].detach().cpu().numpy()
    camera_position = viewpoint_camera.camera_center.detach().cpu().numpy()
    f_scale_ratio = np.sqrt(np.sum(light_position * light_position) / np.sum(camera_position * camera_position))
    
    # 计算光源的焦距
    """ 
    1. 焦距随着距离成比例缩放
    2. 这里将光源作为一个相机
    3. 因此通过计算光源和相机的距离比，来缩放得到光源的 焦距
    """
    fx_far = fx_origin * f_scale_ratio
    fy_far = fy_origin * f_scale_ratio
    
    # 计算光源的视场角，即 FoV
    """
    相机的视锥体（View Frustum）决定了相机可以看到的空间范围：
        - 相机的视锥体是一个四棱锥：
            - 顶点 是相机的光心。
            - 成像平面（传感器 / near 平面），它始终与图像长宽一致，是固定的
            - 远平面 (far 平面)：透视扩展出来的“虚拟底”，它的大小不是固定的，而是透视投影的结果。
            - 图形学中，far 平面的距离可以受到人为规定，对场景进行裁剪，因此根据 fov 最后得到 far 平面的大小
            - 高 是焦距 f（光心到成像平面的距离），即定义了 near 平面。
        - 因此呈现出焦距大，范围小，焦距小，范围大
        - 焦距 和 FoV 是视野大小的不同表达方式，一个用距离，一个用角度，在确定 fov 形式（水平、垂直、对角）的情况下可以互相换算
    """
    # 先计算 FoV 对应的正切值，然后通过 arctan 得出 FoV
    tanfovx_far = 0.5 * viewpoint_camera.image_width / fx_far
    tanfovy_far = 0.5 * viewpoint_camera.image_height / fy_far
    fovx_far = 2 * math.atan(tanfovx_far)
    fovy_far = 2 * math.atan(tanfovy_far)


    # 光源方向的高斯泼溅（1）视角转换准备工作: 
    # 用于计算 shadow （shadow splatting）
    # 计算 世界坐标系 到 光源坐标系 的变换矩阵
    # 目前每个高斯点的坐标采取 行向量 来表示
    object_center=gau.get_xyz.mean(dim=0).detach()
    world_view_transform_light=look_at(light_position,
                                       object_center.detach().cpu().numpy(),
                                       up_dir=np.array([0, 0, 1]))
    world_view_transform_light=torch.tensor(world_view_transform_light,
                                            device=viewpoint_camera.world_view_transform.device,
                                            dtype=viewpoint_camera.world_view_transform.dtype)
    # 为了对齐 点 为行向量: P' = Pro Tw2c P  ->  P'^T = P^T (Tw2c)^T (Pro)^T
    # 这里对 投影矩阵进行转置, torch.transpose(0, 1)
    light_prjection_matric = getProjectionMatrix(znear=viewpoint_camera.znear, zfar=viewpoint_camera.zfar, fovX=fovx_far, fovY=fovy_far).transpose(0,1).cuda()
    # 计算 MVP 矩阵，M2W变换矩阵*W2C变换矩阵*投影矩阵 （这里 高斯点本来就在世界坐标中，所以 M 可以忽略）
    full_proj_transform_light = (world_view_transform_light.unsqueeze(0).bmm(light_prjection_matric.unsqueeze(0))).squeeze(0)
    # 设置光源的高斯泼溅参数
    raster_settings_light = GaussianRasterizationSettings_light(
        image_height = int(viewpoint_camera.image_height),
        image_width = int(viewpoint_camera.image_width),
        tanfovx = tanfovx_far,
        tanfovy = tanfovy_far,
        bg = bg_color[:3],
        scale_modifier = scaling_modifier,
        viewmatrix = world_view_transform_light,
        projmatrix = full_proj_transform_light,
        sh_degree = gau.active_sh_degree,
        campos = viewpoint_camera.pl_pos[0],
        prefiltered = False,
        debug = pipe.debug,
    )
    # 光源方向的高斯泼溅（2）高斯场景准备工作: 
    # 获得高斯点的 3D坐标、透明度、并初始化 2D坐标
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gau.get_xyz, dtype=gau.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    means3D = gau.get_xyz
    means2D = screenspace_points
    opacity = gau.get_opacity

    # 计算高斯点的方差:
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = gau.get_covariance(scaling_modifier)
    else:
        scales = gau.get_scaling
        rotations = gau.get_rotation

    # 光源方向的高斯泼溅（3）光源方向高斯泼溅信息计算: 
    # shadow splatting
    light_stream.wait_stream(torch.cuda.current_stream())   # 等待计算完成，因为下面要使用新的流，且有依赖

    with torch.no_grad():
        """
        !!! 阴影 shadow 从已有的高斯场景信息计算的，不是用来优化高斯本身，因此这里不能有梯度 
        """   
        with torch.cuda.stream(light_stream):   # 在光源流中进行计算
            rasterizer_light = GaussianRasterizer_light(raster_settings=raster_settings_light)
            opacity_light = torch.zeros(scales.shape[0], dtype=torch.float32, device=scales.device)
            _, out_weight, _, shadow = rasterizer_light(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = torch.zeros((2, 3), dtype=torch.float32, device=scales.device),
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                non_trans = opacity_light,
                offset = 0.015,
                thres = 4,
                is_train = is_train)

    # 1）光源方向的高斯泼溅（4）计算最终阴影:
    # 2）计算其他效果:
    # 3) 计算高斯点的最终颜色:
    # MBDRF 和 SH 的选择
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if gau.use_MBRDF:
            with torch.cuda.stream(calc_stream):
                """
                使用计算颜色的流，和光源流同步，两者间不存在数据竞争
                """
                assert viewpoint_camera.pl_pos.shape[0] == 1
                # calculate view and light dirs
                pl_pos_expand = viewpoint_camera.pl_pos.expand(gau.get_xyz.shape[0], -1) # (K, 3)
                wi_ray = pl_pos_expand - gau.get_xyz # (K, 3)
                # 归一化光源方向，可以直接 torch.nn.functional.normalize()，但是 dist_2_inv 后续也要用到，所以采用了这种方式
                dist_2_inv = 1.0 / torch.sum(wi_ray**2, dim=-1, keepdim=True)
                wi = wi_ray * torch.sqrt(dist_2_inv) # (K, 3)
                # 归一化视角方向
                wo = _safe_normalize(viewpoint_camera.camera_center - gau.get_xyz) # (K, 3)
                # local_axises 由 4元数 构建，因此高斯法线是归一化的
                local_z = local_axises[:, :, 2] # (K, 3)

                # transfer to local axis
                # 利用从 mbrdf 参数组中的得到的 local_axises 将 wi 和 wo 转移到对应的 高斯点-asg 坐标系 中
                """
                torch.einsum(): torch 计算的升级版，能够自定义很多种计算方式，例如：
                    - "Ki,Kij->Kj" 相当于 bmm
                    - "ik,kj->ij" 相当于 matmul
                    - "i,i->" 相当于 dot
                    - "ij->j" 相当于 torch.sum(A, dim=0)
                    等等 
                """
                # 右乘旋转矩阵，或者左乘旋转矩阵的逆
                wi_local = torch.einsum('Ki,Kij->Kj', wi, local_axises) # (K, 3)
                wo_local = torch.einsum('Ki,Kij->Kj', wo, local_axises) # (K, 3)

                # shading functions:
                """
                该部分计算每个高斯点的直接光照贡献，不考虑阴影（遮挡）或全局光照（间接光照）的影响。
                计算基于入射光的漫反射（Lambertian 反射）和镜面反射（基于 ASG 近似的高光反射）。

                · asg: asg 的旋转矩阵用于调整 asg 在高斯点局部坐标系中的方向
                        - asg 的 z 轴为 高斯球面的中心，也就是高光最密集的方向
                · mbdrf: 不直接使用表面法线，而是用 半角向量 h 作为“微表面法线” 进行计算
                        - 通过计算 h 和 asg 的夹角，来控制高光强度和形状，最终得出 wo 方位的反射强度

                · colors_precomp: 它代表的是 光照强度(irradiance)，由漫反射和镜面反射的贡献相加计算得到
                · local_axises: 是 高斯点的局部坐标系，由每个高斯点的局部旋转矩阵 local_q 得到
                · local_z: local_axises 中的 z 轴，相当于每一个高斯点的法线
                · cosTheta: Lambertian 余弦项，考虑入射角度对光照强度的影响
                · dist_2_inv: 光源的距离的倒数，考虑光强的距离衰减（通常 光照强度 与距离平方 成反比）
                
                -> 因此使用 colors_precomp * cosTheta * dist_2_inv 来修正颜色
                """
                cosTheta = _NdotWi(local_z, wi, torch.nn.ELU(alpha=0.01), 0.01)     # local_z, wi 都是朝外的，方向一致
                diffuse = gau.get_kd / math.pi
                specular = gau.get_ks * gau.asg_func(wi_local, wo_local, gau.get_alpha_asg, asg_scales, asg_axises) # (K, 3)

                # 刚开始只考虑 漫反射，不考虑其他反射，优化高斯的基础颜色
                if fix_labert:
                    colors_precomp = diffuse
                else:
                    colors_precomp = diffuse + specular 
                # intensity decays with distance
                colors_precomp = colors_precomp * cosTheta * dist_2_inv

            # calc_stream.wait_stream(light_stream)
            torch.cuda.current_stream().wait_stream(light_stream)
            torch.cuda.current_stream().wait_stream(calc_stream)
            
            # shaodow splat values
            opacity_light = torch.clamp_min(opacity_light, 1e-6)    # 防止最小值为0，产生 NaN
            # 归一化阴影值，使其不受不透明度影响
            shadow = shadow / opacity_light # (N,)
            assert not torch.isnan(shadow).any()
            
            # 神经网络优化 shadow 和 其他效果
            # neural components
            decay, other_effects = gau.neural_phasefunc(wi, wo, gau.get_xyz, gau.get_neural_material, shadow.unsqueeze(-1)) # (N, 1), (N, 3)
            
            # combine all components
            colors_precomp = torch.concat([colors_precomp * inten_scale, decay, other_effects * dist_2_inv * inten_scale], dim=-1) # (N, 7)
        
        # 是否使用 SH 来计算颜色
        elif pipe.convert_SHs_python:
            # 获得 SH 的特征
            """
            gau.get_features:   features_dc = self._features_dc
                                features_rest = self._features_rest
                                return torch.cat((features_dc, features_rest), dim=1) 
                                输入：(N, 1, C) + (N, D-1, C)
                                输出：(N, D, C)
            transpose:          (N, D, C)   ->  (N, C, D)
            view:               (N, C, D)   ->  (N, C, D)   
            """
            shs_view = gau.get_features.transpose(1, 2).contiguous().view(-1, 3, (gau.max_sh_degree+1)**2)
            dir_pp = (gau.get_xyz - viewpoint_camera.camera_center.repeat(gau.get_features.shape[0], 1))
            # 归一化方向
            """
            为什么要加 keepdim = True?
            因为:
                广播只能向前广播，比如 (x) -> (1, x) -> (N, x)
                不能向后广播，比如 (x) -> (N, 1) -> (N, x)
            """
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 计算 SH 到 RGB 的转换
            sh2rgb = eval_sh(gau.active_sh_degree, shs_view, dir_pp_normalized)
            # 将 sh2rgb [-0.5, 0.5] -> [0, 1]，并且截断不合理的值
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        else:
            # 不使用 SH，pass
            shs = gau.get_features
    
    # 如果 override_color 不为 None，则使用 override_color 作为颜色
    else:
        colors_precomp = override_color


    # 视角方向的高斯泼溅:
    #（这里使用的 gsplat 库，没有用原装 3dgs 的库）
    torch.cuda.synchronize()    # 等待所有流完成
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 根据 FoV 计算 焦距 
    focalx = fov2focal(viewpoint_camera.FoVx, viewpoint_camera.image_width)
    focaly = fov2focal(viewpoint_camera.FoVy, viewpoint_camera.image_height)
    K = torch.tensor([[focalx, 0, viewpoint_camera.cx], [0, focaly, viewpoint_camera.cy], [0., 0., 1.]], device="cuda")
    rendered_image, alphas, meta = rasterization(
        means = means3D, # [N, 3]
        quats = rotations, # [N, 4]
        scales = scales, # [N, 3]
        opacities = opacity.squeeze(-1), # [N]
        colors = colors_precomp, # [N, 7]
        viewmats = viewpoint_camera.world_view_transform.transpose(0, 1)[None, ...], # [1, 4, 4]
        Ks = K[None, ...], # [1, 3, 3]
        width = int(viewpoint_camera.image_width),
        height = int(viewpoint_camera.image_height),
        near_plane = viewpoint_camera.znear,
        far_plane = viewpoint_camera.zfar,
        eps2d = 0.3,
        sh_degree = None,
        packed = False,
        backgrounds = bg_color[None, ...]
    )

    # The intermediate results from fully_fused_projection
    # (H, W, C) → (C, H, W)，即 (RGB 通道数，高度，宽度)
    rendered_image = rendered_image[0].permute(2, 0, 1)
    radii = meta['radii'].squeeze(0)
    # 作用：确保 meta["means2d"] 在反向传播时保留梯度
    try:
        """
        torch.retain_grad() 作用：保留梯度，使得在反向传播时，该变量的梯度不会被释放 
            - PyTorch 默认只存叶子节点 (即参数） 的 grad
            - 因此即便中间变量的 grad 为 True，也不会保留，因为它不会被用于更新参数
            # intermediate_variable.is_leaf = False
            - 但是有时候我们需要中间变量的 grad，比如我们需要计算中间变量的梯度，或者我们需要中间变量的梯度来更新参数
            -> 因此，我们可以使用 retain_grad() 来保留中间变量的梯度
        """
        meta["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    torch.cuda.synchronize()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    # 返回渲染结果，根据不同的通道进行切分
    """
    !!!  colors_precomp = torch.concat
    ([colors_precomp * inten_scale, decay, other_effects * dist_2_inv * inten_scale], dim=-1) # (N, 7)
    """
    return {"render": rendered_image[0:3, :, :],
            "shadow": rendered_image[3:4, :, :],
            "other_effects": rendered_image[4:7, :, :],
            "viewspace_points": meta["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii,
            # 权重值，由不透明度（累计得到的），2d覆盖范围，深度等信息得到的
            # 例如：
            # - 一个完全不透明、离相机很近、覆盖多个像素的点 -> 大权重
            # - 一个半透明、离相机远、只覆盖一个像素的点 -> 小权重
            "out_weight": out_weight}

def _dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)

def _safe_normalize(x):
    return torch.nn.functional.normalize(x, dim = -1, eps=1e-8)

def _NdotWi(nrm, wi, elu, a):
    """
    nrm: (N, 3)
    wi: (N, 3)
    _dot(nrm, wi): (N, 1)
    return (N, 1)
    """
    tmp  = a * (1. - 1 / math.e)
    return (elu(_dot(nrm, wi)) + tmp) / (1. + tmp)
