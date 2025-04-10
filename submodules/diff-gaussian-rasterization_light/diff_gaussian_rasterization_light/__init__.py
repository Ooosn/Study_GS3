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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, **kwargs):
        
        raster_settings = self.raster_settings
        kwargs["raster_settings"] = raster_settings


        shs = kwargs.get("shs", None)
        colors_precomp = kwargs.get("colors_precomp", None)
        scales = kwargs.get("scales", None)
        rotations = kwargs.get("rotations", None)
        cov3Ds_precomp = kwargs.get("cov3Ds_precomp", None)
        hgs_normals = kwargs.get("hgs_normals", None)
        hgs_opacities = kwargs.get("hgs_opacities", None)


        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or ((scales is not None or rotations is not None) and cov3Ds_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            kwargs["shs"] = torch.Tensor([])
        if colors_precomp is None:
            kwargs["colors_precomp"] = torch.Tensor([])
        if scales is None:
            kwargs["scales"] = torch.Tensor([])
        if rotations is None:
            kwargs["rotations"] = torch.Tensor([])
        if cov3Ds_precomp is None:
            kwargs["cov3Ds_precomp"] = torch.Tensor([])
        if hgs_normals is None:
            kwargs["hgs_normals"] = torch.Tensor([])
        if hgs_opacities is None:
            kwargs["hgs_opacities"] = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(**kwargs)

def rasterize_gaussians(**kwargs):
    # apply 不支持 关键字参数，因此需要手动指定参数顺序
    # 顺便调整参数顺序，方便调试
    param_order = [
        # 高斯点相关
        "means3D", "means2D", "shs", "colors_precomp", "opacities",
        "scales", "rotations", "cov3Ds_precomp", "raster_settings",

        # 阴影相关
        "non_trans", "offset", "thres", 
        
        # prune 相关
        "is_train",

        # hgs 相关
        "hgs", "hgs_normals", "hgs_opacities",

        # 流
        "light_stream"
    ]
    args = [kwargs[k] for k in param_order]
    return _RasterizeGaussians.apply(*args)

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        shs,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,

        # 阴影相关
        non_trans,
        offset,
        thres,

        # prune 相关
        is_train,

        # hgs 相关
        hgs,
        hgs_normals,
        hgs_opacities,

        # 流
        light_stream
    ):
        
        # Restructure arguments the way that the C++ lib expects them
        args = (
            # 高斯点/场景相关
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            shs,
            raster_settings.sh_degree,
            # 相机位置，在向光源泼溅中，则为光源位置
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            # opacity_light = torch.zeros(scales.shape[0], dtype=torch.float32, device=scales.device)
            # non_trans = opacity_light
            non_trans,  
            offset,
            thres,
            # is_train = prune_visibility: 
            is_train,

            # hgs 相关
            hgs,
            hgs_normals,
            hgs_opacities

            # 流
            # light_stream
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, out_weight, radii, trans, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, out_weight, radii, trans, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        # 保存 非 tensor 参数
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.non_trans=non_trans
        ctx.trans=trans

        """
        在 autograd.Function.forward 中，使用 PyTorch 的操作（如 +, *, sin 等）仍然会构建计算图。
        如果你保留了某个 Tensor（没有 .detach()），它就会成为计算图的一部分，占用显存和内存。

        ctx.save_for_backward(...) 的作用：
            - 会自动对传入 Tensor 调用 .detach()，只保存数值副本，不保计算图。
            - 保存的中间变量可以在 backward 中安全使用。
            - 可以防止中间变量继续参与 autograd 图，从而节省显存、避免循环引用。

        如果你不使用 save_for_backward 而直接保留中间变量（比如 ctx.foo = some_tensor），
        那么计算图可能会一直保留在这些变量里，造成内存无法释放。
        """
        # 相当于使用 .detach() ，只保存 tensor 的值，不会存储梯度，计算图等
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, shs, geomBuffer, binningBuffer, imgBuffer)
        
        # 返回 渲染结果，torch 会将他们作为计算图中的下一个节点
        # 这些梯度 会按照顺序 传入 backward 方法
        return color, out_weight, radii, trans

    @staticmethod
    def backward(ctx, grad_out_color, _, __, grad_out_trans):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        non_trans=ctx.non_trans
        trans=ctx.trans
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer= ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_trans,
                trans,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                non_trans,
                raster_settings.image_height,
                raster_settings.image_width)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,

            # 阴影相关
            None,
            None,
            None,

            # prune 相关
            None,

            # hgs 相关
            None,
            None,
            None,

            # 流
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool



