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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

"""
函数：接受相机外参矩阵，[R | T]， 这里的R，T 虽然是基于 W2C 的，但是由于 cuda 中的 glm 库采用列主序 ，旋转矩阵取了转置，这里需要转置恢复
     - 此外，此函数接受 利用 translate 和 scale 对 R，T 进行归一化，先获得 C2W，然后提取平移矩阵进行归一化计算，再转回 W2C
返回：齐次坐标系下的 W2C 矩阵
"""
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_cu(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4), dtype=torch.float32).to(device=R.device)
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    
    C2W = torch.linalg.inv(Rt)
    C2W_translated = C2W.clone()
    C2W_translated[:3, 3] = (C2W[:3, 3] + translate.to(device=R.device)) * scale
    Rt = torch.linalg.inv(C2W_translated)
    return Rt

"""
函数：
    投影变化的第一步，准备透视投影 ———— 把 3D 坐标转换成 4D 齐次坐标 
返回：
    投影矩阵（注意和数学中 投影矩阵 概念区分）    
"""
def getProjectionMatrix(znear, zfar, fovX, fovY):
    # fovX，fovY 是 视锥体的宽高比，或者说 相机的视场角
    # top, bottom, right, left 计算的是 近平面上的视角矩形
	# znear, zfar 定义 视锥体的深度
    """
    物体必须在视锥体范围内才能被渲染，否则会被裁剪掉:
    近平面的位置 = 相机位置 + znear 距离
    远平面的位置 = 相机位置 + zfar 距离
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def look_at(camera_position, target_position, up_dir):

    # 计算相机坐标:
    """
    在 3D 计算机图形学中，相机有自己的局部坐标系，通常规定：
		- X 轴 (Right): 相机的右方向，通常希望与“世界上方向 (up_dir)”尽可能保持正交
		- Y 轴 (Up): 相机的上方向，必须垂直于 X 轴和 Z 轴，确保是右手坐标系
		- Z 轴 (Backward): 相机的朝向方向（但在 OpenGL 里是 -Z）
    """
    # 计算相机方向（从目标指向相机，即相机的 -Z 轴方向）
    camera_direction = camera_position - target_position
    camera_direction = camera_direction / np.linalg.norm(camera_direction)

    # 计算相机的“右方向” (X 轴方向)
    """
    1. up_dir: 通常是 [0,0,1]，代表世界 z 轴的“上”方向
    2. 得到右方向 camera_right: 这个方向垂直于 up_dir 和 camera_direction 组成的平面
    """
    camera_right = np.cross(up_dir, camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)

    # 计算相机的“上方向” (Y 轴方向)  
    """
    利用 camera_right 和 camera_direction 计算出 camera_up
    """
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    # 组装 Tc2w（相机坐标系到世界坐标系的齐次变换矩阵）:
    # 这个变换矩阵将 世界坐标系中的点变换到相机坐标系。
    """
    1. `camera_right` 作为第一行，表示相机的 X 轴
    2. `camera_up` 作为第二行，表示相机的 Y 轴
    3. `camera_direction` 作为第三行，表示相机的 -Z 轴
    !!! 标准的 变换矩阵 应该是列向量作为坐标系的基向量
    !!! 可能是因为计算得到的是 数组，而数组对行向量赋值更方便，因此这里使用了行向量作为坐标系的基向量
    """
    rotation_transform = np.zeros((4, 4))
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[-1, -1] = 1.0    # 

    # 计算 Tw2c (世界坐标系到相机坐标系的齐次变换矩阵):
    # 创建一个 4×4 单位矩阵，并存储平移变换
    """
    np.eye(num):
    torch
    """
    translation_transform = np.eye(4)
    """
    假设相机位置是 camera_position = (X, Y, Z)，我们希望：
		- 把整个世界坐标系 向相机的反方向移动  ( -X, -Y, -Z ) 。
		- 这样，相机在新坐标系下位于原点 (0,0,0)，方便计算。
    ！！！但是，切记，别忘了旋转，所以 平移矩阵应该是 T = -R^T C_W, (R 中列向量作为基向量)
    ！！！这里 R 是 行向量作为坐标轴，因此不需要转置
    """
    translation_transform[:3, -1] = -np.array(camera_position)

    # 组装 平移矩阵和旋转矩阵
    # [:3, :3] 为单位矩阵，[:3, 3] 为 相机在世界坐标系下的坐标
    # 正常来说，旋转矩阵应该
    look_at_transform = np.matmul(rotation_transform, translation_transform)

    # 翻转 Y 轴和 Z 轴，使得矩阵符合期望的坐标系统。
    """
    T = -R^T C_W    ->    T = -(R'R)^T C_W    
        # R' 是相对于 R ，再次变换坐标轴的 变换矩阵，即最终的 变换矩阵（最终坐标系的基向量作为矩阵的列向量）
    !!! 有些是先翻转旋转矩阵，因此在计算平移矩阵时，已经考虑了翻转，这里是在最后统一翻转
    """
    look_at_transform[1:3, :] *= -1
    
    # 为了和后面的计算对齐，因此取转置
    """
    !!! 高斯点的坐标一般为 (N,3)，因此坐标以行向量的形式存在，为了和行向量匹配: 
    设列向量 a ，采用 a^T * (Tc2w)^T 而不是最初的 (Tc2w) * a
    """
    look_at_transform=look_at_transform.T
    
    return look_at_transform
