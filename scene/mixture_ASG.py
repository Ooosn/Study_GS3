from utils.general_utils import build_rotation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _dot(x, y, keepdim=True):
    return torch.sum(x * y, -1, keepdim=keepdim)

class Mixture_of_ASG(nn.Module):
    def __init__(self, basis_asg_num=8):
        super().__init__()
        # asg 数量：
        self.basis_asg_num = basis_asg_num

        # asg 归一化因子：
        self.const = math.sqrt(2) * math.pow(math.pi, 2 / 3)
        # F0 是 Fresnel 反射率 (Fresnel Reflectance at Normal Incidence)，表示 当光线垂直入射时 (θ = 0°) 物体表面的反射率
        # F0 = 0.04 适用于 非金属表面（如塑料、木材、皮肤等）。金属的 F0 通常更高，例如 金 (0.77)、铝 (0.91)、铜 (0.95)
        self.F0 = 0.04

        # 选择激活函数：
        # 设置 softplus 激活函数，保证梯度的同时，且保证输出始终大于 0 ，而 leaky relu 会输出负数，这里的系数不应该存在负数
        """
        1. torch.nn.LeakyReLU()
        2. torch.nn.softplus(): 一种平滑的 ReLU 激活函数: log(1 + e^x)
            - 让输出始终大于 0 ，x > 0 时 类似于线性，x < 0 时，约等于 0 ，但始终有梯度
            - 平滑版的 ReLU，保证 梯度不会突然变 0，优化更稳定。
        注意：这两个都是类，需先定义再使用，除非是 torch.nn.functional.
        """
        self.softplus = torch.nn.Softplus()

        # 初始化 各 asg 的默认扩散范围：
        # initialized as different scale for fast convergency
        # sigma 控制 ASG 作为一个整体的大小
        # 创建等差数列，接着用 softplus 激活函数处理
        """
        torch.linspace(): 创建一个等差数列，从 start 到 end，创建一个元素为 steps 的等差数列
        """
        self.asg_sigma_ratio = self.softplus(torch.linspace(-2., 0., steps=self.basis_asg_num, device="cuda"))
        asg_sigma = torch.zeros((self.basis_asg_num), dtype=torch.float, device="cuda")
        self.asg_sigma = nn.Parameter(asg_sigma.requires_grad_(True))
        
        # 初始化 各 asg 的默认尺度/锐利系数：
        # scale（asg_scales）控制各向异性方向上的缩放（不同轴的拉伸程度），决定它是否是各向同性（圆形）还是各向异性（拉长或扁平）。
        asg_scales = torch.ones((self.basis_asg_num, 2), dtype=torch.float, device="cuda") * -2.1972
        asg_scales[:, 0] = asg_scales[:, 0] * 0.5
        self.asg_scales = nn.Parameter(asg_scales.requires_grad_(True))
        
        # A初始化 各 asg 的旋转方向
        # 依然采用 四元数
        asg_rotation = torch.zeros((self.basis_asg_num, 4), dtype=torch.float, device="cuda")
        asg_rotation[:, 0] = 1
        self.asg_rotation = nn.Parameter(asg_rotation.requires_grad_(True))

    # 这里并没有激活函数，需要使用 sigmoid 或其他激活函数，确保一些特殊参数训练过程中不会变成负数或极端值。
    """
    这里返回的都是计算方式，而不是在init中先计算，这里直接返回变量，是因为我们需要在 forward 中调用这些公式，从而加入计算图，backward 时优化参数
    如果直接返回变量，则它只会计算一次，不会在训练过程中更新，这样 参数 其实不会被优化！
    """
    @property
    def get_asg_lam_miu(self):
        return torch.sigmoid(self.asg_scales) * 10. # (basis_asg_num, 2)
    
    @property
    def get_asg_sigma(self):
        return torch.sigmoid(self.asg_sigma) * self.asg_sigma_ratio # (basis_asg_num)
    
    @property
    def get_asg_axis(self):
        return build_rotation(self.asg_rotation).reshape(-1, 3, 3) # (basis_asg_num, 3, 3)
    
    @property
    def get_asg_normal(self):
        return self.get_asg_axis[:, :, 2] # (basis_asg_num, 3)
    

    def forward(self, wi, wo, alpha, asg_scales, asg_axises):
        """
        wi, wo: (N, 3)
        alpha: (N, basis_asg_num)
        """
        half = F.normalize(wo + wi, p=2, dim=-1)                    # (N, 3)
        
        # 小角度（接近法线入射）→ 反射率低（光线主要透射）
        # 大角度（接近水平入射）→ 反射率高（光线几乎全反射）
        Fresnel = self.F0 + (1 - self.F0) * \
            torch.clamp(1.0 - _dot(wi, half), 0.0, 1.0).pow(5)      # (N, 1)
        
        half = half.unsqueeze(1).expand(-1, self.basis_asg_num, -1) # (N, basis_asg_num, 3)
        alpha = alpha                                               # (N, basis_asg_num)
        
        # axis of ASG frame
        asg_x = asg_axises[:, :, 0].unsqueeze(0)                    # (1, basis_asg_num, 3)
        asg_y = asg_axises[:, :, 1].unsqueeze(0)                    # (1, basis_asg_num, 3)
        asg_z = asg_axises[:, :, 2].unsqueeze(0)                    # (1, basis_asg_num, 3)

        # variance
        # 这里是之前 scale 两维度的具体作用，分别作为两个方向的尺度系数
        # sigma 则作为一个整体的扩展范围
        lam = asg_scales[:, 0].unsqueeze(0)                         # (1, basis_asg_num)
        miu = asg_scales[:, 1].unsqueeze(0)                         # (1, basis_asg_num)
        sigma = self.get_asg_sigma.unsqueeze(0)                     # (1, basis_asg_num)
        
        """
		高斯分布的中心（均值 μ）是 asg_z 方向，而 half 是输入方向
            - 在 ASG 中，高光的“法线”不是表面法线 n，而是 half
		    - half 是 wi 和 wo 的平均方向，代表微表面的局部法线
        """
        """
        asg_z 代表高光峰值中心，h 和 asg_z 的夹角 和 sigma 共同影响 高光的形状和强度
        """
        """
        half: 为 半角向量 ———— (wi + wo) / 2
        asg_z: ASG 坐标系的 z 轴方向
        (_dot(half, asg_z) * asg_z): 计算 half 在 asg_z 方向上的投影
        (half - 这个投影): 得到 half 在 xy 平面的投影。（三角形原则） # 即 s
        F.normalize(): 归一化投影为单位长度
        """

        s = F.normalize(half - _dot(half, asg_z) * asg_z, p=2, dim=-1)

        # 计算 h 偏离 asg_z 的方向，并根据不同方向进行各向异性缩放
        """
        lam 是 asg_x 方向的尺度参数（scale factor），控制 x 方向的高光拉伸程度
        miu 是 asg_y 方向的尺度参数，控制 y 方向的高光拉伸程度
        lam 和 miu 由可学习参数 asg_scales 决定
        s 代表 h 在 asg_x, asg_y 平面的方向
        最终 s 和 lam 、 miu 共同决定 高光的形状
        """
        aniso_ratio = torch.sqrt((_dot(s, asg_x, keepdim=False) / lam).pow(2) \
            + (_dot(s, asg_y, keepdim=False) / miu).pow(2))         # (N, basis_asg_num)
        
        # 计算夹角
        """
        torch.clamp(torch.clamp(input, min=None, max=None)): 将输入数值限制在指定范围内，防止超出某个区间。
            - min：最小值（可选），如果 input 小于 min，则设为 min
            - max：最大值（可选），如果 input 大于 max，则设为 max
        可能会出现浮点误差，当数值范围严格控制时，往往需要加入 1e-6 进行边界保护
        """
        """
        浮点数精度误差可能导致 cos_theta 超出 [-1,1]，因此这里设置让 cos_theta 处于 [-1+1e-6, 1-1e-6]，避免边界计算错误。
            - 浮点数精度误差通常在 1e-7 ~ 1e-15 级别，因此 1e-6 足够防止精度误差引起的问题
        """
        cos_theta = _dot(half, asg_z, keepdim=False)                # (N, basis_asg_num)
        cos_theta = torch.clamp(cos_theta, -1+1e-6, 1-1e-6)
        
        # 计算 asg 球面高斯 
        # \exp \left( -\frac{1}{2} \left( \frac{\theta \cdot \text{aniso_ratio}}{\sigma} \right)^2 \right)
        asg_res = torch.exp(- 0.5 * (torch.arccos(cos_theta) * aniso_ratio / sigma)**2)

        # 归一化
        asg_res = asg_res / (self.const * sigma)
        
        # 组合多个 ASG 的贡献 
        mm_asg_res = torch.sum(alpha * asg_res, dim=-1, keepdim=True) # (N, 1)
        
        return mm_asg_res * Fresnel                                   # (N, 1)   # 逐元素乘法 
