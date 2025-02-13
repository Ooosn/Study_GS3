from collections import OrderedDict
import torch
import torch.nn as nn
import math
import tinycudann as tcnn


"""
- tcnn.Encoding（输入特征编码）
- tcnn.Network（神经网络结构）
- tcnn.Optimizer（优化器）
- tcnn.Loss（损失函数）
- tcnn.Linear（高效的全连接层）

tcnn 默认没有 bias
otype 模型类型


"""
# Neural_phase 继承了 torch.nn.Module，成为一个 PyTorch 的可训练模型
class Neural_phase(nn.Module):
    def __init__(self, hidden_feature_size=32, hidden_feature_layers=3, frequency=4, neural_material_size=6):
        super().__init__()
        self.neural_material_size = neural_material_size
        
        # 利用 tcnn 创建更高效的 mlp
        # configuration
        # frequency encoding
        encoding_config = {
                    "otype": "Frequency",
                    "n_frequencies": frequency
                }
        # shadow refine
        shadow_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "Sigmoid", 
                    "n_neurons": hidden_feature_size,
                    "n_hidden_layers": hidden_feature_layers,
                }
        # other effects
        other_effects_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "Sigmoid", 
                    "n_neurons": 128,
                    "n_hidden_layers": 3,
                }
        
        # 高频特征映射，把低维的 3D 坐标映射到一个高维的特征空间，让神经网络更容易学习高频细节
        # 4-band positional encoding，文中将 frequency 设置为4
        self.encoding = tcnn.Encoding(3, encoding_config)
        # 输入维度为 高斯点的材质维度/高斯点的可学习隐变量（neural_material_size）+ 光源方向，高斯中心高频特征（encoding.n_output_dims * 2） + 原始阴影维度（1）
        # 输出为 修正阴影值（维度为1）
        self.shadow_func = tcnn.Network(self.neural_material_size + self.encoding.n_output_dims * 2 + 1, 1, shadow_config)
        # 输入维度为 高斯点的材质维度/高斯点的可学习隐变量（neural_material_size）+ 观察方向、高斯中心高频特征 encoding.n_output_dims * 2）
        # 输出为 RGB 分量，RPG components
        self.other_effects_func = tcnn.Network(self.neural_material_size + self.encoding.n_output_dims * 2, 3, other_effects_config)
        
        """
        在 PyTorch 中，named_children() 返回当前模型的所有子模块的名称和对象，名称一般是模块的顺序
        """
        # 将所有线性层的参数初始化为 0.0
        # 每个高斯点都是独立的个体，MLP 初始时 不应该对数据进行过早的偏向（bias），而是应该让它们保持一致，等待训练逐步调整。
        # 采用 leaky_relu 作为激活函数，可以防止梯度消失，允许参数更快的跳出0，减少 weight=0 的影响
        for n, m in self.shadow_func.named_children():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight.data, 0.0)
                
        for n, m in self.other_effects_func.named_children():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight.data, 0.0)

    def freeze(self):
        for name, param in self.shadow_func.named_parameters():
            param.requires_grad = False
        for name, param in self.other_effects_func.named_parameters():
            param.requires_grad = False

    def unfreeze(self):
        for name, param in self.shadow_func.named_parameters():
            param.requires_grad = True
        for name, param in self.other_effects_func.named_parameters():
            param.requires_grad = True

    def forward(self, wi, wo, pos, neural_material, hint):
        # 光源方向
        wi_enc = self.encoding(wi)
        # 观察方向
        wo_enc = self.encoding(wo)
        # 高斯中心
        pos_enc = self.encoding(pos)

        # hint：原阴影值，decay：修正阴影值
        # other_effects，RGB 颜色修正（用于考虑其他效果带来的影响）
        """
        torch.concat:
        """
        other_effects = self.other_effects_func(torch.concat([wo_enc, pos_enc, neural_material], dim=-1)) * 2.0  #用于后面变为 [-1，1] 的颜色范围
        assert not torch.isnan(other_effects).any()
        decay = self.shadow_func(torch.concat([wi_enc, pos_enc, neural_material, hint], dim=-1))
        # torch.relu(decay - 1e-5) 如果 decay 过小，小于 1e-5，则设置为 0。这里应该除了为了减少计算开销，同时因为用的 leaky_relu 最后结果可能为负，所以需要用 relu 修正
        return torch.relu(decay - 1e-5), other_effects