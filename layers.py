import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
import torch.nn.functional as F
from timm.layers import to_2tuple, trunc_normal_
from neuron import Multinary_spike_neuron
class MLP(nn.Module):
    def __init__(self, spiking_condition = None, neuron = None, surrogate = None, limiting_resting_potential=None, num_spike=None, trainable = None, beta=None, beta_spike = None, scaling=None, device = None, in_features = None, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        #定义各层参数，详细见forward
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        #全连接相当于卷积的卷积核为
        self.fc1_linear = nn.Linear(in_features, hidden_features, bias = False)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

        self.fc2_linear = nn.Linear(hidden_features, out_features, bias = False)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        # torch.Size([4, 64, 64, 384])
        t, b = x.shape[0], x.shape[1]
        x_ = x.flatten(0, 1)
        # torch.Size([256, 64, 384])
        x = self.fc1_linear(x_).reshape(t, b, -1)
        # 1536是隐藏维度为 4*C
        # torch.Size([256, 64, 1536])
        # torch.Size([4, 64, 64, 1536])
        x = self.fc1_lif(x)
        # torch.Size([4, 64, 64, 1536])

        x = self.fc2_linear(x.flatten(0,1)).reshape(t, b, -1)
        # torch.Size([256, 64, 384])

        # torch.Size([4, 64, 64, 384])
        x = self.fc2_lif(x)
        # torch.Size([4, 64, 64, 384])
        return x

class SSA(nn.Module):
    def __init__(self, spiking_condition = None, neuron = None, surrogate = None, limiting_resting_potential=None, num_spike=None, trainable = None, beta=None, beta_spike = None, scaling=None, device = None, dim = None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        #定义各层参数，详细见forward
        self.q_linear = nn.Linear(dim, dim, bias = False)
        self.q_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

        self.k_linear = nn.Linear(dim, dim, bias = False)
        self.k_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

        self.v_linear = nn.Linear(dim, dim, bias = False)
        self.v_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)
        self.attn_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

        self.proj_linear = nn.Linear(dim, dim, bias = False)
        self.proj_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

    def forward(self, x):
        #这个C就是特征维度D
        T,B,N,C = x.shape

        #以Cifar10的B=64,T=4,D=384,num_heads=12运算为例
        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        # torch.Size([256, 64, 384])

        #计算Query矩阵
        q_linear_out = self.q_linear(x_for_qkv)
        # [TB, N, C]
        # torch.Size([256, 64, 384])

        # torch.Size([4, 64, 64, 384])
        q_linear_out = self.q_lif(q_linear_out)
        # torch.Size([4, 64, 64, 384])
        # permute函数的作用是对tensor进行转置，这里多一个维度是采用多头自注意力机制
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # torch.Size([4, 64, 12, 64, 64])

        # 计算Key矩阵
        k_linear_out = self.k_linear(x_for_qkv)
       
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        # 计算Value矩阵
        v_linear_out = self.v_linear(x_for_qkv)
       
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        #attn为Q*K^T
        #q为torch.Size([4, 64, 12, 64, 64])，k.transpose(-2, -1)为torch.Size([4, 64, 12, 64, 64])
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # torch.Size([4, 64, 12, 64, 64])
        x = attn @ v
        # torch.Size([4, 64, 12, 64, 32])
        # 多头合并
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        # torch.Size([4, 64, 64, 384])
        x = self.attn_lif(x)
        # torch.Size([4, 64, 64, 384])
        x = x.flatten(0, 1)
        # torch.Size([256, 64, 384])
        x = self.proj_lif(self.proj_linear(x).reshape(T, B, N, C))
        # torch.Size([4, 64, 64, 384])
        return x

class Block(nn.Module):
    def __init__(self, spiking_condition = None, neuron = None, surrogate = None, limiting_resting_potential=None, num_spike=None, trainable = None, beta=None, beta_spike = None, scaling=None, device = None, dim = None, num_heads = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        #对单个batch归一化,似乎没有用上
        self.norm1 = norm_layer(dim)
        self.attn = SSA(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device, dim = dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        #imagenet中在这里还有一个DropPath层，是为了防止过拟合
        #drop_path 将深度学习模型中的多分支结构随机 “失效”，而 drop_out 是对神经元随机 “失效”。
        # 换句话说，drop_out 是随机的点对点路径的关闭，drop_path 是随机的点对层之间的关闭
        self.norm2 = norm_layer(dim)
        #计算mlp隐藏维度,按照论文中为4*D
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device, in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        #单个Encoder块，经过SSA自注意力和MLP多功能感知两个过程
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SPS(nn.Module):
    def __init__(self, spiking_condition = None, neuron = None, surrogate = None, limiting_resting_potential=None, num_spike=None, trainable = None, beta=None, beta_spike = None, scaling=None, device = None, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        #后续用于将图像分割为patch_size*patch_size的小块
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        # //是向下取整的运算
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        #四个SPS块，4块的卷积层的输出通道数分别是D/8、D/4、D/2、D，cifar10测试时我们采用的embed_dims为384
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_lif1 =Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_lif2 = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_lif3 = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        #相对位置嵌入
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_lif = Multinary_spike_neuron(spiking_condition = spiking_condition, neuron = neuron, surrogate = surrogate, limiting_resting_potential=limiting_resting_potential, num_spike=num_spike, trainable = trainable, beta=beta, beta_spike = beta_spike, scaling=scaling, device = device)

    def forward(self, x):
        # cifar10测试时为:torch.Size([4, 64, 3, 32, 32])
        # 这里的H,W是一个图片的宽高
        T, B, C, H, W = x.shape
        #flatten(0, 1)在第一维和第二维之间平坦化
        #x.flatten(0, 1)就是若干个C*H*W的矩阵,cifar10下为:torch.Size([256, 3, 32, 32])

        #注：这里前两个SPS没有池化层,但是imagenet的模型里每个SPS都有池化层
        #池化应该就是为了使最终的N不会太大，因为imagenet本身图片比较大所以才需要多池化几次
        #以cifar为例记录维度变化过程
        # torch.Size([4, 64, 3, 32, 32])
        x = self.proj_conv(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        # torch.Size([256, 48, 32, 32])
        # 这个48是384/8
        # torch.Size([4, 64, 48, 32, 32])
        # .reshape(T, B, -1, H, W)是为了传递给神经元
        # print(x.shape)
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        # torch.Size([256, 48, 32, 32])

        x = self.proj_conv1(x).reshape(T, B, -1, H, W).contiguous()
        # torch.Size([256, 96, 32, 32])
        # torch.Size([4, 64, 96, 32, 32])
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        # torch.Size([256, 96, 32, 32])

        x = self.proj_conv2(x).reshape(T, B, -1, H, W).contiguous()
        # torch.Size([256, 192, 32, 32])
        # torch.Size([4, 64, 192, 32, 32])
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        # torch.Size([256, 192, 32, 32])
        x = self.maxpool2(x)
        # torch.Size([256, 192, 16, 16])

        x = self.proj_conv3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        # torch.Size([256, 384, 16, 16])
        # torch.Size([4, 64, 384, 16, 16])
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        # torch.Size([256, 384, 16, 16])
        x = self.maxpool3(x)
        # torch.Size([256, 384, 8, 8])

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        # torch.Size([256, 384, 8, 8])
        x = self.rpe_conv(x).reshape(T, B, -1, H//4, W//4).contiguous()
        # torch.Size([256, 384, 8, 8])
        # torch.Size([4, 64, 384, 8, 8])
        x = self.rpe_lif(x)
        # torch.Size([4, 64, 384, 8, 8])
        x = x + x_feat
        # torch.Size([4, 64, 384, 8, 8])
        # T,B,N,C(这里的C就是特征维度D)
        x = x.flatten(-2).transpose(-1, -2)
        # 8,8合并为64，然后64混合384交换位置
        # torch.Size([4, 64, 64, 384])
        # imagenet中就没有这一行，保留了H和W最后两个分量，imagenet中在SSA中再合并
        return x




