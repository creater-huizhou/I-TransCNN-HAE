import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from einops.layers.torch import Rearrange
import copy


# 反向残差前馈网络IRFFN
class IRFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0., qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qk_dim = dim // qk_ratio

        self.q = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.qk_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        # Exactly same as PVTv1
        if self.sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim, bias=True),
                nn.BatchNorm2d(dim, eps=1e-5),
            )

    def forward(self, x, H, W, relative_pos):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            k = self.k(x_).reshape(B, -1, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, self.qk_dim // self.num_heads).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMT_block(nn.Module):
    def __init__(self, dim, num_heads, IRFFN_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_ratio=1, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, qk_ratio=qk_ratio, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        IRFFN_hidden_dim = int(dim * IRFFN_ratio)
        self.IRFFN = IRFFN(in_features=dim, hidden_features=IRFFN_hidden_dim, act_layer=act_layer, drop=drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        
    def forward(self, x, position):
        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        cnn_feat = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, position))
        x = x + self.drop_path(self.IRFFN(self.norm2(x), H, W))
        return x


# 跨层差异提示模块
class CDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(256+1, 256)

    def forward(self, fea_pred, fea_later): 
        fea_ori = fea_later
        fea_pred = F.normalize(fea_pred, dim=2)
        fea_later = F.normalize(fea_later, dim=2)
        dis = torch.bmm(fea_pred.permute(1, 0, 2), fea_later.permute(1, 2, 0))
        dis = torch.diagonal(dis, dim1=1, dim2=2).unsqueeze(-1)
        dis = 1 - dis
        dis_new = torch.cat((fea_ori, dis.permute(1, 0, 2)), dim=2)
        dis_new = self.linear(dis_new)
        weights = self.sigmoid(dis_new)
        out = fea_ori * weights + fea_ori
        
        return out


# 复制Transformer编码器块
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CMT_blocks(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, withCDP=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.with_cdp = withCDP
        if self.with_cdp is not None:
            self.cdp = CDP()

    def forward(self, src, pos=None):
        output = src
        if self.with_cdp is not None:
            outputs = []
        for layer in self.layers:
            if self.with_cdp is not None:
                outputs.append(output)
            output = layer(output, pos)
        if self.with_cdp is not None:
            output = self.cdp(outputs[-1], output) 
        if self.norm is not None:
            output = self.norm(output)

        return output


class CMTEncoders(nn.Module):
    def __init__(self, d_model=512, nhead=8, IRFFN_ratios=4, qkv_bias=True, qk_scale=None, dropout=0.0, activation=nn.GELU,
                  qk_ratio=1, sr_ratios=4, normalize_before=False, num_encoder_layers=5, withCDP=None):
        super().__init__()
     
        encoder_layer = CMT_block(dim=d_model, num_heads=nhead, IRFFN_ratio=IRFFN_ratios, qkv_bias=qkv_bias, 
                                  qk_scale=qk_scale, drop=dropout, attn_drop=dropout, drop_path=dropout, 
                                  act_layer=activation, norm_layer=nn.LayerNorm, qk_ratio=qk_ratio, sr_ratio=sr_ratios)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = CMT_blocks(encoder_layer, num_encoder_layers, encoder_norm, withCDP)

        self._reset_parameters()

    # 参数初始化
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # 参数初始化服从均匀分布U(−a,a), a = gain * sqrt(6/fan_in+fan_out), gain表示增益的大小，是依据激活函数类型来设定
                nn.init.xavier_uniform_(p)   

    def forward(self, x, position):
        # x: [4, 4096, 256]
        x = self.encoder(x, position)
            
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride, padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect': # 二维反射填充
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate': # 边界重复填充
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class CNNDecoders(nn.Module):
    def __init__(self, input_dim, output_dim, norm, activ, pad_type):
        super().__init__()
        self.model = []
        dim = input_dim
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        dim //= 2
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(dim, dim // 2, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)
        )
        self.conv3 = Conv2dBlock(dim//2, output_dim, 5, 1, 2, norm='none', activation='tanh', pad_type=pad_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        output = self.conv3(x2)
        return output


# 正余弦位置编码
def PatchPositionEmbeddingSine(ksize, stride):
    temperature = 10000
    feature_h = int((256 - ksize) / stride)+1 # 64
    num_pos_feats = 256 // 2 # 128
    mask = torch.ones((feature_h, feature_h))
    # 行方向求元素的累积和
    y_embed = mask.cumsum(0, dtype=torch.float32)
    # 列方向求元素的累积和
    x_embed = mask.cumsum(1, dtype=torch.float32)
    # 产生一维张量[0, 1, 2, ..., num_pos_feats-1]
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = torch.div(dim_t, 2, rounding_mode='floor')
    dim_t = temperature ** (2 * dim_t / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t # x_embed[:, :, None]为[64, 64, 1], dim_t为[128], pos_x为[64, 64, 128]
    pos_y = y_embed[:, :, None] / dim_t # 同上
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1) # pos为[256, 64, 64]
    return pos


class CMT_TransCNN(nn.Module):
    def __init__(self, device='cpu', batch=4):
        super().__init__()
        dim = 256

        self.patch_to_embedding = nn.Sequential(
            # 对张量的维度进行重新变换排序: [b, c, h*p1, w*p2] -> [b, h*w, p1*p2*3]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4),
            nn.Linear(4 * 4 * 3, dim)
        )

        if device == 'cuda':
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        b = batch
        input_pos = PatchPositionEmbeddingSine(ksize=4, stride=4).to(self.device)
        self.input_pos = input_pos.unsqueeze(0).repeat(b, 1, 1, 1) # self.input_pos为[4, 256, 64, 64]
        self.input_pos = self.input_pos.flatten(2).permute(0, 2, 1) # self.input_pos为[4, 4096, 256]    
        self.input_pos = self.input_pos.unsqueeze(1).repeat(1, 2, 1, 1)
        self.CMT_encoder = CMTEncoders(d_model=dim, nhead=2, IRFFN_ratios=2, qkv_bias=True, qk_scale=None, dropout=0.1,
                                       activation=nn.GELU, qk_ratio=1, sr_ratios=4, normalize_before=False, num_encoder_layers=6,
                                       withCDP=True).to(self.device)
        self.CNN_decoder = CNNDecoders(256, 3, 'ln', 'relu', 'reflect').to(self.device)

    def forward(self, inputs):
        patch_embedding = self.patch_to_embedding(inputs) # [4, 3, 256, 256] -> [4, 4096, 256]
        content = self.CMT_encoder(patch_embedding, self.input_pos)
        batchs, length, channels = patch_embedding.size() # 4, 4096, 256
        content = content.permute(0, 2, 1).view(batchs, channels, int(math.sqrt(length)), int(math.sqrt(length)))
        output = self.CNN_decoder(content)
        return output



 
