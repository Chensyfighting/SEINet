import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import warnings
import numbers
from model.module.neuron import LIFAct

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class TAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        return out


class SCMF(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type, in_channel, out_channel):
        super(SCMF, self).__init__()
        self.lamda = 0.5
        self.num_heads = num_heads

        self.norm = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, groups=dim)
        self.conve = TAttention(channel=dim,reduction=8)
        self.convr_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.convr_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)
        self.convr_3 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.convr_4 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)
        self.convr_5 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.convr_6 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.q_lif = LIFAct(step=1)
        self.k_lif = LIFAct(step=1)
        self.v_lif = LIFAct(step=1)
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.x_out = LIFAct(step=1)

    def forward(self, event, rgb):
        b, c, h, w = event.shape  # (B,C2,H/4,W/4)
        x_e = self.norm(event)  # (B,C2,H/4,W/4)-->(B,C2,H/4,W/4)
        x_r = self.norm(rgb)

        out_e = self.project_out(self.conve(x_e))
        out_r = self.project_out(self.convr_1(x_r) + self.convr_2(x_r) + self.convr_3(x_r) +
                                 self.convr_4(x_r) + self.convr_5(x_r) + self.convr_6(x_r))

        # 分别通过e轴和r轴上的融合后的多尺度特征，来生成qkv
        k1 = rearrange(out_e, 'b (head c) h w -> b head h (w c)',
                       head=self.num_heads).unsqueeze(0)  # (B,C2,H/4,W/4)-->(B,k,H/4,d*(W/4))
        v1 = rearrange(out_e, 'b (head c) h w -> b head h (w c)',
                       head=self.num_heads)  # (B,C2,H/4,W/4)-->(B,k,H/4,d*(W/4))
        k2 = rearrange(out_r, 'b (head c) h w -> b head w (h c)',
                       head=self.num_heads).unsqueeze(0)  # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        v2 = rearrange(out_r, 'b (head c) h w -> b head w (h c)',
                       head=self.num_heads)  # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        q2 = rearrange(out_e, 'b (head c) h w -> b head w (h c)',
                       head=self.num_heads).unsqueeze(0)  # (B,C2,H/4,W/4)-->(B,k,W/4,d*(H/4))
        q1 = rearrange(out_r, 'b (head c) h w -> b head h (w c)',
                       head=self.num_heads).unsqueeze(0) # (B,C2,H/4,W/4)-->(1,B,k,H/4,d*(W/4))
        q1 = self.q_lif(q1).squeeze(0)
        q2 = self.q_lif(q2).squeeze(0)
        k1 = self.k_lif(k1).squeeze(0)
        k2 = self.k_lif(k2).squeeze(0)
        v1 = self.v_lif(v1).squeeze(0)
        v2 = self.v_lif(v2).squeeze(0)


        # 交叉轴注意力：q来自r轴的特征,k/v来自e轴特征
        attn1 = (q1 @ k1.transpose(-2, -1))  # 计算注意力矩阵：(B,k,H/4,d*(W/4)) @ (B,k,d*(W/4),H/4) = (B,k,H/4,H/4)
        attn1 = attn1.softmax(dim=-1)  # softmax归一化注意力矩阵：(B,k,H/4,H/4)-->(B,k,H/4,H/4)
        out3 = (attn1 @ v1) + q1  # 对v矩阵进行加权：(B,k,H/4,H/4) @ (B,k,H/4,d*(W/4)) = (B,k,H/4,d*(W/4))

        # 交叉轴注意力：q来自e轴的特征,k/v来自r轴特征
        attn2 = (q2 @ k2.transpose(-2, -1))  # 计算注意力矩阵：(B,k,W/4,d*(H/4)) @ (B,k,d*(H/4),W/4) = (B,k,W/4,W/4)
        attn2 = attn2.softmax(dim=-1)  # softmax归一化注意力矩阵：(B,k,W/4,W/4)-->(B,k,W/4,W/4)
        out4 = (attn2 @ v2) + q2  # 对v矩阵进行加权：(B,k,W/4,W/4) @ (B,k,W/4,d*(H/4)) = (B,k,W/4,d*(H/4))

        # 将x和y轴上的注意力输出变换为与输入相同的shape
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h,
                         w=w)  # (B,k,H/4,d*(W/4))-->(B,C2,H/4,W/4)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h,
                         w=w)  # (B,k,W/4,d*(H/4))-->(B,C2,H/4,W/4)

        out_event = self.project_out(out3) + self.lamda * self.project_out(out4) + x_e
        out_rgb = self.project_out(out3) + (1 - self.lamda) * self.project_out(out4) + x_r
        x_diff = out_event - out_rgb
        x_diffA = self.catconvA(torch.cat([x_diff, out_event],
                                          dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, out_rgb],
                                          dim=1))
        Event_weight = self.sigmoid(self.convA(x_diffA))
        RGB_weight = self.sigmoid(self.convB(x_diffB))

        xA = Event_weight * out_event
        xB = RGB_weight * out_rgb

        x_fusion = self.catconv(torch.cat([xA, xB],
                                   dim=1)).unsqueeze(0)
        x_fusion = self.x_out(x_fusion).squeeze(0) + x_fusion.squeeze(0)

        return out_event, out_rgb, x_fusion
