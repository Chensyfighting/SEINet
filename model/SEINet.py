from model.module.neuron import LIFAct
from model.module.SDSA import MS_Block
from spikingjelly.activation_based import layer, functional
from timm.models.layers import trunc_normal_
from submodule import *
import SCMF

__all__ = ["SEINet"]


class Spike_Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = layer.Conv2d(nIn, nOut, kernel_size=kSize,
                                 stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_lif = BNLIF(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_lif(output)

        return output


class MSResNet(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.lif = LIFAct(step=1)
        self.conv1 = Spike_Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)
        self.bn = layer.BatchNorm2d(nIn, eps=1e-3)
        self.lif = LIFAct(step=1)
        self.conv2 = Spike_Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)
        self.bn = layer.BatchNorm2d(nIn, eps=1e-3)

    def forward(self, input):
        output = self.lif(input)
        output = self.conv1(output)
        output = self.bn(output)
        output = self.lif(output)
        output = self.conv2(output)
        output = self.bn(output)

        return output + input


class BNLIF(nn.Module):

    def __init__(self, nIn, lif=True):
        super().__init__()
        self.bn = layer.BatchNorm2d(nIn, eps=1e-3)
        self.lif = LIFAct(step=1)
        self.lif_acti = lif

    def forward(self, input):
        output = self.bn(input)

        if self.lif_acti:
            output = self.lif(output)

        return output



class BasicInterpolate(nn.Module):

    def __init__(self, size, mode, align_corners):
        super(BasicInterpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        times_window, batch_size = x.shape[0], x.shape[1]
        # [t,b,c,h,w,]->[t*b,c,h,w]
        x = x.reshape(-1, *x.shape[2:])
        x = F.interpolate(x, size=self.size, mode=self.mode,
                          align_corners=self.align_corners)
        # [t*b,c,h,w]->[t,b,c,h,w]
        x = x.view(times_window, batch_size, *x.shape[1:])
        return x


class Spike_LMSBModule(nn.Module):

    def __init__(self, nIn, d=1, kSize=3, dkSize=3, groups=1):
        super().__init__()

        self.ca_groups = groups
        self.bn_lif_1 = BNLIF(nIn)
        self.conv1x1_in = Spike_Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Spike_Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Spike_Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)
        self.dconv3x1 = Spike_Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Spike_Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ca11 = eca_layer(nIn // 2)
        self.ddconv3x1 = Spike_Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1),
                                    groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Spike_Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d),
                                    groups=nIn // 2, bn_acti=True)
        self.ca22 = eca_layer(nIn // 2)

        self.bn_lif_2 = BNLIF(nIn // 2)
        self.conv1x1 = Spike_Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)

        self.conv3x3_1 = Spike_Conv(nIn // 2, nIn // 2, kSize, 1, padding=1, dilation=1, groups=nIn // 2, bn_acti=True)
        self.conv3x3_2 = Spike_Conv(nIn // 2, nIn // 2, dkSize, 1, padding=1, dilation=1, groups=nIn // 2, bn_acti=True)

    def forward(self, input):
        output = self.bn_lif_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_lif_2(output)
        output = self.conv1x1(output)
        output = output + input

        return output



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [T,N,C,H,W] -> [T,N,g,C/g,H,W] -> [T,N,C/g,g,H,w] -> [T,N,C,H,W]"""
        T, N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(T, N, g, int(C / g), H, W).permute(0, 1, 3, 2, 4, 5).contiguous().view(T, N, C, H, W)


class DownSamplingBlock(nn.Module):
    # membrane potential
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Spike_Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = layer.MaxPool2d(2, stride=2)
        self.bnlif = BNLIF(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            # [t,b,c,h,w]
            output = torch.cat([output, max_pool], 2)

        return output


class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = layer.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = layer.BatchNorm2d(noutput, eps=1e-3)
        self.lif = LIFAct(step=1)

    def forward(self, input):
        output = self.conv(input)
        return output


class PA(nn.Module):

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = layer.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class eca_layer(nn.Module):

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = layer.AdaptiveAvgPool2d(1)  # GAP，全局平均池化，对H和W进行全局平均，所以[t,b,c,h,w]→[t,b,c,1,1]
        self.conv = layer.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        t, b, c, h, w = x.size()
        y = self.avg_pool(x)  # [t,b,c,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class LFE(nn.Module):
    def __init__(self, nIn, nOut, kSize, r, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        self.dconv3x1 = Spike_Conv(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.dconv1x3 = Spike_Conv(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        if self.bn_acti:
            self.bn_lif = BNLIF(nOut)

        self.GAP = layer.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = Spike_Conv(nIn, nIn // r, kSize=1, stride=1, padding=0, bn_acti=True)
        self.conv1x1_2 = Spike_Conv(nIn // r, nIn, kSize=1, stride=1, padding=0, bn_acti=False)

        # b,c,h,w
        self.conv3x3 = Spike_Conv(2, 1, kSize=3, stride=1, padding=1, bn_acti=False)
        self.bn = layer.BatchNorm2d(1, eps=1e-3)

        self.conv1x1 = Spike_Conv(nIn, nIn, kSize=1, stride=1, padding=0, bn_acti=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_lif(output)
        # channel
        output1 = self.GAP(output)
        output1 = self.conv1x1_1(output1)
        output1 = self.sigmoid(self.conv1x1_2(output1))
        output1 = input * output1
        # spatial
        mean = torch.mean(output, dim=2, keepdim=True)
        max, _ = torch.max(output, dim=2, keepdim=True)
        output2 = torch.cat((mean, max), dim=2)
        output2 = self.sigmoid(self.bn(self.conv3x3(output2)))
        output2 = input * output2
        # fuse
        output = output + self.conv1x1(output1) + self.conv1x1(output2)

        return output


class SEINet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=3, block_3=3, block_4=1, block_5=1, block_6=1, ohem=True,
                 augment=True):
        super().__init__()
        self.augment = augment
        self.ohem = ohem

        self.init_conv = nn.Sequential(
            Spike_Conv(5, 32, 3, 1, padding=1, bn_acti=True),
            Spike_Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Spike_Conv(32, 32, 3, 2, padding=1, bn_acti=False),
        )
        self.MSResNet_0 = MSResNet(32)
        self.downsample_1 = DownSamplingBlock(32, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), Spike_LMSBModule(64, d=2))
        self.MSResNet_1 = MSResNet(64)
        self.conv_RGB1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.FRM1 = SCMF.SCMF(64, 8, LayerNorm_type='WithBias',in_channel=64, out_channel=64)
        dilation_block_2 = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
        self.downsample_2 = DownSamplingBlock(64, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        Spike_LMSBModule(128, d=dilation_block_2[i], groups=1))
        self.MSResNet_2 = MSResNet(128)
        self.FRM2 = SCMF.SCMF(128, 8, LayerNorm_type='WithBias',in_channel=128, out_channel=128)
        dilation_block_3 = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
        self.downsample_3 = DownSamplingBlock(128, 32)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        Spike_LMSBModule(32, d=dilation_block_3[i]))
        self.MSResNet_3 = MSResNet(32)
        self.FRM3 = SCMF.SCMF(32, 8, LayerNorm_type='WithBias',in_channel=32, out_channel=32)
        self.transformer1 = MS_Block(dim=32, num_heads=2)
        self.transformer2 = MS_Block(dim=32, num_heads=2)
        # DECODER
        dilation_block_4 = [2, 2, 2]
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
            self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                        Spike_LMSBModule(32, d=dilation_block_4[i]))
        self.upsample_1 = UpsampleingBlock(32, 16)
        self.MSResNet_4 = MSResNet(16)

        dilation_block_5 = [2, 2, 2]
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        Spike_LMSBModule(16, d=dilation_block_5[i]))
        self.upsample_2 = UpsampleingBlock(16, 16)
        self.MSResNet_5 = MSResNet(16)

        dilation_block_6 = [2, 2, 2]
        self.DAB_Block_6 = nn.Sequential()
        for i in range(0, block_6):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        Spike_LMSBModule(16, d=dilation_block_6[i]))
        self.upsample_3 = UpsampleingBlock(16, 16)
        self.MSResNet_6 = MSResNet(16)

        self.PA1 = PA(nf=16)
        self.PA2 = PA(nf=16)
        self.PA3 = PA(nf=16)

        self.LFE1 = LFE(nIn=64, nOut=16, kSize=3, r=2, bn_acti=False)
        self.LFE2 = LFE(nIn=128, nOut=16, kSize=3, r=2, bn_acti=False)
        self.LFE3 = LFE(nIn=32, nOut=32, kSize=3, r=2, bn_acti=False)

        self.classifier = nn.Sequential(Spike_Conv(16, classes, 1, 1, padding=0))

        self.apply(self.trunc_init)

    def trunc_init(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Conv2d, layer.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, events_input, RGBs_input):
        output0_e = events_input.unsqueeze(0)
        output0_e = self.init_conv(output0_e)
        output0_e = self.MSResNet_0(output0_e)
        events_input = events_input.unsqueeze(0)
        output1_0_e = self.downsample_1(output0_e)
        output1_e = self.DAB_Block_1(output1_0_e)
        output1_e = self.MSResNet_1(output1_e)

        output0_R = self.conv_RGB1(RGBs_input)
        output0_R = output0_R.unsqueeze(0)
        output1_R = self.downsample_1(output0_R)
        output1_e = output1_e.squeeze(0)
        output1_R = output1_R.squeeze(0)
        outFRM_event_1, outFRM_RGB_1, output_fusion_1 = self.FRM1(output1_e, output1_R)
        outFRM_event_1 = outFRM_event_1.unsqueeze(0)
        outFRM_RGB_1 = outFRM_RGB_1.unsqueeze(0)

        output2_0_e = self.downsample_2(outFRM_event_1)
        output2_e = self.DAB_Block_2(output2_0_e)
        output2_e = self.MSResNet_2(output2_e)
        output2_R = self.downsample_2(outFRM_RGB_1)
        output2_e = output2_e.squeeze(0)
        output2_R = output2_R.squeeze(0)
        outFRM_event_2, outFRM_RGB_2, output_fusion_2 = self.FRM2(output2_e, output2_R)
        outFRM_event_2 = outFRM_event_2.unsqueeze(0)
        outFRM_RGB_2 = outFRM_RGB_2.unsqueeze(0)

        output3_0_e = self.downsample_3(outFRM_event_2)
        output3_e = self.DAB_Block_3(output3_0_e)
        output3_e = self.MSResNet_3(output3_e)
        output3_R = self.downsample_3(outFRM_RGB_2)
        output3_e = output3_e.squeeze(0)
        output3_R = output3_R.squeeze(0)
        outFRM_event_3, outFRM_RGB_3, output_fusion_3 = self.FRM3(output3_e, output3_R)
        output_tf = self.transformer1(output_fusion_3)
        output_tf = self.transformer2(output_tf)
        output_tf = self.DAB_Block_4(output_tf)
        output4 = self.upsample_1(self.LFE1(output_tf + output_fusion_3))
        # output4 = self.upsample_1(output_tf + self.FE3(self.LC3(output_fusion_3)))
        output4 = self.MSResNet_4(output4)
        output5 = self.DAB_Block_5(output4)
        temp = self.FE2(self.LC2(output_fusion_2))
        output5 = BasicInterpolate(size=temp.size()[3:], mode='bilinear', align_corners=False)(output5)
        output5 = self.upsample_2(output5 + temp)
        output5 = self.MSResNet_5(output5)
        if self.augment:
            early_out = BasicInterpolate(size=events_input.size()[3:], mode='bilinear', align_corners=False)(
                output5)
            early_out = self.classifier(early_out)

        output6 = self.DAB_Block_6(output5)
        output6 = self.upsample_3(output6 + self.FE1(self.LC1(output_fusion_1)))
        output6 = self.PA3(output6)

        out = BasicInterpolate(size=events_input.size()[3:], mode='bilinear', align_corners=False)(
            output6)
        out = self.MSResNet_6(out)
        out = self.classifier(out)

        if self.augment & self.ohem:
            early_out = early_out.mean(dim=0)
            out = out.mean(dim=0)
            return [out, early_out]
        elif self.augment:
            early_out = early_out.mean(dim=0)
            return early_out
        else:
            out = out.mean(dim=0)
            return out
