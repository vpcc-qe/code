
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from network_common import AustinInception, AustinPyramid
from utils import TCM, GCM  
import torch
import time

class SimpleAustinNet(ME.MinkowskiNetwork):
    # 通道配置
    CHANNELS = [None, 16, 32, 64]
    TR_CHANNELS = [None, 16, 32, 64]
    BLOCK_1 = AustinInception
    BLOCK_2 = AustinPyramid

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 bn_momentum=0.1,
                 D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        BLOCK_1 = self.BLOCK_1
        BLOCK_2 = self.BLOCK_2

        # 编码器部分
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = ME.MinkowskiBatchNorm(CHANNELS[1], momentum=bn_momentum)
        self.block1 = self.make_layer(CHANNELS[1], bn_momentum, D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(CHANNELS[2], momentum=bn_momentum)
        self.block2 = self.make_layer(CHANNELS[2], bn_momentum, D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = ME.MinkowskiBatchNorm(CHANNELS[3], momentum=bn_momentum)
        self.block3 = self.make_layer(CHANNELS[3], bn_momentum, D)

        # 解码器部分
        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[2], momentum=bn_momentum)
        self.block3_tr = self.make_layer(TR_CHANNELS[2], bn_momentum, D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[1], momentum=bn_momentum)
        self.block2_tr = self.make_layer(TR_CHANNELS[1], bn_momentum, D)

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=TR_CHANNELS[1],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)

    def make_layer(self, channels, bn_momentum, D):
        class BaselineTSCM(nn.Module):
            def __init__(self, channels, bn_momentum, D):
                super(BaselineTSCM, self).__init__()

                # 直接使用导入的TCM和GCM模块
                self.tcm = TCM(channels)
                self.gcm = GCM(channels)

                # 原始Austin路径
                self.austin_path = nn.Sequential(
                    AustinInception(channels=channels, bn_momentum=bn_momentum),
                    AustinPyramid(channels=channels, bn_momentum=bn_momentum),
                    AustinInception(channels=channels, bn_momentum=bn_momentum)
                )

                # 固定权重系数 - 不使用自适应权重，而是使用固定的系数
#                 self.alpha = 0.33  # Austin路径权重
#                 self.beta = 0.33  # TCM路径权重
#                 self.gamma = 0.34  # GCM路径权重

                self.alpha = 0.2  # Austin路径权重
                self.beta = 0.4 # TCM路径权重
                self.gamma = 0.4 # GCM路径权重
                
                # 最终融合层
                self.fusion = ME.MinkowskiConvolution(channels, channels, kernel_size=1, dimension=D)

            def forward(self, x):
                identity = x

                # 三路径特征提取
                austin_feat = self.austin_path(x)
                tcm_feat = self.tcm(x)
                gcm_feat = self.gcm(x)
#                 print(austin_feat)
#                 print(tcm_feat)
#                 print(gcm_feat)


                # 创建权重张量
                alpha_tensor = ME.SparseTensor(
                    features=torch.ones_like(austin_feat.F) * self.alpha,
                    coordinate_map_key=austin_feat.coordinate_map_key,
                    coordinate_manager=austin_feat.coordinate_manager
                )

                beta_tensor = ME.SparseTensor(
                    features=torch.ones_like(tcm_feat.F) * self.beta,
                    coordinate_map_key=tcm_feat.coordinate_map_key,
                    coordinate_manager=tcm_feat.coordinate_manager
                )

                gamma_tensor = ME.SparseTensor(
                    features=torch.ones_like(gcm_feat.F) * self.gamma,
                    coordinate_map_key=gcm_feat.coordinate_map_key,
                    coordinate_manager=gcm_feat.coordinate_manager
                )

                # 使用点乘操作进行加权
                weighted_austin = austin_feat * alpha_tensor
                weighted_tcm = tcm_feat * beta_tensor
                weighted_gcm = gcm_feat * gamma_tensor

                # 加权求和
                weighted = weighted_austin + weighted_tcm + weighted_gcm

                # 最终融合
                output = self.fusion(weighted)

                # 残差连接
                return output + identity

        return BaselineTSCM(channels, bn_momentum, D)

    def forward(self, x):
        # 编码器路径
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)

        out_s3 = self.conv3(out)
        out_s3 = self.norm3(out_s3)
        out_s3 = self.block3(out_s3)
        out = MEF.relu(out_s3)

        # 解码器路径
        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, out_s2)
        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = MEF.relu(out)

        out = out_s1_tr + out_s1
        out = self.conv1_tr(out)
        out = MEF.relu(out)

        out_cls = self.final(out)
        out_cls = out_cls + x
        return out_cls