import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from network_common import AustinInception, AustinPyramid
from utils import TCM, GCM
import torch


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
        self.block1 = self.make_layer(CHANNELS[1], "low", bn_momentum, D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = ME.MinkowskiBatchNorm(CHANNELS[2], momentum=bn_momentum)
        self.block2 = self.make_layer(CHANNELS[2], "mid", bn_momentum, D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = ME.MinkowskiBatchNorm(CHANNELS[3], momentum=bn_momentum)
        self.block3 = self.make_layer(CHANNELS[3], "high", bn_momentum, D)

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
        self.block3_tr = self.make_layer(TR_CHANNELS[2], "mid", bn_momentum, D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=3,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = ME.MinkowskiBatchNorm(TR_CHANNELS[1], momentum=bn_momentum)
        self.block2_tr = self.make_layer(TR_CHANNELS[1], "low", bn_momentum, D)

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

    def make_layer(self, channels, layer_type, bn_momentum, D):
        # 根据层级定制TSCM模块
        class LayerSpecificTSCM(nn.Module):
            def __init__(self, channels, layer_type, bn_momentum, D):
                super(LayerSpecificTSCM, self).__init__()

                # 保留原始Austin块作为基础
                self.austin1 = AustinInception(channels=channels, bn_momentum=bn_momentum)
                self.austin2 = AustinPyramid(channels=channels, bn_momentum=bn_momentum)
                self.austin3 = AustinInception(channels=channels, bn_momentum=bn_momentum)

                if layer_type == "low":  # 16通道，底层特征
                    # 底层使用TCM增强局部特征
                    self.tcm = TCM(channels)
                    # 低层级更侧重局部特征
                    self.alpha = nn.Parameter(torch.ones(1) * 0.7)  # 偏向增强局部特征
                    self.beta = nn.Parameter(torch.ones(1) * 0.3)

                elif layer_type == "mid":  # 32通道，中层特征
                    # 中层同时使用TCM和GCM
                    self.tcm = TCM(channels)
                    self.gcm = GCM(channels)
                    # 平衡局部和全局
                    self.alpha = nn.Parameter(torch.ones(1) * 0.5)  # 平衡权重
                    self.beta = nn.Parameter(torch.ones(1) * 0.5)

                else:  # 64通道，高层特征
                    # 高层使用GCM增强全局特征
                    self.gcm = GCM(channels)
                    # 高层级更侧重全局特征
                    self.alpha = nn.Parameter(torch.ones(1) * 0.3)  # 偏向增强全局特征
                    self.beta = nn.Parameter(torch.ones(1) * 0.7)

                # 最终融合
                self.fusion = ME.MinkowskiConvolution(channels, channels, kernel_size=1, dimension=D)
                self.relu = ME.MinkowskiReLU(inplace=True)

                self.layer_type = layer_type

            def forward(self, x):
                identity = x

                # Austin基础块处理
                main = self.austin1(x)
                main = self.austin2(main)
                main = self.austin3(main)

                # 层级特定增强
                if self.layer_type == "low":
                    enhanced = self.tcm(x)
                elif self.layer_type == "mid":
                    # 中层同时使用TCM和GCM，并融合
                    tcm_out = self.tcm(x)
                    gcm_out = self.gcm(x)
                    enhanced = tcm_out + gcm_out
                else:  # high
                    enhanced = self.gcm(x)

                # 归一化权重
                total = self.alpha + self.beta + 1e-6
                alpha_norm = self.alpha / total
                beta_norm = self.beta / total

                # 修复：将标量张量转换为与 SparseTensor 兼容的形式
                # 创建与 main 和 enhanced 具有相同坐标和特征维度的 SparseTensor
                alpha_tensor = ME.SparseTensor(
                    features=torch.ones_like(main.F) * alpha_norm.item(),
                    coordinate_map_key=main.coordinate_map_key,
                    coordinate_manager=main.coordinate_manager
                )

                beta_tensor = ME.SparseTensor(
                    features=torch.ones_like(enhanced.F) * beta_norm.item(),
                    coordinate_map_key=enhanced.coordinate_map_key,
                    coordinate_manager=enhanced.coordinate_manager
                )

                # 融合特征
                fused = alpha_tensor * main + beta_tensor * enhanced
                out = self.fusion(fused)

                # 残差连接
                out = out + identity
                out = self.relu(out)

                return out

        return LayerSpecificTSCM(channels, layer_type, bn_momentum, D)

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