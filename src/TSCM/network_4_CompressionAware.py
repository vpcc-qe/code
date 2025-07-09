
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from network_common import AustinInception, AustinPyramid
from utils import TCM, GCM
import torch


class SimpleAustinNet_CompressionAware(ME.MinkowskiNetwork):
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
        # 压缩感知的TSCM模块
        class CompressionAwareTSCM(nn.Module):
            def __init__(self, channels, bn_momentum, D):
                super(CompressionAwareTSCM, self).__init__()

                # 基础Austin处理
                self.austin_path = nn.Sequential(
                    AustinInception(channels=channels, bn_momentum=bn_momentum),
                    AustinPyramid(channels=channels, bn_momentum=bn_momentum),
                    AustinInception(channels=channels, bn_momentum=bn_momentum)
                )

                # 特征分解：分离高低频成分
                self.freq_split = ME.MinkowskiConvolution(
                    channels, channels * 2, kernel_size=1, dimension=D)
                self.freq_norm = ME.MinkowskiBatchNorm(channels * 2, momentum=bn_momentum)
                self.freq_act = ME.MinkowskiReLU(inplace=True)

                # 低频处理 - 使用TCM保留结构信息
                self.low_freq_process = TCM(channels)

                # 高频处理 - 使用GCM保留全局上下文
                self.high_freq_process = GCM(channels)

                # 频率重要性注意力
                self.freq_attention = nn.Sequential(
                    ME.MinkowskiGlobalPooling(),
                    ME.MinkowskiConvolution(channels * 2, channels // 2, kernel_size=1, dimension=D),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiConvolution(channels // 2, 2, kernel_size=1, dimension=D),
                    ME.MinkowskiSigmoid()
                )

                # 特征重建
                self.reconstruct = ME.MinkowskiConvolution(channels * 2, channels, kernel_size=1, dimension=D)

                # 融合层
                self.fusion = ME.MinkowskiConvolution(channels * 2, channels, kernel_size=1, dimension=D)
                self.norm = ME.MinkowskiBatchNorm(channels, momentum=bn_momentum)
                self.act = ME.MinkowskiReLU(inplace=True)

            def forward(self, x):
                identity = x

                # Austin路径
                austin_feat = self.austin_path(x)

                # 频率分解
                freq_feat = self.freq_split(x)
                freq_feat = self.freq_norm(freq_feat)
                freq_feat = self.freq_act(freq_feat)

                # 分离高低频
                low_freq, high_freq = torch.split(freq_feat.F, freq_feat.F.shape[1] // 2, dim=1)

                # 创建稀疏张量
                low_freq_tensor = ME.SparseTensor(
                    features=low_freq,
                    coordinate_map_key=freq_feat.coordinate_map_key,
                    coordinate_manager=freq_feat.coordinate_manager
                )
                high_freq_tensor = ME.SparseTensor(
                    features=high_freq,
                    coordinate_map_key=freq_feat.coordinate_map_key,
                    coordinate_manager=freq_feat.coordinate_manager
                )

                # 处理高低频
                low_processed = self.low_freq_process(low_freq_tensor)
                high_processed = self.high_freq_process(high_freq_tensor)

                # 合并处理后的频率成分
                freq_combined = ME.cat(low_processed, high_processed)

                # 生成频率重要性
                pooled = self.freq_attention[0](freq_combined)
                attention = self.freq_attention[1:](pooled)
                attention = ME.MinkowskiBroadcast()(freq_combined, attention)

                # 分离权重
                low_weight, high_weight = torch.split(attention.F, 1, dim=1)

                # 扩展权重到对应通道数
                low_weight = low_weight.expand(-1, channels)
                high_weight = high_weight.expand(-1, channels)

                # 加权频率组件
                weighted_low = ME.SparseTensor(
                    features=low_processed.F * low_weight,
                    coordinate_map_key=low_processed.coordinate_map_key,
                    coordinate_manager=low_processed.coordinate_manager
                )

                weighted_high = ME.SparseTensor(
                    features=high_processed.F * high_weight,
                    coordinate_map_key=high_processed.coordinate_map_key,
                    coordinate_manager=high_processed.coordinate_manager
                )

                # 合并频率处理结果
                freq_result = ME.cat(weighted_low, weighted_high)
                freq_result = self.reconstruct(freq_result)

                # 与Austin路径融合
                combined = ME.cat(austin_feat, freq_result)
                output = self.fusion(combined)
                output = self.norm(output)
                output = self.act(output)

                # 残差连接
                return output + identity

        return CompressionAwareTSCM(channels, bn_momentum, D)

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