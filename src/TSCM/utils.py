import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from tr_block import CodedVTRBlock


class TCM(nn.Module):
    def __init__(self, input_dim, device=None):
        super(TCM, self).__init__()
        self.dim = input_dim
        
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        self.conv1 = ME.MinkowskiConvolution(
            input_dim,
            input_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3).to(device)
        self.conv2 = ME.MinkowskiConvolution(
            input_dim,
            input_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3).to(device)
        self.trans = CodedVTRBlock(input_dim//2, input_dim//2, device=device)
        # self.trans=resblock_dia(input_dim//2,input_dim//2,1)

        self.resblock1 = resblock(input_dim//2, input_dim//2, 1, device)
        
        # 将整个模块移动到指定设备
        self.to(device)

    def forward(self,x):
        conv_x,trans_x=torch.split(self.conv1(x).F,(self.dim//2,self.dim//2),dim=1)
        conv_x=ME.SparseTensor(
                features=conv_x,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
                device=x.device)
        conv_x=self.resblock1(conv_x)+conv_x
        trans_x=ME.SparseTensor(
                features=trans_x,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
                device=x.device)
        trans_x=self.trans(trans_x)
        res=self.conv2(ME.cat(conv_x,trans_x))
        x=x+res
        return x

class GCM(nn.Module):
    def __init__(self, input_dim, device=None):
        super(GCM, self).__init__()
        self.dim = input_dim
        
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        self.conv1 = ME.MinkowskiConvolution(
            input_dim,
            input_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3).to(device)
        self.conv2 = ME.MinkowskiConvolution(
            input_dim,
            input_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3).to(device)
        self.glob = glob_feat(input_dim//2, device)
        self.resblock1 = resblock(input_dim//2, input_dim//2, 1, device)
        
        # 将整个模块移动到指定设备
        self.to(device)

    def forward(self,x):
        conv_x,glo_x=torch.split(self.conv1(x).F,(self.dim//2,self.dim//2),dim=1)
        conv_x=ME.SparseTensor(
                features=conv_x,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
                device=x.device)
        conv_x=self.resblock1(conv_x)+conv_x
        glo_x = ME.SparseTensor(
            features=glo_x,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)
        glo_x=self.glob(glo_x)

        res=self.conv2(ME.cat(conv_x,glo_x))
        x=x+res
        return x



class resblock(nn.Module):
    def __init__(self, input_dim, output_dim, stride, device=None):
        super(resblock, self).__init__()
        
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        self.conv1 = ME.MinkowskiConvolution(
            input_dim,
            output_dim,
            kernel_size=3,
            stride=stride,
            bias=True,
            dimension=3).to(device)
        self.conv2 = ME.MinkowskiConvolution(
            output_dim,
            output_dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3).to(device)
        if stride != 1:
            self.skip = ME.MinkowskiConvolution(
                input_dim,
                output_dim,
                kernel_size=1,
                stride=stride,
                bias=True,
                dimension=3).to(device)
        else:
            self.skip = None
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # 将整个模块移动到指定设备
        self.to(device)

    def forward(self,x):
        identity = x

        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(x)
        out=out+identity
        return out

class resblock_dia(nn.Module):
    def __init__(self, input_dim, output_dim, stride, device=None):
        super(resblock_dia, self).__init__()
        
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        self.conv1 = ME.MinkowskiConvolution(
            input_dim,
            output_dim,
            kernel_size=3,
            dilation=2,
            stride=stride,
            bias=True,
            dimension=3).to(device)
        self.conv2 = ME.MinkowskiConvolution(
            output_dim,
            output_dim,
            kernel_size=3,
            dilation=2,
            stride=1,
            bias=True,
            dimension=3).to(device)
        if stride != 1:
            self.skip = ME.MinkowskiConvolution(
                input_dim,
                output_dim,
                kernel_size=1,
                dilation=2,
                stride=stride,
                bias=True,
                dimension=3).to(device)
        else:
            self.skip = None
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # 将整个模块移动到指定设备
        self.to(device)

    def forward(self,x):
        identity = x

        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(x)
        out=out+identity
        return out

class resblock_up(nn.Module):
    def __init__(self, input_dim, output_dim, stride, device=None):
        super(resblock_up, self).__init__()
        
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        self.conv1 = ME.MinkowskiConvolutionTranspose(
            input_dim,
            output_dim,
            kernel_size=3,
            stride=stride,
            bias=True,
            dimension=3).to(device)
        self.conv2 = ME.MinkowskiConvolution(
            output_dim,
            output_dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3).to(device)
        if stride != 1:
            self.skip = ME.MinkowskiConvolution(
                input_dim,
                output_dim,
                kernel_size=1,
                stride=stride,
                bias=True,
                dimension=3).to(device)
        else:
            self.skip = None
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # 将整个模块移动到指定设备
        self.to(device)

    def forward(self,x):
        identity = x

        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(x)
        out=out+identity
        return out

def ME_conv(inc,outc,kernel_size=3,stride=1):
    return ME.MinkowskiConvolution(inc,outc,kernel_size=kernel_size,stride=stride,bias=True,dimension=3)

def ste_round(x):
    return torch.round(x)-x.detach()+x

class glob_feat(nn.Module):
    def __init__(self, dim, device=None):
        super(glob_feat, self).__init__()
        
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        self.conv1 = ME.MinkowskiConvolution(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3).to(device)
        self.conv2 = ME.MinkowskiConvolution(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3).to(device)
        self.conv3 = ME.MinkowskiConvolution(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3).to(device)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # 将整个模块移动到指定设备
        self.to(device)

    def forward(self,x):
        out1_space=self.relu(self.conv1(x))
        out2_channel=self.relu(self.conv2(x))
        all=[]
        feats1=out1_space.decomposed_features
        feats2=out2_channel.decomposed_features

        for i in range(len(feats1)):
            final_encoding = torch.matmul(feats1[i].mean(dim=-1,keepdim=True), feats2[i].mean(dim=0,keepdim=True))
            final_encoding = torch.sqrt(final_encoding+1e-12) # B,C/8,N
            all.append(final_encoding)
        final_encoding=ME.SparseTensor(
            features=torch.cat(all,dim=0)+out1_space.F+out2_channel.F,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
            device=x.device)
        final_encoding =self.relu(self.conv3(final_encoding))
        final_encoding=x-final_encoding#zui hou meijia relu
        return self.relu(final_encoding)




