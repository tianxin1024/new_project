# 创建自己的model

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        # 确保img_size和patch_size是元组形式
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 使用卷积层来实现patch的提取
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # 确保输入图像的大小与模型的设定相匹配
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 通过卷积层提取patches
        x = self.proj(x)
        # 展平patches并转换维度顺序
        if self.norm is not None:
            x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Conv_Block(nn.Module):
    def __init__(self, in_channels):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Strip_Conv_Block(nn.Module):
    def __init__(self, in_channels):
        super(Strip_Conv_Block, self).__init__()
        self.conv_block = Conv_Block(in_channels)

        self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0:shape[-2]]
        return x

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0:shape[-2]]
        return x.permute(0, 1, 3, 2)

    def forward(self, x):
        x = self.conv_block(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.h_transform(x)
        x3 = self.deconv3(x3)
        x3 = self.inv_h_transform(x3)
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        return x

class Conv_Process(nn.Module):
    def __init__(self, ms_channels, pan_channels, nc):
        super(Conv_Process, self).__init__()
        self.conv_ms = nn.Conv2d(ms_channels, nc, 3, 1, 1)
        self.conv_pan = nn.Conv2d(pan_channels, nc, 3, 1, 1)

    def forward(self, pan, ms):
        return self.conv_pan(pan),  self.conv_ms(ms)


class Transformer_Fusion(nn.Module):
    def __init__(self,nc):
        super(Transformer_Fusion, self).__init__()
        self.conv_trans = nn.Sequential(
            nn.Conv2d(2 * nc, nc, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(nc, nc, kernel_size = 3, stride = 1, padding = 1))

    def batch_index_select(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, in_HSI, in_PAN):
        HSI_unfold  = F.unfold(in_HSI, kernel_size = (3, 3), padding = 1)
        PAN_unfold = F.unfold(in_PAN, kernel_size = (3, 3), padding = 1)
        PAN_unfold = PAN_unfold.permute(0, 2, 1)

        PAN_unfold = F.normalize(PAN_unfold, dim = 2)    # [N, Hr*Wr, C*k*k]
        HSI_unfold  = F.normalize(HSI_unfold, dim = 1)   # [N, C*k*k, H*W]

        R = torch.bmm(PAN_unfold, HSI_unfold)            #[N, Hr*Wr, H*W]
        # 硬注意力映射
        R_star, R_star_arg = torch.max(R, dim = 1)       #[N, H*W]

        ### transfer
        HSI_lv3_unfold = F.unfold(in_PAN, kernel_size = (3, 3), padding = 1)
        T_lv3_unfold = self.batch_index_select(HSI_lv3_unfold, 2, R_star_arg)
        T_lv3 = F.fold(T_lv3_unfold, output_size=in_HSI.size()[-2:], kernel_size=(3, 3), padding=1) / (3. * 3.)
        S = R_star.view(R_star.size(0), 1, in_HSI.size(2), in_HSI.size(3))
        res = self.conv_trans(torch.cat([T_lv3, in_HSI],1)) * S + in_HSI

        return res

class PatchFusion(nn.Module):
    def __init__(self, nc):
        super(PatchFusion, self).__init__()
        self.fuse = Transformer_Fusion(nc)

    def forward(self, HSI_fusion, PAN_fusion):
        ori_copy = HSI_fusion
        B, C, H, W = HSI_fusion.size()
        # [B, C, H, W] -> [B, C * kernel * kernel, ((H+2*padding-kernel) / stride + 1) * ((W * 2*padding - kernel) / stride + 1)]
        HSI_f = F.unfold(HSI_fusion, kernel_size = (24, 24), stride = 8, padding = 8)
        PAN_f = F.unfold(PAN_fusion, kernel_size=(24, 24), stride = 8, padding = 8)
        HSI_f = HSI_f.view(-1, C, 24, 24)
        PAN_f = PAN_f.view(-1, C, 24, 24)
        fusef = self.fuse(HSI_f, PAN_f)
        fusef = fusef.view(B, C * 24 * 24, -1)
        fusef = F.fold(fusef, output_size = ori_copy.size()[-2:], kernel_size=(24, 24), stride = 8, padding = 8)
        return fusef


class MyselfModel(nn.Module):

    def __init__(self):
        super(MyselfModel, self).__init__()
        self.ms_channels = 102
        self.pan_channels = 1
        self.n_feat = 16

        self.pre_conv = Conv_Process(self.ms_channels, self.pan_channels, self.n_feat // 2)
        self.transform_fusion = PatchFusion(self.n_feat // 2)
        self.decoder = Strip_Conv_Block(8)     # 这里修改修改, 暂时默认为8

    def forward(self, HR_PAN , LR_HSI):

        # HR_PAN 四倍下采样
        LR_PAN = F.interpolate(HR_PAN, scale_factor=0.25, mode='bicubic')

        U_LR_PAN = F.interpolate(LR_PAN, scale_factor=4, mode='bicubic')
        U_LR_HSI = F.interpolate(LR_HSI, scale_factor=4, mode='bicubic')

        HR_PAN_A = F.interpolate(HR_PAN, scale_factor=0.5, mode='bicubic')
        LR_PAN_B = F.interpolate(U_LR_PAN, scale_factor=0.5, mode='bicubic')
        LR_HSI_C = F.interpolate(U_LR_HSI, scale_factor=0.5, mode='bicubic')

        HR_PAN_a = F.interpolate(HR_PAN_A, scale_factor=0.5, mode='bicubic')
        LR_PAN_b = F.interpolate(LR_PAN_B, scale_factor=0.5, mode='bicubic')
        LR_HSI_c = F.interpolate(LR_HSI_C, scale_factor=0.5, mode='bicubic')

        HR_PAN_alpha = F.interpolate(HR_PAN_a, scale_factor=0.5, mode='bicubic')
        LR_PAN_beta = F.interpolate(LR_PAN_b, scale_factor=0.5, mode='bicubic')
        LR_HSI_gamma = F.interpolate(LR_HSI_c, scale_factor=0.5, mode='bicubic')

        PAN_A0, HSI_C0 = self.pre_conv(HR_PAN_A, LR_HSI_C)
        PAN_a0, HSI_c0 = self.pre_conv(HR_PAN_a, LR_HSI_c)
        PAN_alpha0, HSI_gamma0 = self.pre_conv(HR_PAN_alpha, LR_HSI_gamma)

        transform_A0_C0 = self.transform_fusion(PAN_A0, HSI_C0)
        transform_a0_c0 = self.transform_fusion(PAN_a0, HSI_c0)
        transform_alpha_gamma = self.transform_fusion(PAN_alpha0, HSI_gamma0)

        print("transform_A0_C0 shape: ", transform_A0_C0.shape)
        print("transform_a0_c0 shape: ", transform_a0_c0.shape)
        print("transform_alpha_gamma shape: ",transform_alpha_gamma.shape)

        scb_AC = self.decoder(transform_A0_C0)
        scb_ac = self.decoder(transform_a0_c0)
        scb_alpha_gamma = self.decoder(transform_alpha_gamma)

        print("scb_AC shape: ", scb_AC.shape)
        print("scb_ac shape: ", scb_ac.shape)
        print("scb_alpha_gamma shape: ", scb_alpha_gamma.shape)

        print(" done!!! ........\n")
    
        # TODO 

        return transform_A0_C0



# ==================================== ADD fix code ==================================== #



if __name__ == "__main__":
    # seed
    torch.manual_seed(0)

    #############################################################
    #################     输入参数       ########################
    #############################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义参数
    batch_size = 1
    hs_channels = 102  # 高光谱通道数
    pan_size = 160     # 全色图像大小
    hs_size = 40       # 多光谱图像大小
    factor = 4         # 上采样因子

    # 创建模拟的多光谱和全色图像数据
    LR_HSI = torch.rand(batch_size, hs_channels, hs_size, hs_size)  # [1, 102, 40, 40]
    HR_PAN = torch.rand(batch_size, 1, pan_size, pan_size)            # [1, 160, 160]

    print("LR_HSI shape: ", LR_HSI.shape)  # [1, 102, 40, 40]
    print("HR_PAN shape: ", HR_PAN.shape)  # [1, 160, 160]

    model = MyselfModel()
    model.to(device)

    # input
    HR_PAN = HR_PAN.to(device)  # [1, 160, 160]
    LR_HSI = LR_HSI.to(device)  # [1, 102, 40, 40]

    output = model(HR_PAN, LR_HSI)

    print("output shape: ", output.shape)




