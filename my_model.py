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


class MyselfModel(nn.Module):

    def __init__(self):
        super(MyselfModel, self).__init__()
        self.n_select_bands = 102
        self.n_bands = 102
        self.channel_P = 1
        self.channel_H = 102
        self.patch_embed_P = PatchEmbed(img_size=80, patch_size=16, in_chans=self.channel_P, embed_dim=768)
        self.patch_embed_H = PatchEmbed(img_size=80, patch_size=16, in_chans=self.channel_H, embed_dim=768)

        self.pre_conv_P = nn.Conv2d(in_channels=self.channel_P, out_channels=self.channel_P, kernel_size=3, padding=1)
        self.pre_conv_H = nn.Conv2d(in_channels=self.channel_H, out_channels=self.channel_H, kernel_size=3, padding=1)


    def forward(self, HR_PAN , LR_HSI):

        P_B, P_C, P_H, P_W = HR_PAN.shape
        H_B, H_C, H_H, H_W = LR_HSI.shape
        # HR_PAN 四倍下采样
        LR_PAN = F.interpolate(HR_PAN, scale_factor=0.25, mode='nearest')
        print(LR_PAN.shape)

        U_LR_PAN = F.interpolate(LR_PAN, scale_factor=4, mode='nearest')
        U_LR_HSI = F.interpolate(LR_HSI, scale_factor=4, mode='nearest')

        HR_PAN_A = F.interpolate(HR_PAN, scale_factor=0.5, mode='nearest')
        LR_PAN_B = F.interpolate(U_LR_PAN, scale_factor=0.5, mode='nearest')
        LR_HSI_C = F.interpolate(U_LR_HSI, scale_factor=0.5, mode='nearest')

        HR_PAN_a = F.interpolate(HR_PAN_A, scale_factor=0.5, mode='nearest')
        LR_PAN_b = F.interpolate(LR_PAN_B, scale_factor=0.5, mode='nearest')
        LR_HSI_c = F.interpolate(LR_HSI_C, scale_factor=0.5, mode='nearest')

        HR_PAN_alpha = F.interpolate(HR_PAN_a, scale_factor=0.5, mode='nearest')
        LR_PAN_beta = F.interpolate(LR_PAN_b, scale_factor=0.5, mode='nearest')
        LR_HSI_gamma = F.interpolate(LR_HSI_c, scale_factor=0.5, mode='nearest')

        HR_PAN_A0 = self.pre_conv_P(HR_PAN_A)
        HR_PAN_a0 = self.pre_conv_P(HR_PAN_a)
        HR_PAN_alpha0 = self.pre_conv_P(HR_PAN_alpha)

        LR_HSI_C0 = self.pre_conv_H(LR_HSI_C)
        LR_HSI_c0 = self.pre_conv_H(LR_HSI_c)
        LR_HSI_gamma0 = self.pre_conv_H(LR_HSI_gamma)

        HR_PAN_A_patch_P = self.patch_embed_P(HR_PAN_A)
        LR_HSI_C_patch_H = self.patch_embed_H(LR_HSI_C)

        print("........\n")
    
        # TODO 



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




