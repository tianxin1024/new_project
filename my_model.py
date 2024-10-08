# 创建自己的model

import torch

if __name__ == "__main__":
    # seed
    torch.manual_seed(0)

    #############################################################
    #################     输入参数       ########################
    #############################################################

    # 定义参数
    batch_size = 1
    hs_channels = 102  # 高光谱通道数
    pan_size = 160     # 全色图像大小
    hs_size = 40       # 多光谱图像大小
    factor = 4         # 上采样因子

    # 创建模拟的多光谱和全色图像数据
    LR_HSI = torch.rand(batch_size, hs_channels, hs_size, hs_size)  # [1, 102, 40, 40]
    HR_PAN = torch.rand(batch_size, pan_size, pan_size)            # [1, 160, 160]

    print("LR_HSI shape: ", LR_HSI.shape)  # [1, 102, 40, 40]
    print("HR_PAN shape: ", HR_PAN.shape)  # [1, 160, 160]




