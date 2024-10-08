# 创建自己的model

import torch
import torch.nn as nn


class MyselfModel(nn.Module):

    def __init__(self):
        super(MyselfModel, self).__init__()
        self.factor = 4

    def forward(self, ms_image, pan_image):
        pass

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
    HR_PAN = torch.rand(batch_size, pan_size, pan_size)            # [1, 160, 160]

    print("LR_HSI shape: ", LR_HSI.shape)  # [1, 102, 40, 40]
    print("HR_PAN shape: ", HR_PAN.shape)  # [1, 160, 160]

    model = MyselfModel()
    model.to(device)

    # input
    MS_image = LR_HSI.to(device)
    PAN_image = HR_PAN.to(device)

    output = model(MS_image, PAN_image)

    print("output shape: ", output.shape)




