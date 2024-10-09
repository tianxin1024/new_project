# 创建自己的model

import torch
import torch.nn as nn
# import ipdb

class MyselfModel(nn.Module):

    def __init__(self):
        super(MyselfModel, self).__init__()
        self.factor = 4

        self.n_select_bands = 102
        self.n_bands = 102

        self.Downsample = nn.Sequential(
            nn.Conv2d(self.n_select_bands, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, self.n_bands, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.n_bands, self.n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


    # def forward(self, ms_image, pan_image):
    #     # ipdb.set_trace()
    #     ms_a = self.Downsample(ms_image)
    #     print("ms_a shape: ", ms_a.shape)
    #
    #     return ms_a
    def forward(self, MS_image, PAN_image):
        #使用上采样因子 self.factor 来上采样 MS_image
        resized_MS_image = nn.functional.interpolate(MS_image, size=(
        self.factor * MS_image.size(2), self.factor * MS_image.size(3)), mode='bilinear', align_corners=False)
        return resized_MS_image
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
    HR_PAN = torch.rand(batch_size, pan_size, pan_size)            # [1, 160, 160]

    print("LR_HSI shape: ", LR_HSI.shape)  # [1, 102, 40, 40]
    print("HR_PAN shape: ", HR_PAN.shape)  # [1, 160, 160]

    model = MyselfModel()
    model.to(device)

    # input
    HS_image = LR_HSI.to(device)
    PAN_image = HR_PAN.to(device)

    output = model(HS_image, PAN_image)

    print("output shape: ", output.shape)




