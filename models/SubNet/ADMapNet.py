"""ADMapNet network definition."""

from models.Layers.ADMapNet_Layers import *


class ADMapNet(nn.Module):

    def __init__(self, sources_num):
        super().__init__()
        self.sources_num = sources_num
        self.conv1 = conv1()
        self.conv2_x = conv2_x()
        self.conv3_x = conv3_x()
        self.conv4_x = conv4_x()
        self.conv5_x = conv5_x()
        self.avg_pool = global_avg_pool()
        self.fc = fc_layer(self.sources_num)

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        x_complex = torch.stack([x_real, x_imag], dim=1)
        x = self.conv1(x_complex)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.squeeze(dim=[2, 3])
        x = self.fc(x)

        return x
