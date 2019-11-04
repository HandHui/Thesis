import torch.nn as nn
from torch.nn.utils import weight_norm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, residual=True):
        super(ConvBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))           ###一维卷积，wieght_norm 防止过拟合
        self.activate = nn.ReLU()
        self.residual = residual
        self.down_sample = nn.Conv1d(in_channels, out_channels, 1) if residual and in_channels != out_channels else None  ### 残差为真且输入不等于输出（即输入行数与卷积核不同），则利用kernel_size为1的卷积核再卷积（应该相当于池化）
		                                                                                                                  ### 否则，池化为空
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight.data, mode='fan_in', nonlinearity='relu')   ### 初始化的过程，为卷积层与池化层的bias设置初始值
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0)
        if self.down_sample is not None:
            nn.init.kaiming_uniform_(self.down_sample.weight.data, mode='fan_in', nonlinearity='relu')
            if self.down_sample.bias is not None:
                self.down_sample.bias.data.fill_(0)

    def forward(self, inputs):
        output = self.activate(self.conv(inputs))
        if self.residual:
            output += self.down_sample(inputs) if self.down_sample else inputs                  ###单卷积核 卷积 输入
			                                                                                    ### kernel_size不同的话，大小也不同啊
																								### 不能加吧，即使能加，为什么要加？？
																								
        return output


class ConvNet(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.5, dilated=False, residual=True):
        super(ConvNet, self).__init__()         ###super(类名,self).__init__()
        num_levels = len(channels)-1
        layers = []
        for i in range(num_levels):
            in_channels = channels[i]
            out_channels = channels[i+1]
            dilation = kernel_size ** i if dilated else 1
            padding = (kernel_size - 1) // 2 * dilation
            layers += [ConvBlock(in_channels, out_channels, kernel_size,
                                 padding=padding, dilation=dilation, residual=residual), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers[:-1])      ### 加*号 解包

    def forward(self, inputs):
        return self.net(inputs)
'''
Sequential(
  (0): ConvBlock(
    (conv): Conv1d(200, 200, kernel_size=(3,), stride=(1,), padding=(1,))
    (activate): ReLU()
  )
)
con=ConvNet([200,200])
print(con.net)
'''
