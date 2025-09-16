import torch
from torch import nn
from torch.nn import Module
from torch.nn import init
import torch.nn.functional as F


class CPCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)

        max_out = self.se(max_result)
        avg_out = self.se(avg_result)

        output = self.sigmoid(max_out + avg_out)

        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        
        out = out * self.sa(out)

        # Residual connection
        return out + residual

# 这段代码实现了一个位置感知的最大池化函数 position_aware_max，它通过滑动窗口的方式在一个张量上进行局部最大池化操作。
# 具体来说，它会根据输入张量的形状、窗口大小和池化的维度，动态选择子区域并计算其最大值。函数的主要作用是进行位置感知的最大池化，这有助于模型捕捉到特征图中的局部最强信号。下面我们详细解释每个部分的功能。
def position_aware_max(score, h, w, part=3, dim=0):
    b, c, H, W = score.shape
    part_rows = h // part
    half_rows = part_rows // 2
    max_value = torch.zeros((b, c, H))
    for i in range(h):
        upper = i - half_rows
        low = i + half_rows
        if upper < 0:
            upper = 0
            low = upper + part_rows
        if low > h:
            low = h
            upper = low - part_rows

        begin_index = upper * w
        end_index = low * w
        if (dim == 0):
            sub_score = score[:, :, begin_index:end_index, i * w:(i + 1) * w]
        elif (dim == 1):
            sub_score = score[:, :, i * w:(i + 1) * w, begin_index:end_index]
        max_value[:, :, i * w:(i + 1) * w] = sub_score.max(dim=dim+2)[0]
    return max_value

class QAConv(Module):
    def __init__(self, num_features, height, width):
        """
        Inputs:
            num_features: the number of feature channels in the final feature map.
            height: height of the final feature map
            width: width of the final feature map
        """
        super(QAConv, self).__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.height * self.width, 1)
        self.logit_bn = nn.BatchNorm1d(1)
        self.reset_parameters()
        self.sigmoid = nn.Sigmoid()
        self.cbam = CBAMBlock(channel=128, reduction=16)
        self.cpca = CPCA(channels=128)
    def reset_running_stats(self):
        self.bn.reset_running_stats()
        self.logit_bn.reset_running_stats()

    def reset_parameters(self):
        self.bn.reset_parameters()
        self.logit_bn.reset_parameters()
        with torch.no_grad():
            self.fc.weight.fill_(1. / (self.height * self.width))

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, prob_fea, gal_fea):
        hw = self.height * self.width
        prob_fea = self.cbam(prob_fea)
        gal_fea = self.cbam(gal_fea)
        #prob_fea = self.cpca(prob_fea)
        #gal_fea = self.cpca(gal_fea)

        prob_size = prob_fea.size(0)
        gal_size = gal_fea.size(0)

        # 如果你使用float32, 确保所有张量都是float32
        # 如果你使用float16, 确保所有张量都是float16
        prob_fea = prob_fea.half().view(prob_size, self.num_features, hw)
        gal_fea = gal_fea.half().view(gal_size, self.num_features, hw)

        # 双线性匹配
        score = torch.einsum('p c s, g c r -> p g r s', prob_fea, gal_fea)

        # 位置感知最大池化并移动到GPU
        score1 = position_aware_max(score, self.height, self.width, part=4, dim=0).half().to(torch.device('cuda'))
        score2 = position_aware_max(score, self.height, self.width, part=4, dim=1).half().to(torch.device('cuda'))

        # 拼接池化结果
        score = torch.cat((score1, score2), dim=-1)

        # 对结果进行批量归一化和全连接操作
        score = score.view(-1, 1, hw)
        score = self.bn(score).view(-1, hw)  # 注意这里的 bn 层已经是 float16 类型
        score = self.fc(score)
        score = score.view(-1, 2).sum(dim=1, keepdim=True)
        score = self.logit_bn(score)
        score = self.sigmoid(score)
        score = score.view(prob_size, gal_size)

        return score

