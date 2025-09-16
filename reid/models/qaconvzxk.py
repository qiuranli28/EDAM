import torch
from torch import nn
from torch.nn import Module


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



class FaceQAConv(Module):
    def __init__(self, num_features, height, width):
        """
        Inputs:
            num_features: the number of feature channels in the final feature map.
            height: height of the final feature map
            width: width of the final feature map
        """
        super(FaceQAConv, self).__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.height * self.width, 1)
        self.logit_bn = nn.BatchNorm1d(1)
        self.reset_parameters()
        self.sigmoid = nn.Sigmoid()

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
        prob_size = prob_fea.size(0)
        gal_size = gal_fea.size(0)
        prob_fea = prob_fea.view(prob_size, self.num_features, hw)
        gal_fea = gal_fea.view(gal_size, self.num_features, hw)
        # score=torch.cosine_similarity(prob_fea, gal_fea, dim=1)
        score = torch.einsum('p c s, g c r -> p g r s', prob_fea, gal_fea)




        score1=position_aware_max(score, self.height, self.width, part=4, dim=0).to(torch.device('cuda'))
        score2 = position_aware_max(score, self.height, self.width, part=4, dim=1).to(torch.device('cuda'))
        score = torch.cat((score1, score2), dim=-1)




        score = score.view(-1, 1, hw)
        score = self.bn(score).view(-1, hw)
        score = self.fc(score)

        score = score.view(-1, 2).sum(dim=1, keepdim=True)
        score = self.logit_bn(score)
        score = self.sigmoid(score)
        score = score.view(prob_size, gal_size)

        return score


if __name__ == '__main__':
    score = torch.rand((16, 16, 192, 192), dtype=torch.float16)
    h = 24
    w = 8
    max_value = position_aware_max(score, h, w, 3)
    print(max_value.shape)
