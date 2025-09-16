import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F


class TripletLoss(Module):
    def __init__(self, matcher, margin=16, clothing_classifier=None, id_classifier=None, num_classes=256):
        """
        Inputs:
            matcher: a class for matching pairs of images
            margin: margin parameter for the triplet loss
        """
        super(TripletLoss, self).__init__()
        self.matcher = matcher
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.clothing_classifier = clothing_classifier
        self.id_classifier = id_classifier
        self.num_classes = num_classes
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def reset_running_stats(self):
        self.matcher.reset_running_stats()

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, feature, target):
        self._check_input_dim(feature)
        # self.matcher.make_kernel(feature)

        score = self.matcher(feature, feature)  # [b, b]

        target1 = target.unsqueeze(1)
        mask = (target1 == target1.t())
        pair_labels = mask.float()

        min_pos = torch.min(score * pair_labels + 
                (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[0]
        max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]

        # Compute ranking hinge loss
        loss = self.ranking_loss(min_pos, max_neg, torch.ones_like(target))

        onehot_target = F.one_hot(target, num_classes=self.num_classes).to(torch.float32)
        clothing_loss = 0
        id_loss = 0
        if (self.clothing_classifier):
            clothing_pred = self.clothing_classifier(feature)
            clothing_loss = self.cross_entropy_loss(clothing_pred, onehot_target)
            # print(clothing_loss)
            # print(feature.shape, clothing_pred.shape, target.shape, score.shape, target, clothing_loss.shape); exit(0)
        if (self.id_classifier):
            id_pred = self.id_classifier(feature)
            id_loss = self.cross_entropy_loss(id_pred, onehot_target)


        loss = loss + 0.005 * (clothing_loss + id_loss)
            

        with torch.no_grad():
            acc = (min_pos >= max_neg).float()

        return loss, acc
