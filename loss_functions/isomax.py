import torch.nn as nn
import torch.nn.functional as F
import torch


class IsoMaxLossFirstPart(nn.Module):
    """Replaces classifier layer"""

    def __init__(self, in_features, out_features, const_init=0):
        super(IsoMaxLossFirstPart, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prototypes = nn.Parameter(torch.Tensor(out_features, in_features), )
        nn.init.constant_(self.prototypes, const_init)
        # nn.init.normal_(self.prototypes, 0.5)

    def forward(self, features):
        # distances = F.pairwise_distance(features.unsqueeze(2), self.weights.t().unsqueeze(0), p=2)
        # distances_protoc = F.pairwise_distance(self.prototypes.unsqueeze(2),
        #                                        self.prototypes.t().unsqueeze(0)).sum()
        distances = torch.cdist(features, self.prototypes, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances  # + distances_protoc
        # distances = torch.cdist(features, self.prototypes, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        # logits = -distances
        return logits

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class IsoMaxLossSecondPart(nn.Module):
    def __init__(self, entropic_scale=200, reduction='none'):
        self.entropic_scale = entropic_scale
        self.reduction = reduction
        super(IsoMaxLossSecondPart, self).__init__()

    def forward(self, outputs, targets, reduction=None, *args, **kwargs):
        self.reduction = reduction if reduction is not None else self.reduction

        probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * outputs)
        targets = targets.argmax(1) if targets.ndim == 2 else targets  # added
        probabilities_at_targets = probabilities_for_training[range(outputs.size(0)), targets]
        loss = -torch.log(probabilities_at_targets + 1e-8)
        if reduction == 'none':
            return loss
        return loss.mean()


class IsoMaxLossFirstPartV1(nn.Module):
    """Replaces classifier layer"""

    def __init__(self, in_features, out_features):
        super(IsoMaxLossFirstPartV1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.distance_scale = nn.Parameter(torch.FloatTensor(1))
        self.p_dist = torch.nn.PairwiseDistance(p=2)
        nn.init.normal_(self.weights, 0.0, 1.0)
        # nn.init.constant_(self.distance_scale, 1.0)
        # nn.init.constant_(self.weights, 0.0)

    def forward(self, features, distance_scale=False):
        """

        :param features:
        :param distance_scale: True for training and False for OOD detection
        :return:
        """
        distances = self.p_dist(
            F.normalize(features, p=2).unsqueeze(2),
            F.normalize(self.weights, p=2).t().unsqueeze(0),
        )
        distances_protoc = self.p_dist(F.normalize(self.weights, p=2).unsqueeze(2),
                                       F.normalize(self.weights, p=2).t().unsqueeze(0))
        logits = distances - distances_protoc.sum() * 1e-1
        if distance_scale:
            return -logits * torch.abs(self.distance_scale)
        return -logits

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)


class IsoMaxLossSecondPartV1(nn.Module):
    def __init__(self, entropic_scale=1):
        self.entropic_scale = entropic_scale
        super(IsoMaxLossSecondPartV1, self).__init__()

    def forward(self, outputs, targets, reduction='none', *args, **kwargs):
        probabilities_for_training = nn.Softmax(dim=1)(self.entropic_scale * outputs)
        probabilities_at_targets = probabilities_for_training[range(outputs.size(0)), targets]
        loss = -torch.log(probabilities_at_targets)
        if reduction == 'none':
            return loss
        return loss.mean()
