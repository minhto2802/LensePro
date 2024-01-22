import torch
from torch import nn
import torch.nn.functional as F

from networks.inception1d import InceptionModel


class InceptionModelNestedDropout(InceptionModel):
    """A PyTorch implementation of the InceptionTime model, adapted to nested dropout training"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.feat_extractor = nn.Sequential(self.blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        self.classifier = NetClassifier(self.feature_size, self.input_args['num_pred_classes'])

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out = self.feat_extractor(x)
        if self.self_train:
            return F.normalize(out, dim=1)
        return self.classifier(out)


# class NetClassifier(nn.Module):
#     def __init__(self, feat_dim, nb_cls):
#         super(NetClassifier, self).__init__()
#         self.weight = torch.nn.Parameter(nn.Linear(feat_dim, nb_cls, bias=False).weight.T,
#                                          requires_grad=True)  # dimension feat_dim * nb_cls
#
#     def get_weight(self, *args):
#         return self.weight, self.bias, self.scale_cls
#
#     def forward(self, feature, *args, **kwargs):
#         batch_size, n_feat = feature.size()
#         class_score = torch.mm(feature, self.weight)
#
#         return class_score


def main():
    from torchinfo import summary
    batch_size, input_dim, input_length, num_pos = 2, 1, 200, 8
    num_blocks, in_channels, pred_classes, out_channels = 3, 1, 2, 32
    bottleneck_channels, kernel_sizes, stride = 12, 15, 1

    net_feat = InceptionModel(num_blocks, in_channels, out_channels=out_channels, input_length=input_length,
                              bottleneck_channels=bottleneck_channels,
                              kernel_sizes=kernel_sizes,
                              use_residuals='default',  # 'default',
                              stride=stride,
                              num_pred_classes=pred_classes,
                              num_positions=num_pos)
    net_cls = NetClassifier(out_channels, pred_classes)

    summary(net_feat, input_size=[(batch_size, input_dim, input_length), (batch_size, num_pos)])
    summary(net_cls, input_size=[(batch_size, out_channels), (batch_size, num_pos)])


if __name__ == '__main__':
    main()
