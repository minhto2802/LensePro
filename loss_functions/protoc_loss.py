import torch
import torch.nn.functional as F


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def protoc_dist(x, y):
    repr_loss = F.mse_loss(x, y)
    # repr_loss = torch.cdist(F.normalize(x), F.normalize(x),
    #                         p=2.0, compute_mode="donot_use_mm_for_euclid_dist")[0, 1]
    # std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    # std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    # std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    return repr_loss


def protoc_loss(model, model2, writer=None, current_epoch=0, n_class=2):
    x1 = model[1].prototypes
    x1 = x1.view(-1, x1.size(-1))
    x2 = model2[1].prototypes.detach()
    x2 = x2.view(-1, x2.size(-1))
    # x_cls = x.view(n_class, -1, x.size(-1)).mean(dim=1)
    # y_cls = y.view(n_class, -1, y.size(-1)).mean(dim=1)
    # net_dist = torch.cdist(F.normalize(x_cls), F.normalize(y_cls), p=2.0,
    #                        compute_mode="donot_use_mm_for_euclid_dist")
    # return net_dist
    # x = x.view(-1, x.size(-1))
    x = torch.concat((x1, x2), dim=0)
    x = x - x.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x))
    cov_x = (x.T @ x) / (x.size(0) - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.size(1))
    return 1 * (std_loss + cov_loss)  # + 1 * dist_loss


def protoc_loss_v0(model, model2=None, writer=None, current_epoch=0, n_class=2):
    # n_class = 2
    x = model[1].prototypes
    x = x.view(-1, x.size(-1))
    # #
    # intra_dist_lost_mat = -torch.cdist(F.normalize(x), F.normalize(x), p=2.0,
    #                                    compute_mode="donot_use_mm_for_euclid_dist")
    # intra_cls_mask = torch.zeros_like(intra_dist_lost_mat)
    # n_protoc = int(intra_cls_mask.size(0) / n_class)
    # for i in range(n_class):
    #     intra_cls_mask[i * n_protoc: (i + 1) * n_protoc, i * n_protoc: (i + 1) * n_protoc] = 1
    # intra_cls_mask *= torch.abs(1 - torch.eye(intra_cls_mask.size(0)).cuda())
    # intra_dist_lost = torch.sum(intra_cls_mask * intra_dist_lost_mat) / intra_cls_mask.sum()
    #
    # if model2 is not None:
    #     y = model2[1].prototypes.detach()
    #     y = y.view(-1, y.size(-1))
    #     # dist_loss = (0.04-F.mse_loss(x, y)) * 10
    #     # dist_loss = torch.abs(2 - torch.cosine_similarity(x, y).sum())
    #     # x = torch.concat([x, y])
    #     inter_dist_loss_mat = torch.cdist(F.normalize(x), F.normalize(y.detach()), p=2.0,
    #                                       compute_mode="donot_use_mm_for_euclid_dist")
    #
    #     inter_cls_mask = torch.abs(1 - intra_cls_mask) - torch.eye(intra_cls_mask.size(0)).cuda()
    #     inter_dist_loss = torch.sum(inter_dist_loss_mat * inter_cls_mask) / inter_cls_mask.sum()
    #
    #     x_cls = x.view(n_class, -1, x.size(-1)).mean(dim=1)
    #     y_cls = y.view(n_class, -1, y.size(-1)).mean(dim=1)
    #     net_dist = torch.cdist(F.normalize(x_cls), F.normalize(y_cls), p=2.0,
    #                            compute_mode="donot_use_mm_for_euclid_dist")
    #     net_dist_loss = torch.sum(torch.abs(net_dist) * torch.eye(n_class).cuda()) / n_class
    #
    #     # dist_loss = torch.abs(intra_dist_lost) / (torch.abs(inter_dist_loss) + torch.abs(net_dist_loss))
    #     # dist_loss = 1 - (torch.abs(intra_dist_lost) / (torch.abs(inter_dist_loss)-torch.abs(intra_dist_lost)))
    #     d1, d2 = torch.abs(intra_dist_lost), torch.abs(inter_dist_loss)
    #     dist_loss = -((d2 + d1) + 1 * (d2 - d1) + 0.1 * net_dist_loss)
    #     # dist_loss = -torch.log((d2 + d1) * torch.square(d2 - d1))
    #
    #     # dist_loss = -(torch.log10(1/(d1 + 1)) + torch.log10(d2 + 1))
    #     # dist_loss = -(torch.log10(1/(d1 + 1)) + torch.log10(d2/(d2+d1) + 1))
    #     # dist_loss = -(torch.log10(d2/(d2+d1) + 1))
    #     # dist_loss = -d2/(d2+d1 + 1)
    #
    #     # dist_loss = torch.abs(intra_dist_lost) / torch.abs(inter_dist_loss)
    #
    #     dist_loss_mat = torch.abs(inter_dist_loss_mat) * inter_cls_mask + \
    #                     torch.abs(intra_dist_lost_mat) * intra_cls_mask
    #
    # else:
    #     inter_cls_mask = torch.abs(1 - intra_cls_mask) - torch.eye(intra_cls_mask.size(0)).cuda()
    #     inter_dist_loss = torch.sum(intra_dist_lost_mat * inter_cls_mask) / inter_cls_mask.sum()
    #     d1, d2 = torch.abs(intra_dist_lost), torch.abs(inter_dist_loss)
    #
    #     # dist_loss = -(d2 - d1)
    #     # dist_loss = 1-(torch.log10(d2+1) - torch.log10(d1+1))
    #     # dist_loss = torch.exp(-1/d1)
    #     # dist_loss = 1 - (d2 + d2/d1)
    #     dist_loss = (torch.square(1 + d1) - d2 + 1e-7) / (d2 + 1)
    #     # dist_loss = (5*d1 - d2 + 1e-4)/(d2+1)
    #
    #     dist_loss_mat = torch.abs(intra_dist_lost_mat) * inter_cls_mask + \
    #                     torch.abs(intra_dist_lost_mat) * intra_cls_mask
    #
    #     # dist_loss = intra_dist_lost
    # if writer:
    #     writer.add_figure(f'train/dist_loss', sns.heatmap(dist_loss_mat.detach().cpu().numpy()).get_figure(),
    #                       global_step=current_epoch)

    # return dist_loss

    # # dist_loss = 0
    # # x = x.view(-1, x.size(-1))
    # x = x - x.mean(dim=0)
    # std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    # std_loss = torch.mean(F.relu(1 - std_x))
    # cov_x = (x.T @ x) / (x.size(0) - 1)
    # cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.size(1))
    # return 1 * (std_loss + cov_loss) # + 1 * dist_loss

    # x = x.view(2, -1, x.size(-1))
    #
    # # single class
    # x0 = x - x.mean(dim=1, keepdims=True)
    # std_x = torch.sqrt(x0.var(dim=1) + 0.0001)
    # std_loss = torch.mean(F.relu(1 - std_x), dim=1)
    #
    # # all classes
    # x_avg = x.mean(dim=1)
    # x0_avg = x_avg - x_avg.mean(dim=0)
    # std_x_avg = torch.sqrt(x0_avg.var(dim=0) + 0.0001)
    # std_loss_avg = torch.mean(F.relu(1 - std_x_avg))
    #
    # return std_loss.mean() / 2 + std_loss_avg / 2
    # return 0