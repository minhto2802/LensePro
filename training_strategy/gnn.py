import torch
from tqdm import tqdm
import grafog.transforms as T
from torch.nn import functional as F
from .vanilla import Model


class GNN(Model):
    def __init__(self, *args, **kwargs):
        super(GNN, self).__init__(*args, **kwargs)
        self.dataset_fields = ['x', 'y', 'edge_index', 'batch']

        self.node_aug = T.Compose([
            T.NodeDrop(p=0.45),
            # T.NodeMixUp(lamb=0.5, classes=kwargs['num_class']),
            T.NodeFeatureMasking(p=0.15),
        ])
        self.edge_aug = T.Compose([
            T.EdgeDrop(0.15),
        ])

    def augment(self, data, **kwargs):
        return self.edge_aug(self.node_aug(data, **kwargs), **kwargs)

    def to_device(self, batch):
        [setattr(batch, _, getattr(batch, _).to(self.device)) for _ in self.dataset_fields]

    def forward_backward_graph(self, x, y, edge_index, batch, *args, **kwargs):
        out = self.infer_graph(x, edge_index, batch)
        loss = self.loss_func(out, y.long())
        # loss = self.loss_func(out, y.float())
        self.optimize(loss)
        return out, loss

    def infer_graph(self, *args):
        return self.net(*args)

    def train(self, epoch, trn_dl, writer=None, *args, **kwargs):
        self.net.train()

        with tqdm(trn_dl, unit="batch") as t_epoch:
            global_step = self.scheduler.last_epoch
            t_epoch.set_description(f"Epoch {epoch}")
            for i, batch in enumerate(t_epoch):
                # Perform a single forward pass.
                # batch = self.augment(batch, batch=batch.batch)
                self.to_device(batch)
                out, loss = self.forward_backward_graph(batch.x, batch.y, batch.edge_index, batch.batch)
                # out, loss = self.forward_backward_graph(batch.x, batch.inv, batch.edge_index, batch.batch)

                t_epoch.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])

                if writer:
                    writer.add_scalar(f'LEARNING_RATE', self.optimizer.param_groups[0]['lr'], global_step)
        return loss

    def eval(self, tst_dl, device=None, **kwargs):
        self.net.eval()

        predictions = []
        for batch in tst_dl:  # Iterate in batches over the training/test dataset.
            self.to_device(batch)
            out = self.infer_graph(batch.x, batch.edge_index, batch.batch)
            predictions.append(self.to_prob(out))
            # predictions.append(out.max(dim=1)[1])

        return torch.concat(predictions)[:, 1].cpu().numpy()
        # return torch.concat(predictions).cpu().numpy()
