
import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class WeightedMseLoss(_WeightedLoss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        weight: Tensor = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.sum(self.weight * (input - target) ** 2)

class DeepKoopmanExplicitLoss(nn.Module):
    # this loss function is explained in Bethany Lusch's Natura Communication paper
    # Lusch, B., Kutz, J.N. and Brunton, S.L., 2018. 
    # Deep learning for universal linear embeddings of nonlinear dynamics. Nature communications, 9(1), p.4950.
    def __init__(self, alpha_1, alpha_2, alpha_reg, weights_x):
        super(DeepKoopmanExplicitLoss, self).__init__()
        self._alpha_1 = alpha_1
        self._alpha_2 = alpha_2
        self._alpha_reg = alpha_reg  # Regularization strength
        self._weights_x = weights_x

    def recon(self, x0, x0_pred):
        loss_recon = F.mse_loss(x0, x0_pred)
        return loss_recon
    
    def pred(self, x_plus, x_plus_pred):
        loss = 0
        for i in range(len(x_plus)):
            # loss = loss + F.mse_loss(x_plus[i], x_plus_pred[i])
            loss = loss + torch.mean(torch.mean(torch.square((x_plus[i] - x_plus_pred[i])*self._weights_x), dim=1))
        loss_pred = loss / len(x_plus)
        return loss_pred
        
    def lin(self, y_plus, y_plus_pred):
        loss = 0
        for i in range(len(y_plus)-1):
            # loss = loss + F.mse_loss(y_plus[i], y_plus_pred[i])
            loss = loss + torch.mean(torch.mean(torch.square(y_plus[i] - y_plus_pred[i]), dim=1))
        loss_lin = loss / len(y_plus)
        return loss_lin

    def inf_norm(self, x0, x0_pred, x_plus, x_plus_pred):
        lin_f1_penalty = torch.norm(torch.norm(x0 - x0_pred, dim=1, p=float('inf')), p=float('inf'))
        lin_f2_penalty = torch.norm(torch.norm(x_plus - x_plus_pred, dim=1, p=float('inf')), p=float('inf'))
        return lin_f1_penalty + lin_f2_penalty
    
    def forward_rec(self, loss_recon, model_parameters):
        # Regularization term (L2 norm of the parameters)
        l2_reg = torch.tensor(0.).to(loss_recon.device)
        for param in model_parameters:
            l2_reg += torch.norm(param)

        total_loss = self._alpha_1 * loss_recon + self._alpha_reg * l2_reg
        return total_loss

    def forward(self, loss_recon, loss_pred, loss_lin, loss_inf, model_parameters):
        # Regularization term (L2 norm of the parameters)
        l2_reg = torch.tensor(0.).to(loss_recon.device)
        for param in model_parameters:
            l2_reg += torch.norm(param)

        total_loss = self._alpha_1 * (loss_recon + loss_pred) + loss_lin + self._alpha_2 * loss_inf + self._alpha_reg * l2_reg
        return total_loss

class DeepKoopmanExplicitMetric(nn.Module):
    # this loss function is explained in Bethany Lusch's Natura Communication paper
    def __init__(self):
        super(DeepKoopmanExplicitMetric, self).__init__()

    def recon(self, x0, x0_pred):
        metrics_recon = torch.mean(torch.abs(x0 - x0_pred))
        return metrics_recon
    
    def pred(self, x_plus, x_plus_pred):
        metrics = 0
        for i in range(len(x_plus)):
            # loss = loss + F.mse_loss(x_plus[i], x_plus_pred[i])
            metrics = metrics + torch.mean(torch.mean(torch.abs(x_plus[i] - x_plus_pred[i]), dim=1))
        metrics_pred = metrics / len(x_plus)
        return metrics_pred
        
    def lin(self, y_plus, y_plus_pred):
        metrics = 0
        for i in range(len(y_plus)-1):
            # loss = loss + F.mse_loss(y_plus[i], y_plus_pred[i])
            metrics = metrics + torch.mean(torch.mean(torch.abs(y_plus[i] - y_plus_pred[i]), dim=1))
        metrics_lin = metrics / len(y_plus)
        return metrics_lin  

    def forward(self, metric_recon, metric_pred, metric_lin):
        return torch.tensor([metric_recon, metric_pred, metric_lin]).to(metric_recon.device)

class MaeMetric(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "none") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.mean(torch.abs(input - target), dim=0)

