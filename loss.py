from functools import reduce
import torch
import torch.nn.functional as F
import torch.nn as nn

def linear_combination(x, y, epsilon):
    return epsilon * x + (1-epsilon) * y

def reduce_loss(loss, reduction='mean'):
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss

class LabelSmoothingCE(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingCE, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1] # 类别个数
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction) # batch 维度的 reduce
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

def log_t(u, t):
    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0-t) - 1.0) / (1.0 - t)

def exp_t(u, t):
    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

def compute_normalization_fixed_point(activations, t, num_iters=5):
    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu

def tempered_softmax(activations, t, num_iters=5):
    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization_fixed_point(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)

class BiTemperedLoss(nn.Module):
    def __init__(self, t1=1.0, t2=1.8, label_smoothing=0.01, num_iters=5, reduction='mean'):
        super(BiTemperedLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters
        self.reduction = reduction

    def forward(self, preds, labels):
        if labels.shape != preds.shape:
            labels_onehot = torch.zeros_like(preds)
            labels_onehot.scatter_(1, labels[..., None], 1)
        else:
            labels_onehot = labels

        if self.label_smoothing > 0.0:
            num_classes = labels_onehot.shape[-1]
            labels_onehot = (1 - num_classes / (num_classes - 1) * self.label_smoothing) * labels_onehot + self.label_smoothing / (num_classes - 1)

        probabilities = tempered_softmax(preds, self.t2, self.num_iters)

        temp1 = (log_t(labels_onehot + 1e-10, self.t1) - log_t(probabilities, self.t1)) * labels_onehot
        temp2 = (1 / (2 - self.t1)) * (torch.pow(labels_onehot, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss_values = temp1 - temp2

        return reduce_loss(torch.sum(loss_values, dim=-1), reduction=self.reduction)

class TaylorSoftmax(nn.Module):
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n
    
    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingCE(epsilon=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        loss = self.lab_smooth(log_probs, labels)
        return loss