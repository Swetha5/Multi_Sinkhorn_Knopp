import torch
import torch.nn.functional as F


def normalize_embeddings(a, eps=1e-8):
    a_n = a.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    return a_norm


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a = normalize_embeddings(a, eps)
    b = normalize_embeddings(b, eps)

    sim_mt = torch.mm(a, b.transpose(0, 1))
    return sim_mt


class WeightEMA(object):
    def __init__(self, model, ema_model, ema_decay=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = ema_decay
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        # self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                # param.mul_(1 - self.wd)


def cosine_pdist(x, y):
    return F.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
