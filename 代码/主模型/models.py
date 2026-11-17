import torch
from torch import nn
import torch.nn.functional as F

def unit(x):
    return F.normalize(x.float(), dim=-1)

class Adapter(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, width), nn.GELU(), nn.Linear(width, 256))
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return self.net(h)

class Detection(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([Adapter() for _ in range(4)])

        self.gate = nn.Sequential(nn.Linear(274, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, h, summary, mode='sparse', expert=None):
        r = torch.stack([e(h) for e in self.experts], 1)
        w = F.softmax(self.gate(torch.cat([h, summary], -1)), -1)
        if mode == 'uniform':
            w = torch.ones_like(w) / 3
        elif mode == 'sparse':
            mask = torch.zeros_like(w).scatter(1, w.topk(2, dim=1).indices, 1.)
            w = w * mask
            w = w / w.sum(1, keepdim=True)
        if expert == 0:
            correction = r[:, 0]
        elif expert is not None:
            correction = .5 * r[:, 0] + .5 * r[:, expert]
        else:
            correction = .5 * r[:, 0] + .5 * (w[..., None] * r[:, 1:]).sum(1)
        return unit(h + .3 * correction), w

def register_bits(logits, payload):
    return torch.logical_xor(logits >= 0, payload.bool())

def recover_bits(record, logits):
    return torch.logical_xor(record.bool(), logits >= 0)

def accepts(recovered, payload, threshold):
    if not isinstance(threshold, int) or not 0 <= threshold <= 256:
        raise ValueError('Threshold must be an integer in [0,256].')
    return (recovered == payload).sum(-1) >= threshold

class PlainRegistration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=torch.nn.Sequential(torch.nn.Linear(402,256),torch.nn.LayerNorm(256))
        self.net=torch.nn.Sequential(torch.nn.Linear(274,254),torch.nn.GELU(),torch.nn.Linear(254,256))
        torch.nn.init.normal_(self.net[-1].weight,std=.001)
        torch.nn.init.zeros_(self.net[-1].bias)

    def encode(self,x):
        return self.projection(x)

    def forward(self,x,expert=None):
        h=self.encode(x)
        return unit(h+.3*self.net(torch.cat([h,x[:,-18:]],dim=-1))),None
