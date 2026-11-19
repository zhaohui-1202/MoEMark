import torch

from torch import nn

from geometry_encoder import FiLMHostResidual as OriginalFiLM

class SpatialFiLM(OriginalFiLM):
    def __init__(self,dim=256,rho=.5):
        super().__init__(dim,rho)

        with torch.random.fork_rng(devices=[]):
            self.image_projection=nn.Linear(5*dim,dim)
        with torch.no_grad():
            self.image_projection.weight.zero_()
            self.image_projection.weight[:,:dim].copy_(torch.eye(dim))
            self.image_projection.bias.zero_()
    def forward(self,v,c):
        return super().forward(v,self.image_projection(c))

def paired(model,v,c):
    n,m=len(v),len(c)
    r=model(v[:,None,:].expand(n,m,-1).reshape(-1,v.shape[-1]),
            c[None,:,:].expand(n,m,-1).reshape(-1,c.shape[-1]))
    return {k:t.reshape(n,m,-1) for k,t in r.items()}
