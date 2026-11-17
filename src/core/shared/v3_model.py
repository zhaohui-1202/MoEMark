import numpy as np

import torch

from torch import nn

import torch.nn.functional as F

def transformer(width, layers=1):
    layer=nn.TransformerEncoderLayer(width,4,width*3,dropout=0.,batch_first=True,norm_first=True)
    return nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)

def group_mean(x, owner, count):

    x=x.float()
    out=x.new_zeros((count,x.shape[-1])).index_add(0,owner,x)
    n=torch.bincount(owner,minlength=count).clamp_min(1).to(x.dtype)
    return out/n[:,None]

class SetExpert(nn.Module):

    def __init__(self,width,dim=256,latents=8):
        super().__init__()
        self.queries=nn.Parameter(torch.randn(1,latents,width)*.02)
        self.attention=nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.transformer=transformer(width)
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,dim))

    def forward(self,x):
        x=x[None]
        q=self.queries
        h=q+self.attention(q,x,x,need_weights=False)[0]
        return self.head(self.transformer(h).mean(1))[0].float()

class StructuralMoE(nn.Module):
    def __init__(self,width=128,dim=256,routing='dense',checkpoint_geometry=False):
        super().__init__()
        self.width=width; self.dim=dim; self.routing=routing
        self.checkpoint_geometry=checkpoint_geometry
        self.vertex=nn.Linear(4,width)
        self.role=nn.Embedding(5,width)
        self.geometry=transformer(width,2)
        self.part=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.LayerNorm(width))
        self.feature=nn.Sequential(nn.Linear(width+7,width),nn.GELU(),nn.LayerNorm(width))
        self.geometry_expert=SetExpert(width,dim)
        self.relationship=nn.Linear(8,width)
        self.local_transformer=transformer(width,2)
        self.local_expert=SetExpert(width,dim)
        self.region_projection=nn.Linear(10,width)
        self.region_transformer=transformer(width,2)
        self.region_expert=SetExpert(width,dim)
        self.router=nn.Sequential(nn.Linear(18,64),nn.GELU(),nn.Linear(64,3))

    def forward_one(self,s):
        device=next(self.parameters()).device
        t={k:torch.as_tensor(v,device=device) for k,v in s.items()}
        h=self.vertex(t['rings'])
        sequence=t['roles']!=0
        base=h.mean(1)
        if sequence.any():

            ids=torch.where(sequence)[0]
            result=[]
            from torch.utils.checkpoint import checkpoint
            for ix in ids.split(256):
                if self.training and self.checkpoint_geometry:
                    y=checkpoint(self.geometry,h[ix],use_reentrant=False)
                else: y=self.geometry(h[ix])
                result.append(y.mean(1))
            base=base.index_copy(0,ids,torch.cat(result))
        ring=group_mean(base,t['segment_ring'],len(t['ring_part']))
        ring=ring+self.role(t['ring_role'])
        part=self.part(group_mean(ring,t['ring_part'],len(t['part_feature'])))
        feature=group_mean(part,t['part_feature'],len(t['meta']))

        feature=self.feature(torch.cat([feature,t['meta'][:,2:]],-1))
        anchors=t['anchors']
        e1=self.geometry_expert(feature[anchors])
        local=feature[t['neighbors']]+self.relationship(t['relations'])
        local=self.local_transformer(local)[:,0]
        e2=self.local_expert(local)
        global_tokens=self.region_transformer(self.region_projection(t['regions'])[None])[0]
        e3=self.region_expert(global_tokens)
        experts=F.normalize(torch.stack([e1,e2,e3]),dim=-1)
        p=F.softmax(self.router(t['summary']).float(),dim=-1)
        if self.routing=='uniform': weights=p*0+1/3
        elif self.routing=='top2':
            mask=torch.zeros_like(p).scatter(0,p.topk(2).indices,1.)
            weights=p*mask; weights=weights/weights.sum()
        elif self.routing=='dense': weights=p
        else: raise ValueError(self.routing)
        vector=F.normalize((weights[:,None]*experts).sum(0),dim=-1)
        return vector,weights,experts

    def forward(self,samples):
        results=[self.forward_one(s) for s in samples]
        return tuple(torch.stack([r[i] for r in results]) for i in range(3))

class FiLMHostResidual(nn.Module):
    def __init__(self,dim=256,rho=.5):
        super().__init__()
        if not 0<rho<1: raise ValueError('rho must be in (0,1)')
        self.rho=rho
        self.host=nn.Linear(dim,dim)
        self.condition=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,dim*2))
        nn.init.normal_(self.condition[-1].weight,std=.001)
        nn.init.zeros_(self.condition[-1].bias)
        self.residual=nn.Linear(dim,dim,bias=False)
        self.hash=nn.Linear(dim,256)

    def forward(self,v,c):
        h=F.normalize(self.host(v),dim=-1)
        gamma,beta=self.condition(F.normalize(c,dim=-1)).chunk(2,-1)
        r=self.residual(gamma*h+beta)
        norm=r.float().norm(dim=-1,keepdim=True).clamp_min(1e-8)
        delta=r*torch.clamp(self.rho/norm,max=1.)
        z=F.normalize(h+delta,dim=-1)
        return {'z':z,'logits':self.hash(z),'delta':delta,'host':h}

def subset_anchors(s,indices):

    ix=np.asarray(indices,dtype=np.int64)
    neighbors=s['neighbors'][ix]
    keep=np.unique(neighbors)
    remap=np.full(len(s['meta']),-1,np.int64); remap[keep]=np.arange(len(keep))
    parts=np.flatnonzero(np.isin(s['part_feature'],keep))
    part_map=np.full(len(s['part_feature']),-1,np.int64); part_map[parts]=np.arange(len(parts))
    rings=np.flatnonzero(np.isin(s['ring_part'],parts))
    ring_map=np.full(len(s['ring_part']),-1,np.int64); ring_map[rings]=np.arange(len(rings))
    segments=np.flatnonzero(np.isin(s['segment_ring'],rings))
    return dict(rings=s['rings'][segments],roles=s['roles'][segments],
                segment_ring=ring_map[s['segment_ring'][segments]],ring_part=part_map[s['ring_part'][rings]],
                ring_role=s['ring_role'][rings],part_feature=remap[s['part_feature'][parts]],
                meta=s['meta'][keep],anchors=remap[s['anchors'][ix]],neighbors=remap[neighbors],
                relations=s['relations'][ix],regions=s['regions'],summary=s['summary'])
