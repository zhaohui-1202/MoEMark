import numpy as np

import shapely

from scipy.spatial import cKDTree

from vector_data import build_cache,attack_frame,write_and_reload

def distance_unit(frame,metric,scale):
    xy=shapely.get_coordinates(shapely.centroid(frame.geometry.array))*metric
    dist,_=cKDTree(xy).query(xy,k=min(4,len(xy)))
    d=np.asarray(dist)[:,1:]; d=np.where(d>1e-10,d,np.inf).min(1)
    good=d[np.isfinite(d)]
    return (float(np.median(good)),False) if len(good) else (float(scale*.001),True)

def execute(frame,ops,seed,metric,scale):
    rng=np.random.default_rng(seed); out=frame.copy()
    ell,fallback=distance_unit(frame,metric,scale)
    origin=shapely.get_coordinates(shapely.centroid(frame.geometry.array)).mean(0)
    records=[]
    for i,op in enumerate(ops):
        name=op['name']
        if name in ('delete','rotate'):
            out,_=attack_frame(out,[op],seed+i)
        elif name=='feature_jitter':
            delta=rng.normal(0,op['sigma_ratio']*ell,(len(out),2))
            norms=np.linalg.norm(delta,axis=1,keepdims=True)
            delta*=np.minimum(1,3*op['sigma_ratio']*ell/np.maximum(norms,1e-20))

            geom=np.asarray(out.geometry.array,dtype=object).copy()
            coords,owners=shapely.get_coordinates(geom,return_index=True)
            out.geometry=shapely.set_coordinates(geom,coords+delta[owners]/metric)
        elif name=='simplify':
            transformed=shapely.transform(out.geometry.array,lambda xy:(xy-origin)*metric)
            transformed=shapely.simplify(transformed,op['tolerance_ratio']*ell,preserve_topology=True)
            out.geometry=shapely.transform(transformed,lambda xy:xy/metric+origin)
        elif name=='quantize':
            grid=op['step_ratio']*ell
            out.geometry=shapely.transform(out.geometry.array,lambda xy:np.round((xy-origin)*metric/grid)*grid/metric+origin)
        else: raise ValueError('Unknown attack '+name)
        bad=out.geometry.is_empty|~out.geometry.is_valid
        if bad.any(): raise ValueError(f'{name} produced {int(bad.sum())} invalid/empty geometries; candidate rejected without repair')
        if len(out)<2: raise ValueError('Too few surviving features')
        records.append(dict(**op,remaining_features=len(out)))
    return out,dict(seed=seed,operations=records,ell=ell,ell_fallback=fallback,
                    original_features=len(frame),remaining_features=len(out),
                    actual_deleted_fraction=1-len(out)/len(frame),
                    input_vertices=int(shapely.get_num_coordinates(frame.geometry.array).sum()),
                    output_vertices=int(shapely.get_num_coordinates(out.geometry.array).sum()),
                    jitter_definition='rigid per-feature shift, NOT independent vertex noise',
                    valid=True)
