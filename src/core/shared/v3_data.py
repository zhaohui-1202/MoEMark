from pathlib import Path

import numpy as np

import shapely

import geopandas as gpd

from scipy.spatial import cKDTree

from geometry_v1 import components,resample

def regional_tokens(meta):
    xy=meta[:,:2]
    lo=xy.min(0); span=np.maximum(xy.max(0)-lo,1e-8)
    out=[]

    for resolution in (2,4,8):
        ij=np.minimum(((xy-lo)/span*resolution).astype(int),resolution-1)
        for a in range(resolution):
            for b in range(resolution):
                selected=(ij[:,0]==a)&(ij[:,1]==b)
                n=int(selected.sum())
                values=meta[selected,2:6].mean(0) if n else np.zeros(4)
                out.append([(a+.5)/resolution,(b+.5)/resolution,1/resolution,
                            n/len(meta),np.log1p(n)/np.log1p(len(meta)),*values,float(n>0)])
    return np.asarray(out,np.float32)

def all_metadata(frame,center,scale,metric):
    geoms=frame.geometry.array
    centroids=shapely.get_coordinates(shapely.centroid(geoms))
    bounds=shapely.bounds(geoms)
    radius=np.linalg.norm((bounds[:,2:]-bounds[:,:2])*metric,axis=1)/2

    lengths=shapely.length(shapely.transform(geoms,lambda x:x*metric))
    area=shapely.area(geoms)*np.prod(metric)
    n_parts=shapely.get_num_geometries(geoms)
    holes=np.zeros(len(frame),np.float32)
    parts,owners=shapely.get_parts(geoms,return_index=True)
    np.add.at(holes,owners,shapely.get_num_interior_rings(parts))
    ispoint=np.isin(shapely.get_type_id(geoms),[0,4]).astype(float)
    meta=np.column_stack([(centroids-center)*metric/scale,radius/scale,lengths/scale,
                          np.sqrt(area)/scale,np.log1p(n_parts),np.log1p(holes),np.log1p(n_parts+holes),ispoint]).astype(np.float32)
    return meta

def encode_hierarchy(geoms,metric):
    chunks=[]; roles=[]; segment_ring=[]; ring_part=[]; ring_role=[]; part_feature=[]
    for owner,g in enumerate(geoms):
        g=shapely.normalize(g)
        center=np.asarray(g.centroid.coords[0])[:2]*metric
        coords=shapely.get_coordinates(g)*metric
        radius=max(float(np.linalg.norm(coords-center,axis=1).max()),1e-8)
        for part in components(g):
            pi=len(part_feature); part_feature.append(owner)
            if part.geom_type=='Point': specs=[(part.coords,0)]
            elif part.geom_type=='LineString': specs=[(part.coords,2 if part.is_ring else 1)]
            elif part.geom_type=='Polygon': specs=[(part.exterior.coords,3)]+[(r.coords,4) for r in part.interiors]
            else: raise ValueError('Unsupported geometry '+part.geom_type)
            for coordinates,role in specs:
                ri=len(ring_part); ring_part.append(pi); ring_role.append(role)
                a=np.asarray(coordinates)[:,:2]*metric


                starts=range(0,max(1,len(a)-1),127)
                for start in starts:
                    segment=a[start:min(start+128,len(a))]
                    sampled=(resample(segment,32)-center)/radius
                    delta=np.roll(sampled,-1,axis=0)-sampled
                    if len(a)>128 or role in (0,1): delta[-1]=0
                    chunks.append(np.concatenate([sampled,delta],-1)); roles.append(role); segment_ring.append(ri)
    return dict(rings=np.asarray(chunks,np.float32),roles=np.asarray(roles,np.int64),
                segment_ring=np.asarray(segment_ring,np.int64),ring_part=np.asarray(ring_part,np.int64),
                ring_role=np.asarray(ring_role,np.int64),part_feature=np.asarray(part_feature,np.int64))

def build_cache(frame,center,scale,metric,seed,anchors=1024,k=8):
    if len(frame)<2: raise ValueError('Need at least two file features')
    meta=all_metadata(frame,center,scale,metric)
    rng=np.random.default_rng(seed)
    anchor_source=np.sort(rng.choice(len(frame),min(anchors,len(frame)),replace=False))
    tree=cKDTree(meta[:,:2].astype(np.float64))
    _,candidate=tree.query(meta[anchor_source,:2],k=min(k+1,len(frame)))

    nbors=[]
    for source,near in zip(anchor_source,np.atleast_2d(candidate)):
        others=[int(x) for x in near if int(x)!=source][:k]
        while len(others)<k: others.append(int(source))
        nbors.append([int(source)]+others)
    nbors=np.asarray(nbors,np.int64)
    keep=np.unique(nbors)
    mapping=np.full(len(frame),-1,np.int64); mapping[keep]=np.arange(len(keep))
    target=meta[anchor_source]
    relxy=meta[nbors,:2]-target[:,None,:2]
    distance=np.linalg.norm(relxy,axis=-1,keepdims=True)
    size=np.log((meta[nbors,2:3]+1e-8)/(target[:,None,2:3]+1e-8)).clip(-10,10)
    g=np.asarray(frame.geometry.array,dtype=object)
    intersects=shapely.intersects(g[anchor_source,None],g[nbors])[...,None]
    contains=shapely.contains(g[anchor_source,None],g[nbors])[...,None]
    self_flag=(nbors==anchor_source[:,None])[...,None]
    same_dim=(meta[nbors,8:9]==target[:,None,8:9]).astype(float)
    relations=np.concatenate([relxy,distance,size,intersects,contains,self_flag,same_dim],axis=-1).astype(np.float32)
    encoded=encode_hierarchy(frame.geometry.iloc[keep],metric)
    encoded.update(meta=meta[keep],anchors=mapping[anchor_source],neighbors=mapping[nbors],
                   relations=relations,regions=regional_tokens(meta),
                   summary=np.r_[meta.mean(0),meta.std(0)].astype(np.float32))
    audit=dict(source_features=len(frame),anchor_features=len(anchor_source),encoded_features=len(keep),
               segments=len(encoded['rings']),rings=len(encoded['ring_part']),parts=len(encoded['part_feature']),
               full_file_regional_features=len(meta),anchor_source_indices=anchor_source.tolist())
    return encoded,audit

def attack_frame(frame,operations,seed):


    rng=np.random.default_rng(seed); out=frame.copy(); records=[]
    for op in operations:
        name=op['name']
        if name=='delete':
            count=max(2,int(len(out)*(1-op['fraction'])))
            chosen=np.sort(rng.choice(len(out),count,replace=False))
            out=out.iloc[chosen].copy()
        elif name=='rotate':
            centers=shapely.get_coordinates(shapely.centroid(out.geometry.array))
            center=centers.mean(0)
            metric=np.array([np.cos(np.deg2rad(center[1])),1.]) if out.crs.is_geographic else np.ones(2)
            angle=np.deg2rad(op['degrees']); rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            out.geometry=shapely.transform(out.geometry.array,lambda x:((x-center)*metric)@rot.T/metric+center)
        else: raise ValueError('Operation not implemented in S1: '+name)
        if out.empty or out.geometry.is_empty.any() or not out.geometry.is_valid.all():
            raise ValueError('Attack produced invalid geometry; do not silently repair')
        records.append(dict(**op,remaining_features=len(out)))
    return out,dict(operations=records,original_features=len(frame),final_features=len(out),actual_deleted_fraction=1-len(out)/len(frame))

def write_and_reload(frame,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    frame.to_file(path,driver='ESRI Shapefile',engine='pyogrio',encoding='UTF-8',index=False)
    check=gpd.read_file(path,engine='pyogrio')
    assert len(check)==len(frame) and check.crs==frame.crs
    assert check.geometry.is_valid.all() and not check.geometry.is_empty.any()
    assert np.all(shapely.equals(frame.geometry.array,check.geometry.array)), 'Shapefile geometry changed on write'
    if 'source_id' in frame: assert frame['source_id'].astype(str).tolist()==check['source_id'].astype(str).tolist()
    return check
