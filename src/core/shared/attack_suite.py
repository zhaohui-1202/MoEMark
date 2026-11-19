import numpy as np

import pandas as pd

import geopandas as gpd

import shapely

from shapely.geometry import Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon, box

from attack_operations import execute, distance_unit

def map_geometry(g, fn):
    kind=g.geom_type
    if kind=='Point':return Point(fn(np.asarray(g.coords),False))
    if kind=='LineString':return LineString(fn(np.asarray(g.coords),g.is_closed))
    if kind=='Polygon':
        return Polygon(fn(np.asarray(g.exterior.coords),True),
                       [fn(np.asarray(r.coords),True) for r in g.interiors])
    constructors={'MultiPoint':MultiPoint,'MultiLineString':MultiLineString,'MultiPolygon':MultiPolygon}
    if kind in constructors:return constructors[kind]([map_geometry(p,fn) for p in g.geoms])
    raise ValueError('Unsupported geometry '+kind)

def sequence_edit(a,closed,op,rng,metric,ell):
    a=np.asarray(a,dtype=np.float64)[:,:2];name=op['name']
    u=a[:-1].copy() if closed else a.copy()
    n=len(u)
    if name in ('gaussian_noise','uniform_noise','outlier_noise'):
        if name=='gaussian_noise':
            sigma=op['sigma_ratio']*ell;d=rng.normal(0,sigma,(n,2))
            d*=np.minimum(1,3*sigma/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-30))
        elif name=='uniform_noise':
            bound=np.sqrt(3)*op['sigma_ratio']*ell;d=rng.uniform(-bound,bound,(n,2))
        else:
            selected=rng.random(n)<op['fraction'];angles=rng.uniform(0,2*np.pi,n)
            d=np.column_stack([np.cos(angles),np.sin(angles)])*op['distance_ratio']*ell*selected[:,None]
        u+=d/metric
    elif name=='vertex_delete':
        eligible=np.arange(n) if closed else np.arange(1,n-1)
        expected=n*op['fraction'];requested=int(np.floor(expected))+int(rng.random()<expected-np.floor(expected))
        count=min(max(0,n-(3 if closed else 2)),requested,len(eligible))
        if count:u=np.delete(u,rng.choice(eligible,count,replace=False),axis=0)
    elif name=='vertex_add':
        segment_count=len(a)-1
        count=min(segment_count,int(round(segment_count*op['fraction'])))


        expected=segment_count*op['fraction'];count=int(np.floor(expected))
        count+=int(rng.random()<expected-count)
        chosen=set(rng.choice(segment_count,count,replace=False).tolist()) if count else set()
        points=[]
        for i in range(segment_count):
            points.append(a[i])
            if i in chosen:points.append((a[i]+a[i+1])/2)
        points.append(a[-1]);return np.asarray(points)
    elif name=='reverse':u=u[::-1]
    elif name=='ring_reverse':
        if closed:u=u[::-1]
    elif name=='ring_start':
        if closed:u=np.roll(u,int(rng.integers(n)),axis=0)
    else:raise ValueError(name)
    return np.vstack([u,u[0]]) if closed else u

def family(g):
    if 'Point' in g.geom_type:return 'point'
    if 'LineString' in g.geom_type:return 'line'
    if 'Polygon' in g.geom_type:return 'polygon'
    return None

def extract_family(g,kind):
    if g.is_empty:return None
    if family(g)==kind:return g
    if g.geom_type!='GeometryCollection':return None
    parts=[]
    for child in g.geoms:
        item=extract_family(child,kind)
        if item is not None:
            parts.extend(list(item.geoms) if hasattr(item,'geoms') else [item])
    if not parts:return None
    if len(parts)==1:return parts[0]
    return {'point':MultiPoint,'line':MultiLineString,'polygon':MultiPolygon}[kind](parts)

def apply(frame,op,seed,center,scale,metric,donor=None,donor_id=None):
    name=op['name'];rng=np.random.default_rng(seed);out=frame.copy()
    ell,fallback=distance_unit(frame,metric,scale)
    info=dict(operations=[op],seed=seed,ell=float(ell),ell_fallback=fallback,R=float(scale),
              input_features=len(frame),input_vertices=int(shapely.get_num_coordinates(frame.geometry.array).sum()))
    if name in ('rotate','delete','feature_jitter','simplify','quantize'):
        out,other=execute(frame,[op],seed,metric,scale);info.update(other)
    elif name in ('translate','scale','stretch','shear'):
        def affine(xy):
            x=(xy-center)*metric
            if name=='translate':
                direction={'x':[1,0],'y':[0,1],'diagonal':[2**-.5,2**-.5]}[op['axis']]
                x=x+np.asarray(direction)*op['ratio']*scale
            elif name=='scale':x=x*op['factor']
            elif name=='stretch':x=x*np.array([op['factor'],1] if op['axis']=='x' else [1,op['factor']])
            else:
                x=x.copy();i=0 if op['axis']=='x' else 1
                x[:,i]+=np.tan(np.deg2rad(op['degrees']))*x[:,1-i]
            return x/metric+center
        out.geometry=shapely.transform(frame.geometry.array,affine)
    elif name in ('gaussian_noise','uniform_noise','outlier_noise','vertex_add','vertex_delete','reverse','ring_reverse','ring_start','quantize_valid','part_order'):
        geoms=[];rejected=0;edited=0;moved=0;displacement_ss=0.;displacement_max=0.;coordinate_count=0
        for g in frame.geometry.array:
            if name=='quantize_valid':
                grid=op['step_ratio']*ell
                candidate=shapely.transform(g,lambda x:np.round((x-center)*metric/grid)*grid/metric+center)
            elif name=='part_order':
                if not hasattr(g,'geoms'):candidate=g
                else:
                    ps=list(g.geoms);order=rng.permutation(len(ps));constructor=type(g)
                    candidate=constructor([ps[i] for i in order])
            else:candidate=map_geometry(g,lambda a,c:sequence_edit(a,c,op,rng,metric,ell))
            if candidate.is_empty or not candidate.is_valid:
                rejected+=1;candidate=g
            changed=not shapely.equals_exact(candidate,g,0)
            edited+=int(changed);geoms.append(candidate)
            if name in ('gaussian_noise','uniform_noise','outlier_noise','quantize_valid'):
                a=shapely.get_coordinates(g);b=shapely.get_coordinates(candidate)
                assert a.shape==b.shape
                d=np.linalg.norm((b-a)*metric,axis=1)
                moved+=int((d>0).sum());displacement_ss+=float((d*d).sum())
                displacement_max=max(displacement_max,float(d.max(initial=0)));coordinate_count+=len(d)
        out.geometry=geoms
        info.update(constraint_policy='one proposal per feature; invalid proposal retains original feature; no repair or retry',
                    rejected_feature_proposals=rejected,unchanged_features=len(frame)-edited,
                    actual_edited_features=edited,actual_edited_feature_fraction=edited/len(frame),
                    no_effect=edited==0)
        if coordinate_count:
            info.update(actual_moved_vertex_fraction=moved/coordinate_count,
                        actual_rms_displacement=float(np.sqrt(displacement_ss/coordinate_count)),
                        actual_max_displacement=displacement_max,
                        displacement_unit='local affine coordinate units; geographic inputs are degree-scaled, not metres')
    elif name=='feature_add':
        if donor is None:raise ValueError('Feature addition requires a separately identified same-layer donor')
        assert donor.crs==frame.crs
        count=max(1,int(round(len(frame)*op['fraction'])))
        chosen=rng.choice(len(donor),count,replace=count>len(donor));added=donor.iloc[chosen].copy()
        dg=np.asarray(added.geometry.array,dtype=object).copy()
        dc=shapely.get_coordinates(shapely.centroid(dg))
        donor_lat=shapely.get_coordinates(shapely.centroid(donor.geometry.array)).mean(0)[1]
        dm=np.array([np.cos(np.deg2rad(donor_lat)),1.]) if frame.crs.is_geographic else np.ones(2)
        lo,hi=frame.total_bounds[:2],frame.total_bounds[2:]
        destination=rng.uniform(lo,hi,(count,2))
        coords,owner=shapely.get_coordinates(dg,return_index=True)
        added.geometry=shapely.set_coordinates(dg,(coords-dc[owner])*dm/metric+destination[owner])

        if 'source_id' in frame:
            out['source_id']=out['source_id'].astype(str)
            added['source_id']=[f'ADD_{seed}_{i}' for i in range(count)]
        out=gpd.GeoDataFrame(pd.concat([out,added],ignore_index=True),geometry=frame.geometry.name,crs=frame.crs)
        info.update(donor_id=donor_id,donor_sample_with_replacement=count>len(donor),
                    placement='independent uniform positions inside target bounding box; donor local shapes retained in affine metric',
                    added_features=count,actual_added_fraction=count/len(frame))
    elif name=='cluster_delete':
        centers=shapely.get_coordinates(shapely.centroid(frame.geometry.array))*metric
        selected=int(rng.integers(len(frame)));d=((centers-centers[selected])**2).sum(1)
        count=min(len(frame)-2,int(round(len(frame)*op['fraction'])))
        remove=np.argsort(d,kind='stable')[:count];keep=np.ones(len(frame),bool);keep[remove]=False
        out=frame.iloc[np.flatnonzero(keep)].copy();info['cluster_center_row']=selected
    elif name=='crop':
        lo,hi=frame.total_bounds[:2],frame.total_bounds[2:];middle=(lo+hi)/2
        half=(hi-lo)*np.sqrt(op['retained_area'])/2
        window=box(*(middle-half),*(middle+half));kind=family(frame.geometry.iloc[0])
        clipped=shapely.intersection(frame.geometry.array,window)
        keep=[];geoms=[];lower_dim=0
        for i,g in enumerate(clipped):
            item=extract_family(g,kind)
            if item is not None:keep.append(i);geoms.append(item)
            elif not g.is_empty:lower_dim+=1
        out=frame.iloc[keep].copy();out.geometry=geoms
        info.update(window_bounds=list(window.bounds),window_area_retained=op['retained_area'],
                    removed_empty_or_outside=len(frame)-len(out)-lower_dim,removed_boundary_only_lower_dimension=lower_dim)
    elif name=='reorder':out=frame.iloc[rng.permutation(len(frame))].copy()
    elif name=='rewrite':pass
    else:raise ValueError('Unknown single attack '+name)
    if len(out)<2:raise ValueError('Attack leaves fewer than two features')
    bad=out.geometry.is_empty|~out.geometry.is_valid
    if bad.any():raise ValueError(f'{name}: {int(bad.sum())} invalid geometries; no repair')
    if out.crs.is_geographic:
        b=out.total_bounds
        if b[0]<-180 or b[2]>180 or b[1]<-90 or b[3]>90:raise ValueError('Coordinates outside geographic range')
    info.update(output_features=len(out),output_vertices=int(shapely.get_num_coordinates(out.geometry.array).sum()),
                feature_retention=len(out)/len(frame),valid=True)
    info['vertex_count_change_fraction']=(info['output_vertices']-info['input_vertices'])/max(1,info['input_vertices'])
    for quantity,fn in [('length',shapely.length),('area',shapely.area)]:
        before=float(fn(frame.geometry.array).sum());after=float(fn(out.geometry.array).sum())
        info[quantity+'_retention']=after/before if before>0 else None
    return out,info

def align_principal(frame,center,scale,metric):

    xy=(shapely.get_coordinates(shapely.centroid(frame.geometry.array))-center)*metric/scale
    covariance=xy.T@xy/max(1,len(xy));values,vectors=np.linalg.eigh(covariance)
    axis=vectors[:,-1];projection=xy@axis;third=float(np.mean(projection**3))
    if abs(third)>1e-8:sign=1 if third>=0 else -1;rule='third moment'
    else:
        extreme=int(np.argmax(np.abs(projection)));sign=1 if projection[extreme]>=0 else -1;rule='farthest projected centroid; ties use source order'
    axis=axis*sign;rotation=np.column_stack([axis,[-axis[1],axis[0]]])
    result=frame.copy()
    result.geometry=shapely.transform(frame.geometry.array,lambda a:((a-center)*metric)@rotation/metric+center)
    gap=float((values[-1]-values[0])/max(values[-1],1e-20))
    audit=dict(method='full-file centroid PCA; proper rotation determinant +1; candidate-only',
               eigenvalues=values.tolist(),relative_eigengap=gap,ambiguous=gap<.001,
               sign_rule=rule,axis=axis.tolist(),center=center.tolist(),scale=scale)
    return result,audit
