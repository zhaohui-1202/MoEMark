import numpy as np

import torch,shapely

from v3_attack_suite import apply,map_geometry,align_principal

def custom_attack(frame,op,seed,c,s,m):
    name=op['name']
    if name not in ['mirror','vertex_reverse','object_reverse','vertex_shuffle_valid']:
        return apply(frame,op,seed,c,s,m)
    rng=np.random.default_rng(seed);out=frame.copy()
    audit=dict(op=op,seed=seed,input_features=len(frame))
    if name=='mirror':

        signs={'x':[1,-1],'y':[-1,1],'xy':[-1,-1]}[op['axis']]
        out.geometry=shapely.transform(frame.geometry.array,lambda a:(a-c)*np.asarray(signs)+c)
        audit.update(center=c.tolist(),meaning='x flips Y; y flips X; xy flips both',signs=signs)
    elif name=='object_reverse':out=frame.iloc[::-1].copy()
    else:
        accepted=[];rejected=0;sequence_changed=0;shape_changed=0;eligible_total=0;requested_total=0;actual_moved=0
        for g in frame.geometry.array:
            local=dict(eligible=0,requested=0,moved=0)
            def edit(a,closed):
                a=np.asarray(a)[:,:2]
                u=a[:-1].copy() if closed else a.copy()
                original=u.copy();n=len(u)
                if name=='vertex_reverse':
                    u=u[::-1];local['eligible']+=n;local['requested']+=n
                else:
                    eligible=np.arange(n) if closed else np.arange(1,max(1,n-1))
                    local['eligible']+=len(eligible)
                    expected=len(eligible)*op['fraction'];k=int(np.floor(expected))+int(rng.random()<expected-np.floor(expected))
                    local['requested']+=k
                    if k>=2:
                        ix=rng.choice(eligible,k,replace=False);u[ix]=u[ix][rng.permutation(k)]
                local['moved']+=int(np.any(u!=original,axis=1).sum())
                return np.vstack([u,u[0]]) if closed else u
            candidate=map_geometry(g,edit)
            eligible_total+=local['eligible'];requested_total+=local['requested']
            if candidate.is_empty or not candidate.is_valid:
                rejected+=1;candidate=g
            else:actual_moved+=local['moved']
            sequence_changed+=int(not shapely.equals_exact(g,candidate,0))
            shape_changed+=int(not shapely.equals(g,candidate))
            accepted.append(candidate)
        out.geometry=accepted
        audit.update(policy='one proposal per feature; invalid geometry retains original, no repair or retries',
                     line_endpoints='preserved for random shuffle; reverse swaps orientation',
                     invalid_feature_proposals=rejected,invalid_proposal_fraction=rejected/len(frame),
                     actual_sequence_changed_features=sequence_changed,actual_sequence_changed_fraction=sequence_changed/len(frame),
                     actual_geometry_changed_features=shape_changed,actual_geometry_changed_fraction=shape_changed/len(frame),
                     eligible_vertices=eligible_total,requested_vertices=requested_total,actual_moved_vertices=actual_moved,
                     actual_moved_vertex_fraction=actual_moved/max(1,eligible_total),no_effect=sequence_changed==0)
    assert out.geometry.is_valid.all() and not out.geometry.is_empty.any()
    audit.update(output_features=len(out),valid=True)
    return out,audit
