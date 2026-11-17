import geopandas as gpd

import numpy as np

import shapely

def read_metric(path):
    frame=gpd.read_file(path,engine='pyogrio')
    if frame.crs is None or frame.empty: raise ValueError('Missing CRS or empty file')
    if not frame.geometry.is_valid.all() or frame.geometry.is_empty.any():
        raise ValueError('Invalid source input; no silent deletion')
    centroids=shapely.get_coordinates(shapely.centroid(frame.geometry.array))
    center=centroids.mean(0)
    if frame.crs.is_geographic:
        if frame.crs.to_epsg()!=4326: raise ValueError('Geographic CRS must be EPSG:4326')
        metric=np.array([np.cos(np.deg2rad(center[1])),1.])
    else:
        metric=np.ones(2)
    scale=max(float(np.sqrt(np.mean(np.sum(((centroids-center)*metric)**2,axis=1)))),1e-8)


    return frame,center,scale,metric
