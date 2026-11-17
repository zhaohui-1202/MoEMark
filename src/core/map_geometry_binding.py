import hashlib
import hmac
import numpy as np
import shapely

DOMAIN=b'map-geometry-category-v1\0'

def geometry_category(geometries):
    geometries=np.asarray(geometries,dtype=object)
    if not len(geometries) or np.any(shapely.is_empty(geometries)) or not np.all(shapely.is_valid(geometries)):
        raise ValueError('Geometry binding requires nonempty valid geometries')
    parts=shapely.get_parts(geometries)
    types=set(shapely.get_type_id(parts).tolist())
    if types=={0}:return 'point'
    if types=={1}:return 'line'
    if types=={3}:return 'area'
    raise ValueError('Mixed or unsupported geometry families are outside the validated scope')

def category_code(category,key):
    if category not in ('point','line','area'):raise ValueError('Unknown geometry category')
    if len(key)!=32:raise ValueError('Expected the existing 32-byte image-binding key')
    digest=hmac.digest(key,DOMAIN+category.encode('ascii'),'sha256')
    return np.unpackbits(np.frombuffer(digest,np.uint8)).astype(bool)

def bind_bits(image_bound_bits,geometries,key):
    bits=np.asarray(image_bound_bits)
    if bits.dtype!=bool or bits.shape[-1]!=256:raise ValueError('Expected 256 boolean fingerprint bits')
    return bits^category_code(geometry_category(geometries),key)

def protocol_id(key):
    if len(key)!=32:raise ValueError('Expected a 32-byte key')
    return 'image-and-geometry-category-v1:'+hashlib.sha256(key).hexdigest()[:16]
