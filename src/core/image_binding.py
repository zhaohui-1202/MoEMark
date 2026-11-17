import hashlib
import hmac
import struct

import numpy as np
import torch
from PIL import Image


def identity_code(path, key):
    if len(key) != 32:
        raise ValueError('Expected a 32-byte HMAC key')
    with Image.open(path) as im:
        rgb = im.convert('RGB')
        digest = hashlib.sha256(b'zw-rgb8-v1\0' + struct.pack('>II', *rgb.size))
        digest.update(rgb.tobytes())
    code = hmac.digest(key, digest.digest(), 'sha256')
    return torch.from_numpy(np.unpackbits(np.frombuffer(code, dtype=np.uint8)).astype(bool))


def modulate(logits, code):
    return torch.where(code, -logits, logits)


def bound_bits(logits, code):

    return (logits >= 0) ^ code
