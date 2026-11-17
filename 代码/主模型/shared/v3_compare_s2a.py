import hashlib

def seed_for(label): return int(hashlib.sha256(('comparison-v1:'+label).encode()).hexdigest()[:8],16)
