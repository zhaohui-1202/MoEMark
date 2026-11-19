import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / 'shared'))
from models import PlainRegistration, Detection
from image_binding import identity_code, bound_bits
from map_geometry_binding import category_code, geometry_category, protocol_id
from feature_mapping import SpatialFiLM, paired
from geometry_encoder import StructuralMoE, subset_anchors

DEFAULT_ROOT = Path('/root/autodl-tmp/vector_zw_20261020')
FEATURE_RUN = 'results/dual_moe_single_watermark_v1'
MODEL_RUN = 'results/dual_paper_ablation_6x20_v1'
BACKBONE = 'results/transformer_moe_v3_s2a/expert_ablation_v1/full_seed42/vector_final.pt'
IMAGE_BANK = 'results/transformer_moe_v3_s2a/fusion_A_spatial_identity_v1/spatial_image_features.pt'
KEY = 'results/dual_binding_A_v1/experiment.key'
PAYLOAD_PROTOCOL = 'gray-nearest16-th128-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def corpus(root):
    rows = sorted(read(root / 'data/file_corpus_v2/seven_layer_manifest.json')['datasets'], key=lambda r: r['dataset_id'])
    for row in rows:
        path = Path(row['shp'])
        try:
            row['shp'] = str(root / path.relative_to(DEFAULT_ROOT))
        except ValueError:
            pass
    return rows


def payload(path):
    with Image.open(path) as im:
        return (np.asarray(im.convert('L').resize((16, 16), Image.Resampling.NEAREST)) >= 128).reshape(256)


def scores(reference, recovered):

    ref, rec = reference.bool(), recovered.bool()
    nc = F.cosine_similarity(ref.float(), rec.float(), dim=-1)
    bcr = (ref == rec).float().mean(-1)
    return nc, bcr


class System:
    def __init__(self, root, seed=42, device='cuda'):
        self.root, self.device = Path(root), torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Full evaluation requires CUDA. Use --stage check or --stage plots without a GPU.')
        self.model_dir = self.root / MODEL_RUN / f'A1_{seed}'
        cp_path = self.model_dir / 'deployment.pt'
        cp = torch.load(cp_path, map_location=self.device, weights_only=True)
        self.version = sha(cp_path)
        self.scaler = cp['summary_scaler']
        self.backbone_hash = cp['backbone_sha256']
        self.reg, self.det, self.head = PlainRegistration(), Detection(), SpatialFiLM()
        for module, key in [(self.reg, 'registration'), (self.det, 'detector'), (self.head, 'readout')]:
            module.load_state_dict(cp[key], strict=True)
            module.to(self.device).eval().requires_grad_(False)
        self.key = (self.root / KEY).read_bytes()
        if len(self.key) != 32:
            raise ValueError('Expected the existing 32-byte binding key; do not generate a new key.')
        self.binding = protocol_id(self.key)
        self.backbone = None
        self.image_encoder = None

    def tensor(self, raw):
        raw = np.asarray(raw, dtype=np.float32).copy()
        raw[..., -18:] = (raw[..., -18:] - np.asarray(self.scaler['mean'], np.float32)) / np.asarray(self.scaler['std'], np.float32)
        return torch.tensor(raw, device=self.device)

    @torch.inference_mode()
    def features(self, raw, images, registration=False):
        x = self.tensor(raw)
        z = self.reg(x)[0] if registration else self.det(self.reg.encode(x), x[:, -18:])[0]
        return paired(self.head, z, images)['logits']

    def bind(self, logits, image_codes, category):
        geometry = torch.tensor(category_code(category, self.key), device=self.device)
        return bound_bits(logits, image_codes) ^ geometry

    def load_backbone(self):
        if self.backbone is None:
            path = self.root / BACKBONE
            if sha(path) != self.backbone_hash:
                raise ValueError('Frozen backbone checksum differs from deployment checkpoint.')


            self.backbone = StructuralMoE(routing='dense', checkpoint_geometry=True).to(self.device).eval().requires_grad_(False)
            state = torch.load(path, map_location='cpu', weights_only=True)
            self.backbone.load_state_dict(state['encoder'], strict=True)
            self.capture = {}
            self.backbone.feature.register_forward_hook(lambda m, a, o: self.capture.update(h=o.detach()))

    @torch.inference_mode()
    def extract(self, sample):
        self.load_backbone()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, _, experts = self.backbone([sample])
        anchors = torch.tensor(sample['anchors'], device=self.device)
        local = self.capture['h'].float()[anchors].mean(0).cpu().numpy()
        return np.r_[local, experts[0, 2].float().cpu().numpy(), sample['summary']].astype(np.float32)

    @torch.inference_mode()
    def encode_image(self, path):
        if self.image_encoder is None:
            from image_encoder import ImageEncoder
            self.image_encoder = ImageEncoder(pretrained=False).to(self.device)
            path_weights = self.root / 'assets/mamba_image_encoder_final.pth'
            state = torch.load(path_weights, map_location='cpu', weights_only=True)
            self.image_encoder.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state.items()}, strict=True)
            self.image_encoder.eval().requires_grad_(False)
            self.image_version = sha(path_weights)
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB').resize((224, 224)), dtype=np.float32) / 255
        seq = self.image_encoder(torch.tensor(rgb, device=self.device).permute(2, 0, 1)[None])
        grid = seq.transpose(1, 2).reshape(1, 256, 8, 8)
        spatial = F.adaptive_avg_pool2d(grid, (2, 2)).permute(0, 2, 3, 1).reshape(1, 1024)
        return torch.cat([seq.mean(1), spatial], -1)

    def encode_map(self, path):
        with tempfile.TemporaryDirectory(prefix='moemark_query_') as temp:
            sample, audit = prepare_candidate(({'shp': str(path)}, None, 'clean'), temp)
            return self.extract(sample)[None], audit['geometry_category']

    @torch.inference_mode()
    def register(self, map_path, image_path):
        raw, category = self.encode_map(map_path)
        image = self.encode_image(image_path)
        bits = self.bind(self.features(raw, image, True)[0, 0], identity_code(image_path, self.key).to(self.device), category)
        pay = torch.tensor(payload(image_path), device=self.device)
        return dict(watermark=(bits ^ pay).cpu().tolist(), model_version=self.version,
                    image_encoder_version=self.image_version, binding=self.binding, payload_protocol=PAYLOAD_PROTOCOL)

    @torch.inference_mode()
    def verify(self, map_path, image_path, record):
        image = self.encode_image(image_path)
        for key, value in [('model_version', self.version), ('image_encoder_version', self.image_version), ('binding', self.binding), ('payload_protocol', PAYLOAD_PROTOCOL)]:
            if record.get(key) != value:
                raise ValueError(f'Registration record mismatch: {key}')
        threshold = read(self.model_dir / 'sparse_threshold.json')
        if not threshold['feasible'] or not isinstance(threshold['threshold'], int) or not 0 <= threshold['threshold'] <= 256:
            raise ValueError('Invalid acceptance threshold')
        raw, category = self.encode_map(map_path)
        bits = self.bind(self.features(raw, image)[0, 0], identity_code(image_path, self.key).to(self.device), category)
        watermark = torch.tensor(record['watermark'], dtype=torch.bool, device=self.device)
        if watermark.shape != (256,):
            raise ValueError('Expected a 256-bit registered zero-watermark')
        recovered = watermark ^ bits
        pay = torch.tensor(payload(image_path), device=self.device)
        nc, bcr = scores(pay, recovered)
        matching = int((pay == recovered).sum())
        return dict(nc=float(nc), bcr=float(bcr), matching_bits=matching,
                    threshold=threshold['threshold'], accepted=matching >= threshold['threshold'])


def prepare_candidate(job, output_dir):

    from map_preprocessing import read_metric
    from vector_data import build_cache, write_and_reload
    from attack_suite import align_principal
    from structural_attacks import custom_attack
    from random_seed import seed_for
    row, operation, key = job
    frame, center, scale, metric = read_metric(row['shp'])
    attack = []
    if operation is not None:
        try:
            frame, aa = custom_attack(frame, operation, seed_for(f'dual-v1:{key}:0'), center, scale, metric)
            attack.append(aa)
        except ValueError as error:
            if operation['name'] == 'quantize' and ('invalid' in str(error).lower() or 'empty' in str(error).lower()):
                return None, {'invalid': str(error)}
            raise
    with tempfile.TemporaryDirectory(dir=output_dir, prefix='candidate_') as temp:
        path = Path(temp) / 'candidate.shp'
        write_and_reload(frame, path)
        candidate, c, s, m = read_metric(path)
        aligned, alignment = align_principal(candidate, c, s, m)
        cache, cache_audit = build_cache(aligned, c, s, m, 20261105, 1024, 8)
        rng = np.random.default_rng(20261105)
        sample = subset_anchors(cache, rng.choice(len(cache['anchors']), min(512, len(cache['anchors'])), replace=False))
        audit = dict(attack=attack, alignment=alignment, cache_audit=cache_audit,
                     geometry_category=geometry_category(candidate.geometry.array),
                     geometry_sha256=hashlib.sha256(b''.join(candidate.geometry.to_wkb())).hexdigest())
    return sample, audit
