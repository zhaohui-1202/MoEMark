import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE / '主模型'))


def check_assets(root, protocol):
    paths = [
        'data/file_corpus_v2/seven_layer_manifest.json',
        'results/dual_moe_single_watermark_v1/manifest.json',
        'results/dual_moe_single_watermark_v1/summary_scaler.json',
        'results/dual_binding_A_v1/experiment.key',
        'results/transformer_moe_v3_s2a/expert_ablation_v1/full_seed42/vector_final.pt',
        'results/transformer_moe_v3_s2a/fusion_A_spatial_identity_v1/spatial_image_features.pt',
        'assets/mamba_image_encoder_final.pth',
    ]
    for seed in protocol['detector_seeds']:
        paths.extend([f'results/dual_paper_ablation_6x20_v1/A1_{seed}/deployment.pt',
                      f'results/dual_paper_ablation_6x20_v1/A1_{seed}/sparse_threshold.json'])
    paths.extend('assets/copyright/' + n for n in protocol['images20'])
    missing = [str(root / p) for p in paths if not (root / p).is_file()]
    corpus_path = root / paths[0]
    if corpus_path.is_file():
        rows = sorted(json.loads(corpus_path.read_text())['datasets'], key=lambda r: r['dataset_id'])
        owners = [i for i, r in enumerate(rows) if r['dataset_id'] in protocol['dev_ids']]
        if len(owners) != 6:
            missing.append('Six evaluation maps are not all present in the corpus manifest.')
        for i in owners:
            p = Path(rows[i]['shp'])
            prefix = Path('/root/autodl-tmp/vector_zw_20261020')
            try:
                p = root / p.relative_to(prefix)
            except ValueError:
                pass
            for suffix in ['.shp', '.shx', '.dbf', '.prj']:
                if not p.with_suffix(suffix).is_file():
                    missing.append(str(p.with_suffix(suffix)))
        manifest = root / 'results/dual_moe_single_watermark_v1/manifest.json'
        if manifest.is_file():
            clean = {r['owner']: r['key'] for r in json.loads(manifest.read_text()) if r['split'] == 'clean'}
            for owner in owners:
                if owner not in clean:
                    missing.append(f'Clean feature entry for corpus owner {owner}')
                    continue
                p = manifest.parent / 'features' / (clean[owner] + '.npz')
                if not p.is_file():
                    missing.append(str(p))
    deps = ['numpy', 'torch', 'torchvision', 'PIL', 'scipy', 'pandas', 'geopandas', 'shapely', 'pyogrio', 'matplotlib']
    absent = [name for name in deps if importlib.util.find_spec(name) is None]
    print('Missing packages:', ', '.join(absent) or 'none')
    print('Missing assets:', len(missing))
    for path in missing:
        print('  ', path)
    print('This check does not initialize CUDA or change server files.')
    return not missing and not absent


def main():
    p = argparse.ArgumentParser(description='MoEMark final-model robustness and discriminability')
    p.add_argument('--stage', choices=['check', 'all', 'robustness', 'discriminability', 'plots', 'register', 'verify'], default='check')
    p.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/vector_zw_20261020'))
    p.add_argument('--output', type=Path, help='New results directory; never overwrites the historical evaluation directories')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--case-ids', type=int, nargs='+', help='Optional smoke run; produces robustness_subset rather than a 92-setting summary')
    p.add_argument('--published-results', type=Path, help='Plot the existing local results folder; no GPU or model assets required')
    p.add_argument('--map', type=Path)
    p.add_argument('--image', type=Path)
    p.add_argument('--record', type=Path)
    p.add_argument('--seed', type=int, choices=[42, 73, 101], default=42)
    args = p.parse_args()
    if args.workers < 1:
        p.error('--workers must be positive')
    protocol_path = HERE / '评测协议' / 'single92.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if args.case_ids and not set(args.case_ids) <= {c['id'] for c in protocol['cases']}:
        p.error('Unknown case ID')
    if args.stage == 'check':
        return 0 if check_assets(args.root, protocol) else 1
    if args.stage == 'plots':
        from 绘制结果 import plot_results
        output = args.output or (HERE.parent / '结果' / '主模型_重新绘图' if args.published_results else args.root / 'results/moemark_final_evaluation')
        plot_results(output, protocol, args.published_results)
        return 0
    output = args.output or args.root / 'results/moemark_final_evaluation'
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(name, '1')
    if args.stage in ['register', 'verify']:
        if not all([args.map, args.image, args.record]):
            p.error('register/verify require --map, --image, and --record')
        from pipeline import System, read, save
        model = System(args.root, args.seed)
        if args.stage == 'register':
            if args.record.exists():
                p.error('Registration record already exists; choose a new path')
            save(args.record, model.register(args.map, args.image))
            print('Saved registration record:', args.record)
        else:
            print(json.dumps(model.verify(args.map, args.image, read(args.record)), indent=2))
        return 0
    from evaluate import Evaluation
    import torch
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another evaluation is writing to this output directory')
        evaluation = Evaluation(args.root, protocol_path, output, args.workers)
        if args.stage in ['all', 'discriminability']:
            evaluation.discriminability()
        if args.stage in ['all', 'robustness']:
            evaluation.robustness(args.case_ids)
        from 绘制结果 import plot_results
        plot_results(output, protocol)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
