import csv
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from image_binding import identity_code
from map_geometry_binding import geometry_category
from pipeline import System, FEATURE_RUN, IMAGE_BANK, corpus, payload, prepare_candidate, read, save, scores, sha


def worker_init():
    torch.set_num_threads(1)


def bounded_jobs(jobs, output, workers):

    if workers == 1:
        for job in jobs:
            yield job, prepare_candidate(job['input'], output)
        return
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=worker_init) as pool:
        it = iter(jobs)
        queue = []
        for _ in range(workers):
            job = next(it, None)
            if job is not None:
                queue.append((job, pool.submit(prepare_candidate, job['input'], output)))
        while queue:
            job, future = queue.pop(0)
            yield job, future.result()
            new = next(it, None)
            if new is not None:
                queue.append((new, pool.submit(prepare_candidate, new['input'], output)))


class Evaluation:
    def __init__(self, root, protocol, output, workers=6):
        self.root, self.output = Path(root), Path(output)
        self.protocol = read(protocol)
        self.workers = workers
        self.output.mkdir(parents=True, exist_ok=True)
        self.rows = corpus(self.root)
        all_ids = [r['dataset_id'] for r in self.rows]
        self.owners = [all_ids.index(fid) for fid in self.protocol['dev_ids']]
        self.files = [self.rows[i] for i in self.owners]
        self.systems = [System(self.root, seed) for seed in self.protocol['detector_seeds']]
        bank = torch.load(self.root / IMAGE_BANK, map_location='cpu', weights_only=True)
        names = self.protocol['images20']
        self.images = bank['features'][[bank['names'].index(n) for n in names]].float().cuda()
        self.payload = torch.tensor(np.stack([payload(self.root / 'assets/copyright' / n) for n in names]), device='cuda')
        self.codes = torch.stack([identity_code(self.root / 'assets/copyright' / n, self.systems[0].key) for n in names]).cuda()
        from prepare_file_training import read_metric
        self.categories = [geometry_category(read_metric(row['shp'])[0].geometry.array) for row in self.files]
        base = self.root / FEATURE_RUN
        manifest = read(base / 'manifest.json')
        clean = {r['owner']: r['key'] for r in manifest if r['split'] == 'clean'}
        self.raw = np.stack([np.load(base / 'features' / (clean[i] + '.npz'))['x'] for i in self.owners])
        scaler = read(base / 'summary_scaler.json')
        if not all(s.scaler == scaler and s.binding == self.systems[0].binding for s in self.systems):
            raise ValueError('Models do not share the same preprocessing and binding protocol.')
        sources = {}
        for row in self.files:
            p = Path(row['shp'])
            sources[row['dataset_id']] = {f.name: sha(f) for f in sorted(p.parent.glob(p.stem + '.*')) if f.is_file()}
        signature = dict(protocol=self.protocol, models={str(seed): s.version for seed, s in zip(self.protocol['detector_seeds'], self.systems)},
                         binding=self.systems[0].binding, image_bank_sha256=sha(self.root / IMAGE_BANK),
                         image_sha256={n: sha(self.root / 'assets/copyright' / n) for n in names},
                         clean_features={str(i): sha(base / 'features' / (clean[i] + '.npz')) for i in self.owners},
                         sources=sources, metric='binary recovered-watermark cosine NC and bit-correct rate',
                         payload='gray-nearest16-th128-v1', implementation='moemark-evaluation-v1')
        sig_path = self.output / 'protocol.json'
        if sig_path.exists() and read(sig_path) != signature:
            raise ValueError('Output belongs to different inputs/models/protocol. Choose a new --output directory.')
        save(sig_path, signature)
        self.reference = [s.features(self.raw, self.images, True) for s in self.systems]
        self.registered = [torch.stack([s.bind(ref[i], self.codes, cat) ^ self.payload for i, cat in enumerate(self.categories)])
                           for s, ref in zip(self.systems, self.reference)]

    @torch.inference_mode()
    def discriminability(self):
        map_nc, image_nc, map_bcr, image_bcr = [], [], [], []
        for s, registered in zip(self.systems, self.registered):
            logits = s.features(self.raw, self.images)
            candidate = torch.stack([s.bind(logits[i], self.codes, cat) for i, cat in enumerate(self.categories)])

            recovered = registered[:, None] ^ candidate[None, :]
            nc, bcr = scores(self.payload, recovered)
            map_nc.append(nc.mean(-1).cpu().numpy())
            map_bcr.append(bcr.mean(-1).cpu().numpy())


            recovered = registered[:, :, None] ^ candidate[:, None, :]
            nc, bcr = scores(self.payload[None, None], recovered)
            image_nc.append(nc.cpu().numpy())
            image_bcr.append(bcr.cpu().numpy())
        data = dict(map=dict(files=self.protocol['dev_ids'], nc=np.mean(map_nc, 0).tolist(), bcr=np.mean(map_bcr, 0).tolist()),
                    image=dict(images=self.protocol['images20'], nc=np.mean(image_nc, (0, 1)).tolist(),
                               bcr=np.mean(image_bcr, (0, 1)).tolist(), per_map_nc=np.mean(image_nc, 0).tolist()))
        save(self.output / 'discriminability.json', data)
        print('Saved 6 x 6 map matrix and 20 x 20 copyright-image matrix.', flush=True)

    def score_query(self, raw, file_index, category):
        per_seed = {}
        for seed, s, registered in zip(self.protocol['detector_seeds'], self.systems, self.registered):
            logits = s.features(raw[None], self.images)[0]
            recovered = registered[file_index] ^ s.bind(logits, self.codes, category)
            nc, bcr = scores(self.payload, recovered)
            per_seed[str(seed)] = dict(nc=float(nc.mean()), bcr=float(bcr.mean()))
        return dict(nc=float(np.mean([r['nc'] for r in per_seed.values()])),
                    bcr=float(np.mean([r['bcr'] for r in per_seed.values()])), seeds=per_seed)

    @torch.inference_mode()
    def robustness(self, case_ids=None):
        records = self.output / 'records'
        records.mkdir(exist_ok=True)
        cases = [c for c in self.protocol['cases'] if not case_ids or c['id'] in case_ids]
        jobs = []
        for case in cases:
            for fi, (owner, row) in enumerate(zip(self.owners, self.files)):
                if row['kind'] not in case['applies']:
                    continue
                for repeat in range(case['repeats']):
                    key = f"full94_{case['source_id']:02d}_{owner}_{repeat}"
                    jobs.append(dict(key=key, case_id=case['id'], file_index=fi,
                                     file=row['dataset_id'], repeat=repeat, input=(row, case['op'], key)))
        remaining = [j for j in jobs if not (records / (j['key'] + '.json')).exists()]
        done = len(jobs) - len(remaining)
        for job, (sample, audit) in bounded_jobs(remaining, str(self.output), self.workers):
            result = {k: job[k] for k in ['key', 'case_id', 'file', 'repeat']}
            result['audit'] = audit
            if sample is not None:
                raw = self.systems[0].extract(sample)
                result.update(self.score_query(raw, job['file_index'], audit['geometry_category']))
                result['status'] = 'scored'
            else:
                result['status'] = 'invalid_geometry'
            save(records / (job['key'] + '.json'), result)
            done += 1
            save(self.output / 'status.json', dict(completed=done, planned=len(jobs)))
            print(f'Robustness {done}/{len(jobs)}: {job["key"]}', flush=True)
        grouped = defaultdict(lambda: defaultdict(list))
        invalid = []
        for job in jobs:
            row = read(records / (job['key'] + '.json'))
            if row['status'] == 'scored':
                grouped[row['case_id']][row['file']].append(row)
            else:
                invalid.append(row['key'])
        summary = []
        for case in cases:
            by_map = grouped[case['id']]
            if not by_map:
                raise ValueError(f'No valid records for setting {case["id"]}')
            summary.append(dict(case_id=case['id'], attack=case['name'], op=case['op'],
                                nc=float(np.mean([np.mean([r['nc'] for r in rs]) for rs in by_map.values()])),
                                bcr=float(np.mean([np.mean([r['bcr'] for r in rs]) for rs in by_map.values()])),
                                maps=len(by_map), records=sum(map(len, by_map.values()))))
        result = dict(settings=len(summary), planned_records=len(jobs), invalid_geometry=invalid,
                      complete_92=len(summary) == 92, nc=float(np.mean([r['nc'] for r in summary])),
                      bcr=float(np.mean([r['bcr'] for r in summary])), cases=summary)
        stem = 'robustness' if len(summary) == 92 else 'robustness_subset'
        save(self.output / (stem + '.json'), result)
        with (self.output / (stem + '.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
            fields = ['case_id', 'attack', 'nc', 'bcr', 'maps', 'records']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summary)
        print(f'{len(summary)} settings: NC={result["nc"]:.6f}, BCR={100*result["bcr"]:.4f}%', flush=True)
