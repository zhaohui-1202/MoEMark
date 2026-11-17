import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TITLES = {
    'gaussian_noise': 'Gaussian noise', 'uniform_noise': 'Uniform noise', 'outlier_noise': 'Outlier noise',
    'feature_jitter': 'Feature shift', 'rotate': 'Rotation', 'scale': 'Uniform scaling',
    'vertex_add': 'Vertex insertion', 'vertex_delete': 'Vertex deletion', 'delete': 'Feature deletion',
    'simplify': 'Simplification', 'quantize': 'Quantization', 'quantize_valid': 'Constrained quantization',
}
MAP_LABELS = {'buildings': 'Buildings', 'forest': 'Forest', 'poi': 'POI', 'railways': 'Railways', 'roads': 'Roads', 'water': 'Water'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def strength(op):
    if 'sigma_ratio' in op:
        return op['sigma_ratio'] * 100, r'Noise / offset scale (% $\ell$)'
    if 'fraction' in op:
        return op['fraction'] * 100, 'Edited (%)'
    if 'degrees' in op:
        return op['degrees'], 'Angle (deg)'
    if 'ratio' in op:
        return op['ratio'], r'Distance / $R$'
    if 'factor' in op:
        return op['factor'], 'Scale factor'
    if 'tolerance_ratio' in op:
        return op['tolerance_ratio'] * 100, r'Tolerance (% $\ell$)'
    if 'step_ratio' in op:
        return op['step_ratio'] * 100, r'Step (% $\ell$)'
    return None, None


def plot_curves(cases, path):
    groups = {}
    for case in cases:
        op = case['op']
        x, unit = strength(op)
        if x is None:
            continue
        key = (op['name'], op.get('axis'))
        title = TITLES.get(op['name'], f'{op.get("axis", "").capitalize()} {"translation" if op["name"] == "translate" else op["name"]}')
        groups.setdefault(key, dict(title=title, unit=unit, values=[]))['values'].append((x, case['nc'], case['bcr']))
    if not groups:
        return
    nrows = math.ceil(len(groups) / 4)
    fig, axes = plt.subplots(nrows, 4, figsize=(13, 2.7 * nrows), squeeze=False)
    handles = None
    for i, group in enumerate(groups.values()):
        ax = axes.flat[i]
        values = np.asarray(sorted(group['values']))
        line1, = ax.plot(values[:, 0], values[:, 1], '-o', color='#2176c7', label='NC', ms=4, lw=1.5)
        line2, = ax.plot(values[:, 0], values[:, 2], '-s', color='#e45b30', label='BCR', ms=4, lw=1.5)
        handles = [line1, line2]
        ax.set_ylim(.888, 1.007)
        ax.set_yticks([.90, .92, .94, .96, .98, 1.])
        ax.grid(axis='y', color='#dddddd', lw=.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(group['unit'])
        if i % 4 == 0:
            ax.set_ylabel('Mean score')
        ax.text(.5, -.40, f'({chr(97 + i)}) {group["title"]}', transform=ax.transAxes, ha='center', va='top')
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)
    fig.subplots_adjust(top=.95, bottom=.06, hspace=.85, wspace=.35)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_matrix(matrix, labels, path, kind):
    a = np.asarray(matrix, dtype=float)
    assert a.shape == (len(labels), len(labels)) and np.isfinite(a).all()
    fig, ax = plt.subplots(figsize=(12, 11) if len(labels) == 20 else (8, 7.5))
    im = ax.imshow(a, vmin=0, vmax=1, cmap='YlGnBu', interpolation='nearest')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90 if len(labels) == 20 else 45, ha='center' if len(labels) == 20 else 'right')
    ax.set_yticklabels(labels)
    for i in range(len(a)):
        for j in range(len(a)):
            text = f'{a[i,j]:.2f}' if len(a) == 20 else f'{a[i,j]:.3f}'
            ax.text(j, i, text, ha='center', va='center', fontsize=7 if len(a) == 20 else 11, color='white' if a[i,j] > .70 else '#15232e')
    ax.set_xlabel(f'Candidate {kind}')
    ax.set_ylabel(f'Registered {kind}')
    fig.colorbar(im, ax=ax, label='NC', fraction=.045)
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


def plot_results(output, protocol, published=None):
    output = Path(output)
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    if published:
        published = Path(published)
        tables = []
        expected_ids = {c['id'] for c in protocol['cases']}
        for path in [published / 'robustness.csv', *sorted(published.glob('01_*/*.csv'))]:
            if not path.is_file():
                continue
            with path.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.DictReader(stream)
                if not {'case_id', 'nc', 'bcr'} <= set(reader.fieldnames or []):
                    continue
                records = list(reader)
            if len(records) == len(expected_ids) and {int(r['case_id']) for r in records} == expected_ids:
                tables.append({int(r['case_id']): r for r in records})
        if len(tables) != 1:
            raise ValueError('Expected exactly one robustness CSV containing all 92 setting IDs')
        rows = tables[0]
        cases = [dict(c, nc=float(rows[c['id']]['nc']), bcr=float(rows[c['id']]['bcr'])) for c in protocol['cases']]
        matrix_sets = []
        for path in [published / 'discriminability.json', *sorted(published.glob('02_*/*.json'))]:
            if path.is_file():
                data = read(path)
                if isinstance(data, dict) and {'map', 'image'} <= data.keys():
                    matrix_sets.append(data)
        if len(matrix_sets) != 1:
            raise ValueError('Expected exactly one JSON file containing map and image matrices')
        matrices = matrix_sets[0]
    else:
        source = output / 'robustness.json'
        if not source.exists():
            source = output / 'robustness_subset.json'
        cases = read(source)['cases'] if source.exists() else []
        source = output / 'discriminability.json'
        matrices = read(source) if source.exists() else None
    if cases:
        plot_curves(cases, figures / 'robustness_nc_bcr.png')
    if matrices:
        maps, images = matrices['map'], matrices['image']
        labels = [MAP_LABELS.get(n.split('_')[1], n) for n in maps['files']]
        plot_matrix(maps['nc'], labels, figures / 'map_discriminability.png', 'map')
        plot_matrix(images['nc'], [Path(n).stem for n in images['images']], figures / 'copyright_image_discriminability.png', 'image')
    if not cases and not matrices:
        raise FileNotFoundError('No completed robustness or discriminability results in the selected directory')
    print('PNG figures:', figures)
