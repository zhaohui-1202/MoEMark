# MoEMark

**Robust zero-watermarking for vector geographic data through asymmetric registration and attack-guided expert detection.**

MoEMark establishes copyright evidence for point, polyline, and polygon layers without modifying their geometry. A single-branch registration encoder defines a reference identity, while a mixture-of-experts detector recovers that identity from geometrically modified maps. This repository provides the core models, registration and verification interface, robustness evaluation, and map and copyright-image discriminability analysis.

## Method overview

### Asymmetric registration and detection

A shared map representation combines local geometry, multiscale spatial context, and layer-level statistics. During registration, a single-branch encoder maps this representation to a reference identity. A shared feature mapping conditions the identity on a copyright image to construct the registered zero-watermark.

During verification, the detector adapts to the geometry of the candidate map and recovers an identity compatible with the registration reference. The same registered zero-watermark supports verification under different attack conditions.

![Overview of the asymmetric registration and detection framework in MoEMark.](docs/assets/moemark-overview.png)

### Attack-guided expert learning

The detector combines an always-active general expert with three specialists. The specialists learn recovery functions for vertex edits, feature deletion, and other geometric disturbances. A geometry-dependent router selects two specialists and combines their corrections with the general expert through weighted residual aggregation. Verification uses the candidate geometry without requiring attack labels.

Cross-stage alignment guides both candidate map identities and copyright-conditioned feature vectors toward their fixed registration targets. The registration branch and shared feature mapping remain frozen during detector training.

![Attack-guided expert training and cross-stage identity alignment.](docs/assets/expert-learning-alignment.png)

### Identity binding and zero-watermark verification

The copyright-conditioned feature vector is binarised and combined with a binary copyright payload and keyed binding codes by exclusive OR (XOR). Separate codes bind the registered zero-watermark to the copyright image and geometry category.

Verification derives the candidate binary feature sequence, recovers the watermark using the stored registration record, and compares it with the payload of the claimed copyright image. Normalised correlation (NC) measures the similarity between the recovered and reference binary watermarks; bit-correct rate (BCR) measures the proportion of correctly recovered bits.

## Code structure

| Component | Main files and purpose |
| --- | --- |
| Entry point | `src/run_evaluation.py`: evaluation, registration, verification, and plotting commands |
| Models | `src/core/models.py`: single-branch registration encoder and mixture-of-experts detector |
| Inference | `src/core/pipeline.py`: checkpoint loading, feature extraction, registration, and verification |
| Identity binding | `src/core/image_binding.py` and `map_geometry_binding.py`: copyright-image and geometry-category codes |
| Shared representation and attacks | `src/core/shared/`: geometry preprocessing, frozen encoders, feature mapping, and geometric attacks |
| Evaluation | `src/core/evaluate.py` and `src/protocols/single92.json`: robustness and discriminability evaluation |
| Visualisation | `src/plot_results.py`: NC/BCR curves and discriminability heatmaps |
| Method diagrams | `docs/assets/`: framework overview and expert-learning illustration |

## Data and checkpoints

The evaluation protocol uses six vector layers derived from [OpenStreetMap](https://www.openstreetmap.org/copyright), 20 copyright images, and three detector seeds (42, 73, and 101). Each evaluation map is paired with every copyright image. Robustness is evaluated across 92 attack settings covering coordinate perturbations, geometric transformations, structural edits, and changes in data representation.

Organise the dataset manifest, Shapefiles, copyright images, trained checkpoints, feature caches, and binding key under a common asset directory. Specify this directory with `--root`; asset paths are defined in `src/core/pipeline.py`. Keep all component files of each Shapefile together.

## Environment

Use Python 3.11 on Linux. The implementation uses PyTorch, torchvision, NumPy, SciPy, GeoPandas, Shapely, and Matplotlib. Clone the repository and install the dependencies:

```bash
git clone https://github.com/zhaohui-1202/MoEMark.git
cd MoEMark
python -m pip install -r src/requirements.txt
```

See [requirements.txt](src/requirements.txt) for dependency versions.

## Basic usage

Register a vector layer with a copyright image:

```bash
python src/run_evaluation.py \
  --stage register \
  --root /path/to/moemark_assets \
  --seed 42 \
  --map maps/original.shp \
  --image images/logo.png \
  --record outputs/registration.json
```

Verify a candidate using the corresponding model and registration record:

```bash
python src/run_evaluation.py \
  --stage verify \
  --root /path/to/moemark_assets \
  --seed 42 \
  --map maps/candidate.shp \
  --image images/logo.png \
  --record outputs/registration.json
```

The interface reports NC, BCR, the number of matching bits, and the acceptance decision. The copyright payload is obtained by converting the image to grayscale, resizing it to 16 × 16 with nearest-neighbour interpolation, and applying an intensity threshold of 128.

## Evaluation

Run robustness and discriminability evaluation together:

```bash
python src/run_evaluation.py \
  --stage all \
  --root /path/to/moemark_assets \
  --output outputs/evaluation \
  --workers 6
```

Use `--stage robustness` or `--stage discriminability` to run either evaluation separately.

- **Robustness:** evaluates watermark recovery across 92 attack settings and reports NC and BCR. Results are averaged within each map and then across applicable maps and attack settings.
- **Map discriminability:** constructs a 6 × 6 registered-map versus candidate-map matrix, averaged over the copyright images and detector seeds.
- **Copyright-image discriminability:** constructs a 20 × 20 registered-image versus candidate-image matrix, averaged over the evaluation maps and detector seeds.

| Output | Description |
| --- | --- |
| `robustness.json` and `robustness.csv` | Per-setting NC, BCR, and evaluation counts |
| `records/` | Individual attack records and per-seed scores |
| `discriminability.json` | Map and copyright-image discriminability matrices |
| `figures/robustness_nc_bcr.png` | NC and BCR across attack strengths |
| `figures/map_discriminability.png` | Map discriminability heatmap |
| `figures/copyright_image_discriminability.png` | Copyright-image discriminability heatmap |

Regenerate figures from a completed evaluation:

```bash
python src/run_evaluation.py --stage plots --output outputs/evaluation
```

For the complete command-line options:

```bash
python src/run_evaluation.py --help
```

## License

This project is licensed under the [MIT License](LICENSE).
