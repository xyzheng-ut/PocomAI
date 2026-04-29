# PocomAI: An AI-Driven Framework for Polymer Composite Analysis from X-ray CT Data

PocomAI is a modular deep learning framework for end-to-end analysis of polymer composites from X-ray CT data. It consists of six sequential deep learning models that transform raw image data into segmented microstructures, reconstructed 3D representations, mesh-based descriptions, and predicted effective material properties. The repository also includes data generation utilities for creating synthetic training datasets.

---

## Framework Overview

```
Raw X-ray CT Images
        │
        ▼
┌─────────────────┐
│  PocomAI-seg    │  2D semantic segmentation of CT slices
│  (grayscale →   │  (matrix / filler binary mask)
│   binary mask)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PocomAI-gen    │  Conditional diffusion model
│  (images →      │  (generates 3D voxel grids)
│   voxel grids)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PocomAI-sup    │  3D super-resolution
│  (coarse →      │  (64³ → 128³ voxel grids)
│   fine grids)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ PocomAI-seg3D   │  3D instance segmentation
│  (voxel grid →  │  (matrix / filler / boundary + instance labels)
│   labeled grid) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PocomAI-ident  │  Filler geometry identification
│  (instances →   │  (type, position, size, orientation → mesh)
│   mesh params)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PocomAI-homo   │  Homogenization property prediction
│  (voxel grid →  │  (φ, Young's modulus E, conductivity k)
│   properties)   │
└─────────────────┘
```

---

## Repository Structure

```
├── deep learning models/
│   ├── 1PocomAI-seg.py        # 2D semantic segmentation
│   ├── 2PocomAI-gen.py        # Conditional 3D diffusion generation
│   ├── 3PocomAI-sup.py        # 3D voxel super-resolution
│   ├── 4PocomAI-seg3D.py      # 3D instance segmentation
│   ├── 5PocomAI-ident.py      # Filler geometry identification
│   └── 6PocomAI-homo.py       # Homogenized property prediction
└── modeling/
    ├── 1generate_filler_information.py      # Synthetic microstructure generation
    ├── 2load_filler_and_generate_mesh.py    # CSV → 3D mesh reconstruction
    └── 3load_filler_and_generate_voxels.py  # CSV → labeled voxel grid
```

---

## Deep Learning Models

### 1. PocomAI-seg — 2D Semantic Segmentation

**File:** `deep learning models/1PocomAI-seg.py`

Performs semantic segmentation of X-ray CT image slices. Takes grayscale CT images as input and classifies each pixel as either matrix or filler, producing binary segmentation masks.

**Architecture:** BASNet (Boundary-Aware Salient Object Detection Network) adapted for composite segmentation.
- **Encoder:** ResNet-34 backbone (via `keras_hub`) with 4 feature pyramid levels
- **Prediction module:** Encoder–bridge–decoder with skip connections and multi-scale side outputs
- **Residual Refinement Module (RRM):** Refines the coarse prediction by learning a residual correction
- **Input:** Grayscale CT images — shape `(H, W, 1)`, default 128×128
- **Output:** Binary probability map (filler probability per pixel)

**Training data:** Synthetic CT slices are generated on-the-fly from binary 3D voxel volumes (`.npz` files). The synthesis pipeline includes: physics-based contrast simulation, Perlin-like bias fields, Poisson photon noise, motion blur, and ring artifacts — to mimic realistic X-ray CT appearance.

**Loss:** Weighted sum of binary cross-entropy, SSIM loss, and IoU loss applied to all decoder outputs simultaneously.

**Key hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Image size | 128 × 128 |
| Batch size | 8 |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 100 |

---

### 2. PocomAI-gen — Conditional 3D Voxel Generation

**File:** `deep learning models/2PocomAI-gen.py`

A conditional diffusion model that generates batches of 3D voxel grids whose microstructures are consistent with one or more provided 2D conditioning images.

**Architecture:** Conditional DDPM (Denoising Diffusion Probabilistic Model) with a 3D U-Net.
- **Noise schedule:** Linear β schedule from 0.0001 to 0.02, T = 200 timesteps
- **U-Net:** 3D convolutional U-Net with ResNet blocks, linear attention (downsampling path), full attention (bottleneck), and sinusoidal timestep embeddings
- **Condition encoder (`SetConditionEncoder`):** Encodes a variable-length set of 2D slice images into a fixed-size context vector using a CNN image encoder + attention-based pooling; supports masked conditioning (missing slices set to zero)
- **Class conditioning:** Context vector is spatially expanded and concatenated at every U-Net level
- **Input:** 64×64×64 binary voxel grids; conditioning images taken as every 4th slice of the volume (16 slices)
- **Output:** 64×64×64 generated voxel grids (continuous, binarized at inference)

**Inference:** Standard DDPM reverse diffusion (200 denoising steps); output is clipped and rounded to binary.

**Key hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Voxel resolution | 64³ |
| Batch size | 4 |
| Timesteps | 200 |
| Condition dim | 256 |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 20 |

---

### 3. PocomAI-sup — 3D Voxel Super-Resolution

**File:** `deep learning models/3PocomAI-sup.py`

Enhances the spatial resolution of coarse voxel grids by transforming 64³ inputs into 128³ outputs while preserving microstructural characteristics such as filler morphology and volume fraction.

**Architecture:** EDSR (Enhanced Deep Super-Resolution) adapted to 3D volumetric data.
- **Feature extraction:** Initial 3D convolution followed by 8 residual blocks, each containing two 3×3×3 convolutions with a skip connection
- **Upsampling:** 3D sub-pixel convolution (`depth_to_space_3d`), expanding spatial dimensions by factor 2 in all three axes
- **Input/Output:** 64×64×64 → 128×128×128 float voxel grids

**Loss:** Mean absolute error (L1). **Metric:** PSNR (peak signal-to-noise ratio).

**Key hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Filters | 64 |
| Residual blocks | 8 |
| Upsampling factor | 2× (each axis) |
| Batch size | 16 |
| Optimizer | Adam with piecewise lr (1e-3 → 1e-4 after 5000 steps) |
| Epochs | 200 |

---

### 4. PocomAI-seg3D — 3D Instance Segmentation

**File:** `deep learning models/4PocomAI-seg3D.py`

Performs 3D instance segmentation on composite voxel grids. It simultaneously classifies voxels into matrix, filler, and boundary regions while distinguishing individual filler instances (up to 100 instances per volume).

**Architecture:** 3D BASNet — a 3D extension of the BASNet architecture.
- **Backbone:** Custom 3D ResNet-34 with 3D convolutions throughout (3×3×3 kernels), max-pooling, and skip connections
- **Prediction module:** 3D encoder–bridge–decoder with `UpSampling3D` and 3D convolution blocks
- **Residual Refinement Module:** 3D RRM refining predictions by learning a residual
- **Input:** 128×128×128 binary voxel grids — shape `(D, H, W, 1)`
- **Output:** 101-channel probability map per voxel (1 matrix class + up to 100 filler instance classes)

**Loss:** Sum of categorical cross-entropy, SSIM loss, and IoU loss across all decoder outputs.

**Key hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Input resolution | 128³ |
| Output classes | 101 |
| Batch size | 8 |
| Optimizer | Adam (lr=1e-4) |
| Epochs | 100 |

---

### 5. PocomAI-ident — Filler Geometry Identification

**File:** `deep learning models/5PocomAI-ident.py`

Converts voxelized filler instances into parametric mesh descriptions by predicting the geometric parameters of each individual filler: type, position, size, and orientation. The predicted parameters can then be used to reconstruct the composite mesh analytically.

**Architecture:** Multi-task 3D CNN.
- **Backbone:** Five Conv3D blocks (16→32→64→128→256 filters, kernel 5×5×5) with MaxPooling3D, followed by GlobalAveragePooling and two Dense layers (128 units)
- **Task heads:**
  - `type`: 4-class softmax (sphere, cube, cylinder, ellipsoid)
  - `center`: 3 regression outputs (x, y, z center coordinates)
  - `size`: 3 regression outputs (s1, s2, s3 — geometry-specific dimensions)
  - `vec`: 3 regression outputs (orientation unit vector)
- **Input:** Individual filler instance voxel grids — shape `(128, 128, 128, 1)`, one sample per filler
- **Output:** CSV file with predicted geometric parameters for all fillers

**Loss:** Sparse categorical cross-entropy for type; MAE for center, size, and orientation.

**Key hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Input resolution | 128³ per filler |
| Filler types | 4 (sphere, cube, cylinder, ellipsoid) |
| Batch size | 16 |
| Optimizer | Adam (lr=2e-4) |
| Epochs | 100 |

---

### 6. PocomAI-homo — Homogenized Property Prediction

**File:** `deep learning models/6PocomAI-homo.py`

Predicts homogenized effective material properties directly from composite voxel grids, serving as a fast surrogate for conventional finite-element or FFT-based homogenization. Predicts three properties simultaneously: solid volume fraction φ, Young's modulus E, and thermal conductivity k.

**Architecture:** Deep 3D CNN regression network.
- **Backbone:** Six Conv3D blocks (16→32→64→128→256→512 filters, kernel 3×3×3) with MaxPooling3D, followed by three Dense layers (512→128→3)
- **Periodic padding:** `pad_dim` utility applies periodic boundary padding before convolutions, respecting the periodic nature of the RVE
- **Input:** 128×128×128 binary voxel grids — shape `(D, H, W, 1)`
- **Output:** Vector of 3 predicted properties [φ, E, k]

**Training:** Targets are normalized before training and denormalized for evaluation. Parity plots (true vs. predicted) are generated every 10 epochs with R² scores reported for each property.

**Loss:** MAE. **Metric:** R² score (computed with `sklearn`).

**Key hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Input resolution | 128³ |
| Output properties | 3 (φ, E, k) |
| Batch size | 32 |
| Optimizer | Adam (lr=1e-4, β₁=0.5) |
| Epochs | 100 |

---

## Data Generation Scripts

### 1. Synthetic Microstructure Generation

**File:** `modeling/1generate_filler_information.py`

Generates synthetic polymer composite microstructures with non-overlapping fillers in a unit RVE [0,1]³. Produces a CSV per composite containing full geometric information for every filler.

**Supported filler geometries:**
| Type | Parameters | `s1` | `s2` | `s3` | `vec` |
|------|-----------|------|------|------|-------|
| Sphere | radius | radius | 0 | 0 | (0,0,0) |
| Cylinder | radius, length, axis | radius | length | 0 | axis (unit) |
| Cube | side, rotation | side | side | side | local +Z axis |
| Ellipsoid | a=b, c, axis | a | a | c | c-axis (unit) |

**Configuration:**
- 15 type-mixture probability presets (pure single-type through uniform 4-type mixture)
- 10,000 composites per preset
- 50–100 fillers per composite
- Non-overlap verified via bounding-sphere broad phase + mesh surface-point penetration test

**Output:** `out_microstructures/csv/composite_XXYYYY.csv` — one CSV per composite.

---

### 2. Mesh Reconstruction from CSV

**File:** `modeling/2load_filler_and_generate_mesh.py`

Reconstructs colored 3D composite meshes from the CSV parameter files and exports them as GLB or PLY for visualization.

**Features:**
- Rebuilds each filler geometry from its stored parameters using `trimesh`
- Assigns type-based colors: sphere (red), cylinder (blue), cube (green), ellipsoid (purple)
- Adds an RVE boundary box (wireframe or solid, configurable)
- Exports to GLB (scene with named nodes) or PLY (merged colored mesh)

**Usage:**
```bash
python 2load_filler_and_generate_mesh.py --input out_microstructures/csv \
    --out_dir reconstructed --fmt glb --box wireframe
```

---

### 3. Voxel Grid Generation from CSV

**File:** `modeling/3load_filler_and_generate_voxels.py`

Voxelizes composite microstructures from CSV parameter files into labeled integer voxel grids and exports them as compressed `.npz` files for use as training data.

**Features:**
- Analytical voxelization: each filler type is rasterized by a closed-form membership test (no mesh sampling)
- Each filler gets a unique integer label (1, 2, 3, …); matrix voxels are 0
- Configurable resolution (default 64³)
- Optional PLY export (voxel cube meshes or point clouds) for visualization

**Usage:**
```bash
python 3load_filler_and_generate_voxels.py --input out_microstructures/csv \
    --out_dir vox_npy64 --res 64 --export_ply --ply_dir vox_ply
```

**Output format (`.npz`):** Each file contains a single array `vol` of shape `(res, res, res)` with `uint16` labels.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| TensorFlow ≥ 2.x | Core deep learning framework |
| Keras | High-level model API |
| keras-hub | Pre-built ResNet backbone |
| NumPy | Array operations |
| SciPy | Image filtering, CT synthesis |
| OpenCV (`cv2`) | Image processing |
| trimesh | 3D mesh creation, export, overlap checks |
| einops | Tensor rearrangement in diffusion model |
| Matplotlib | Visualization and training plots |
| Pandas | CSV handling and training history logging |
| scikit-learn | Data shuffling, R² evaluation |
| seaborn | Joint-plot visualization for property predictions |
| tqdm | Progress bars |

Install with:
```bash
pip install tensorflow keras keras-hub numpy scipy opencv-python trimesh einops \
    matplotlib pandas scikit-learn seaborn tqdm
```

---

## Data Format

All volumetric data is stored as compressed NumPy archives:

```python
# Binary volume (matrix/filler)
data = np.load("sample.npz")
vol = data["vol"]      # shape: (D, H, W), dtype: bool or uint8, values: {0, 1}

# Instance-labeled volume
data = np.load("labeled.npz")
vol = data["vol"]      # shape: (D, H, W), dtype: uint16, 0=matrix, 1..N=filler instances
```

CT training data for PocomAI-seg is synthesized on-the-fly from binary volumes; no separate CT image dataset is required.

---

## Workflow

The six models form a sequential pipeline. A typical end-to-end run proceeds as follows:

1. **Data preparation:** Run `modeling/1generate_filler_information.py` to create synthetic microstructure CSVs, then `modeling/3load_filler_and_generate_voxels.py` to produce voxel grids at the desired resolution.

2. **Segmentation training (`PocomAI-seg`):** Train on binary voxel volumes; CT slices are synthesized automatically. Set `npz_root` to your 256³ voxel directory.

3. **Generation training (`PocomAI-gen`):** Train on 64³ voxel grids. Set `npz_root` to your 64³ voxel directory.

4. **Super-resolution training (`PocomAI-sup`):** Train on paired 64³ (input) and 128³ (target) voxel grids.

5. **3D segmentation training (`PocomAI-seg3D`):** Train on labeled 128³ voxel grids (instance labels produced in step 1).

6. **Identification training (`PocomAI-ident`):** Train on individual filler voxel instances with CSV geometric labels.

7. **Homogenization training (`PocomAI-homo`):** Train on 128³ binary voxel grids paired with tabulated FEM/FFT homogenization results.
