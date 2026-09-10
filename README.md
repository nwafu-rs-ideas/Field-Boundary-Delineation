# SAM-Based Agricultural Field Boundary Delineation with Adaptive Region–Boundary Prompting and Dual-Branch Adapter Fine-Tuning
﻿This repository contains the code for the paper **"SAM-Based Agricultural Field Boundary Delineation with Adaptive Region–Boundary Prompting and Dual-Branch Adapter Fine-Tuning"**. The code provides the training, validation, and testing pipeline on the **AI4Boundaries (AI4B)** dataset.
﻿
## Requirements
* Python 3.8+
* PyTorch 2.0+ with CUDA
* xarray
* rasterio
* scikit-image
* opencv-python
* pandas
* tqdm
* matplotlib
* scikit-learn
﻿
The modified SAM implementation is included in `segment-anything-main/`. Install it in editable mode:
```text
cd segment-anything-main
pip install -e .
```
﻿The code has been tested in the following environment:
﻿* OS: Windows 10
* GPU: NVIDIA RTX A4000 (16 GB)
* Python: 3.10
* PyTorch: 2.5.1
* CUDA: 11.8

## Data Preparation
﻿The AI4Boundaries (AI4B) dataset files used for model training and evaluation are not included in this repository due to their large size.
﻿The original **AI4Boundaries (AI4B)** dataset can be downloaded from the [JRC Open Data Catalogue](https://data.jrc.ec.europa.eu/dataset/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9).
The downloaded AI4B data should be placed in a data directory with the following structure:
```text
AI4BOUNDARIES/
└── sentinel2/
├── ai4boundaries_ftp_urls_sentinel2_split.csv
├── copyright.txt
├── images/
├── masks/
├── masks.zip
├── test.csv
├── test.zip
├── train.csv
├── train.zip
├── val.csv
└── val.zip
```
﻿The `images/`and `masks/` directories contain the original Sentinel-2 imagery and corresponding annotations. The `train.csv`, `val.csv`, and `test.csv` files provide the corresponding data splits.
﻿
## Pretrained Models
### SAM ViT-H
The experiments use the **SAM ViT-H** model.
﻿Download the pretrained checkpoint `sam_vit_h_4b8939.pth` from the official [Segment Anything repository](https://github.com/facebookresearch/segment-anything) and specify its path in `SAM_CKPT`.
﻿For example:
```text
SAM_CKPT = r"path/to/sam_vit_h_4b8939.pth"
```
The SAM checkpoint is not included in this repository.
﻿
### Prompt Network
A pretrained prompt network is required to generate the region and boundary prompts used by the proposed framework. The resulting checkpoint should be specified in `PROMPTER_CKPT`.
﻿
## Usage
The main script is:
```text
run_sam_adapter_ai4b_patch_encoder.py
```
Before running the script, modify the configuration section according to the local environment.
The main paths to be configured include:
```text
─ SAM_CKPT
─ PROMPTER_CKPT
─ TRAIN_NC
─ TRAIN_REGION
─ TRAIN_BOUNDARY
─ VAL_NC
─ VAL_REGION
─ VAL_BOUNDARY
─ TEST_NC_DIR
─ TEST_MASK_DIR
─ TEST_BOUNDARY
─ OUT_DIR
─ CKPT_DIR
```
Then run:
```text
python run_sam_adapter_ai4b_patch_encoder.py
```
The script performs model training and validation and provides the test evaluation pipeline.

## Code Structure
```text
.
├── README.md
├── LICENSE
├── run_sam_adapter_ai4b_patch_encoder.py
└── segment-anything-main/
    ├── segment_anything/
    ├── segment_anything1/
    ├── scripts/
    ├── demo/
    ├── assets/
    ├── notebooks/
    ├── setup.py
    └── setup.cfg
```
─ run_sam_adapter_ai4b_patch_encoder.py: Main training and evaluation script. It defines the dataset loading, model construction, training loop, validation, and testing procedures.
─ segment-anything-main/: Modified SAM implementation used in this study, including the Adapter modules and dual-branch mask decoder.

## Citation
If you find this code useful in your research, please cite our paper:
```text
[Paper citation will be added after publication.]
```
## License
This project is released under the **MIT License**. See the [LICENSE] file for details.
