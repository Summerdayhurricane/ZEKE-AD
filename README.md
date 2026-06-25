# ZEKE-AD

Official evaluation code for **ZEKE-AD**.

This repository contains the core inference scripts for few-shot anomaly detection and segmentation.

## Files

| File | Description |
|------|-------------|
| `test_all_in_one.py` | Main evaluation script. Computes image-level AUROC and segmentation IoU across multiple random seeds. |
| `dataset.py` | Dataset class for loading test images and memory/reference samples. |
| `run_test_all.sh` | Example bash script to run evaluations for k-shot = 1, 2, 3, 4, 5. |

## Requirements

- Python >= 3.8
- PyTorch
- torchvision
- torchmetrics
- Pillow
- opencv-python
- open_clip

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation

Organize your dataset following the MVTec-AD structure with `meta.json`:

```
data/
├── meta.json
├── train/
│   └── <class_name>/
│       └── good/
├── test/
│   └── <class_name>/
│       ├── good/
│       └── <defect_type>/
└── ground_truth/
    └── <class_name>/
        └── <defect_type>/
```

## Usage

### Single evaluation

```bash
python test_all_in_one.py \
    --data_path /path/to/your/dataset \
    --save_path ./results/all_in_one \
    --model ViT-L-14-336 \
    --pretrained openai \
    --few_shot_features 6 12 18 24 \
    --image_size 518 \
    --k_shot 5 \
    --gpu 0 \
    --seeds 42 1 10 200 500
```

### Run all k-shot experiments

```bash
bash run_test_all.sh
```

> Remember to update `run_test_all.sh` with your actual dataset path and GPU id.


## License

This project is released under the MIT License.
