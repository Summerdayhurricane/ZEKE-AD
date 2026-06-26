import os
import json
import argparse
import torch
import torchvision.transforms as transforms
import open_clip

from utils import setup_seed, create_gaussian_kernel_3x3, compute_statistics
from evaluation import evaluate_single_seed


def test(args):
    img_size = args.image_size
    dataset_dir = args.data_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(args.gpu)

    # Initialize model
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)

    # Dataset preprocessing
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    # Get all classes and filter by scene_ids
    meta = json.load(open(os.path.join(dataset_dir, 'meta.json'), 'r'))
    all_classes = sorted(meta['test'].keys())

    if args.scene_ids is not None and len(args.scene_ids) > 0:
        selected_scenes = set(str(s) for s in args.scene_ids)
        all_classes = [cls for cls in all_classes if cls.split('_')[0] in selected_scenes]
        print(f"Selected scene IDs: {args.scene_ids}")

    print(f"Total classes to evaluate: {len(all_classes)}")

    iou_thresholds = [0.2, 0.4, 0.6]
    # Pre-create base Gaussian kernel (avoid repeated creation per layer)
    base_gaussian_kernel = create_gaussian_kernel_3x3(sigma=0.5).view(1, 1, 3, 3)

    # Run once for each seed
    all_seed_results = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running with seed: {seed}")
        print(f"{'='*60}")
        setup_seed(seed)
        seed_result = evaluate_single_seed(
            args, model, preprocess, transform, all_classes, base_gaussian_kernel, device, seed
        )
        seed_result['seed'] = seed
        all_seed_results.append(seed_result)

    # ==================== Summarize results over all seeds ====================
    result_file = os.path.join(save_path, 'results.txt')
    with open(result_file, 'w') as f:
        # Write header information
        header = (
            f"Dataset: {dataset_dir}\n"
            f"Scene IDs: {args.scene_ids if args.scene_ids else 'all'}\n"
            f"Total classes: {len(all_classes)}\n"
            f"Seeds: {args.seeds}\n"
            f"K-shot: {args.k_shot}\n"
            f"Model: {args.model}\n"
            f"Image size: {img_size}\n"
            f"Few-shot features: {args.few_shot_features}\n"
        )
        header += "=" * 60 + "\n"
        f.write(header)
        print(header)

        # Write per-seed results
        f.write("Per-seed Results:\n")
        print("Per-seed Results:")
        for res in all_seed_results:
            seed = res['seed']
            auroc = res['auroc']
            line = f"  Seed {seed}: AUROC={auroc:.4f}, MeanIoU={res['iou']['mean_iou']:.4f}"
            for t in iou_thresholds:
                acc = res['iou']['iou_acc'][t]
                line += f", IoU-Acc@{t}={acc:.4f}"
            f.write(line + "\n")
            print(line)

        # Compute statistics
        f.write("\n" + "=" * 60 + "\n")
        f.write("Statistics over seeds (mean / var / std):\n")
        print("\n" + "=" * 60)
        print("Statistics over seeds (mean / var / std):")

        auroc_values = [res['auroc'] for res in all_seed_results]
        auroc_stats = compute_statistics(auroc_values)
        line = (
            f"  AUROC:      "
            f"mean={auroc_stats['mean']:.4f}, var={auroc_stats['var']:.6f}, std={auroc_stats['std']:.4f}"
        )
        f.write(line + "\n")
        print(line)

        mean_iou_values = [res['iou']['mean_iou'] for res in all_seed_results]
        class_iou_values = [res['iou']['class_mean_iou'] for res in all_seed_results]
        mean_iou_stats = compute_statistics(mean_iou_values)
        class_iou_stats = compute_statistics(class_iou_values)
        line = (
            f"  MeanIoU: mean="
            f"{mean_iou_stats['mean']:.4f} / {mean_iou_stats['var']:.6f} / {mean_iou_stats['std']:.4f},  "
            f"class-mean="
            f"{class_iou_stats['mean']:.4f} / {class_iou_stats['var']:.6f} / {class_iou_stats['std']:.4f}"
        )
        f.write(line + "\n")
        print(line)

        for t in iou_thresholds:
            acc_values = [res['iou']['iou_acc'][t] for res in all_seed_results]
            acc_stats = compute_statistics(acc_values)
            line = (
                f"  IoU-Acc@{t}: mean="
                f"{acc_stats['mean']:.4f} / {acc_stats['var']:.6f} / {acc_stats['std']:.4f}"
            )
            f.write(line + "\n")
            print(line)

    print(f"\nAll results saved to: {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("ZEKE-AD All-in-One Evaluation", add_help=True)
    parser.add_argument("--data_path", type=str, default="/mnt1/qi/hanlinao/ReMP-AD-main/data/data",
                        help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/all_in_one', help='path to save results')
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[6, 12, 18, 24],
                        help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--k_shot", type=int, default=5, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--scene_ids", type=int, nargs="+", default=None,
                        help="scene IDs (1-20) to evaluate. If not set, evaluate all scenes.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="list of random seeds to evaluate. Statistics will be computed over seeds.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id to use")
    args = parser.parse_args()

    test(args)
