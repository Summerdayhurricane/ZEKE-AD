import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import json
import cv2
import open_clip
from dataset import MRIADDataset
from torchmetrics import AUROC


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, threshold=0.35):
    if pred.max() <= 0.45:
        pred = np.where(pred < threshold, pred, (pred - pred.min()) / (0.5 - pred.min()))
    else:
        pred = np.where(pred < threshold, pred, (pred - pred.min()) / (pred.max() - pred.min()))
    return pred


def create_gaussian_kernel_3x3(sigma=1.0):
    x = torch.arange(-1, 2, dtype=torch.float32)
    y = torch.arange(-1, 2, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    gaussian_kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    gaussian_kernel /= gaussian_kernel.sum()
    return gaussian_kernel.to(dtype=torch.float16)


def compute_mask_iou(pred_mask, gt_mask):
    # Strictly align with the IoU computation in test_gaussian_auroc_fast.py calculate_iou_with_mask:
    # IoU = intersection_area / union_area
    intersection = cv2.bitwise_and(pred_mask, gt_mask)
    union = cv2.bitwise_or(pred_mask, gt_mask)
    intersection_area = np.sum(intersection > 0)
    union_area = np.sum(union > 0)
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


def build_memory_from_test_good(args, model, preprocess, transform, cls_name, device):
    """
    Randomly sample k_shot images from test/good of the current class as references (memory),
    and compute their patch-token features.
    Returns: mem_features_dict {cls_name: [layer_features]}, ref_entries (list of sampled entries)
    """
    dataset_dir = args.data_path
    k_shot = args.k_shot
    few_shot_features = args.few_shot_features

    meta = json.load(open(os.path.join(dataset_dir, 'meta.json'), 'r'))
    # Sample k_shot reference images from the union of test/good and train (both normal images)
    test_good_entries = [e for e in meta['test'][cls_name] if e['specie_name'] == 'good']
    train_entries = meta['train'][cls_name]
    normal_entries = test_good_entries + train_entries

    if len(normal_entries) < k_shot:
        raise ValueError(
            f"Class {cls_name} has only {len(normal_entries)} normal images "
            f"(test/good: {len(test_good_entries)}, train: {len(train_entries)}), "
            f"but k_shot={k_shot}"
        )

    # setup_seed has been called before, so torch.randperm is reproducible under the same seed
    indices = torch.randperm(len(normal_entries))[:k_shot]
    ref_entries = [normal_entries[i] for i in indices]

    features = []
    for entry in ref_entries:
        img_path = os.path.join(dataset_dir, entry['img_path'])
        img = Image.open(img_path).convert('RGB')
        img = preprocess(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, patch_tokens = model.encode_image(img, few_shot_features)
            if 'ViT' in args.model:
                patch_tokens = [p[0, 1:, :] for p in patch_tokens]
            else:
                patch_tokens = [p[0].view(p.shape[1], -1).permute(1, 0).contiguous() for p in patch_tokens]
            features.append(patch_tokens)

    mem_features = [torch.cat([features[j][i] for j in range(len(features))], dim=0)
                    for i in range(len(features[0]))]

    return {cls_name: mem_features}, ref_entries


def evaluate_single_seed(args, model, preprocess, transform, all_classes, base_gaussian_kernel, device, seed):
    """
    Run all specified scenes under a fixed seed and return the metrics for that seed.
    """
    dataset_dir = args.data_path
    save_path = args.save_path
    img_size = args.image_size
    few_shot_features = args.few_shot_features
    iou_thresholds = [0.2, 0.4, 0.6]
    binary_threshold = 0.5

    auroc_metric = AUROC(task="binary").to(device)
    iou_results = {}  # {cls_name: [iou_values]}

    for cls_name in all_classes:
        test_data = MRIADDataset(
            root=dataset_dir,
            transform=preprocess,
            target_transform=transform,
            aug_rate=-1,
            mode='test',
            obj_name=cls_name
        )
        # Randomly sample k_shot reference images from test/good and generate memory features
        mem_features, ref_entries = build_memory_from_test_good(
            args, model, preprocess, transform, cls_name, device
        )

        # Record the paths of reference images sampled for each class under the current seed
        ref_save_path = os.path.join(save_path, f'ref_samples_seed{seed}.txt')
        os.makedirs(save_path, exist_ok=True)
        with open(ref_save_path, 'a') as f:
            for entry in ref_entries:
                f.write(entry['img_path'] + '\n')

        # Remove images sampled as references from test_data (whether from test/good or train),
        # to avoid testing on reference images themselves; remaining normal images (including train) continue to be used for testing.
        ref_img_paths = set(e['img_path'] for e in ref_entries)
        test_data.data_all = [d for d in test_data.data_all if d['img_path'] not in ref_img_paths]
        test_data.length = len(test_data.data_all)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        for items in test_dataloader:
            image = items['img'].to(device)
            cls_name_val = items['cls_name'][0]
            path = items['img_path']

            # Determine label: parse specie_name (good or 1) from the path
            path_parts = path[0].split('/')
            try:
                cls_idx = path_parts.index(cls_name_val)
                specie_name = path_parts[cls_idx + 2]  # e.g., test/good/xxx.png -> good
            except ValueError:
                specie_name = path_parts[-2]

            if specie_name == 'good':
                label_gt = 0
                has_mask = False
            else:
                label_gt = 1
                has_mask = True

            # Get original image size
            img_pil = Image.open(path[0])
            w, h = img_pil.size

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features, patch_tokens = model.encode_image(image, few_shot_features)
                # Accumulate anomaly maps on GPU to reduce CPU-GPU transfers
                anomaly_maps_few_shot_gpu = []

                for idx, p in enumerate(patch_tokens):
                    if 'ViT' in args.model:
                        p = p[0, 1:, :]
                    else:
                        p = p[0].view(p.shape[1], -1).permute(1, 0).contiguous()

                    feature_dim = p.shape[-1]
                    num_patches_1d = int(np.sqrt(p.shape[0]))

                    # Reshape to 2D patch grid
                    p = p.view(num_patches_1d, num_patches_1d, feature_dim)
                    p = p.permute(2, 0, 1).unsqueeze(0)

                    gaussian_kernel = base_gaussian_kernel.expand(feature_dim, 1, 3, 3).to(device)
                    convolved_tensor = F.conv2d(p, gaussian_kernel, padding=1, groups=feature_dim)
                    convolved_tensor = convolved_tensor.view(feature_dim, num_patches_1d * num_patches_1d)
                    p_G = convolved_tensor.permute(1, 0)

                    mem_G = mem_features[cls_name_val][idx].view(
                        args.k_shot, num_patches_1d * num_patches_1d, feature_dim
                    ).permute(1, 0, 2)
                    mem_G = mem_G.view(num_patches_1d, num_patches_1d, args.k_shot, feature_dim).permute(2, 3, 0, 1)
                    convolved_tensor = F.conv2d(mem_G, gaussian_kernel, padding=1, groups=feature_dim)
                    convolved_tensor = convolved_tensor.view(
                        args.k_shot, feature_dim, num_patches_1d * num_patches_1d
                    ).permute(1, 0, 2).reshape(feature_dim, num_patches_1d * num_patches_1d * args.k_shot)
                    mem_G0 = convolved_tensor.permute(1, 0)

                    # Compute cosine similarity directly on GPU to avoid .cpu() transfer
                    cos_9grid = torch.mm(
                        F.normalize(mem_G0.float(), dim=1),
                        F.normalize(p_G.float(), dim=1).t()
                    )
                    height = int(np.sqrt(cos_9grid.shape[1]))
                    anomaly_map_few_shot_cos_9grid = (1 - cos_9grid).min(dim=0)[0].view(1, 1, height, height)

                    # F.interpolate operates directly on GPU tensor
                    anomaly_map_few_shot_resized = F.interpolate(
                        anomaly_map_few_shot_cos_9grid,
                        size=(h, w), mode='bilinear', align_corners=True
                    )
                    anomaly_maps_few_shot_gpu.append(anomaly_map_few_shot_resized[0])

                # Average multi-layer anomaly maps on GPU and transfer to CPU only once
                anomaly_map_few_shot = torch.stack(anomaly_maps_few_shot_gpu).mean(dim=0)  # [1, h, w] on GPU
                anomaly_map_gpu = anomaly_map_few_shot[0]  # [h, w] on GPU

                # ---- Image-level AUROC ----
                # Use topk on GPU to get top-20, avoiding CPU np.sort
                anomaly_map_20 = anomaly_map_gpu.flatten().topk(20)[0].mean()
                auroc_metric.update(
                    anomaly_map_20.unsqueeze(0),
                    torch.tensor([label_gt], dtype=torch.long, device=device)
                )

                # Move to CPU for subsequent IoU computation (only when needed)
                anomaly_map = anomaly_map_gpu.cpu().numpy()  # (h, w)

                # ---- Segmentation SIoU (only anomalous samples have mask) ----
                if has_mask:
                    mask_path = path[0].replace('/test/', '/ground_truth/')
                    if os.path.exists(mask_path):
                        gt_mask = Image.open(mask_path).convert('L')
                        gt_mask = np.array(gt_mask)

                        # Use the original normalize function to process the anomaly map
                        anomaly_map_norm = normalize(anomaly_map)

                        # Refer to eval_segmentation: binarize prediction map
                        if anomaly_map_norm.max() <= 1.0:
                            binary_pred = (anomaly_map_norm > binary_threshold).astype(np.uint8)
                        else:
                            binary_pred = (anomaly_map_norm > binary_threshold * 255).astype(np.uint8)

                        # Ground truth binarization
                        if gt_mask.max() > 1:
                            gt_binary = (gt_mask > 127).astype(np.uint8)
                        else:
                            gt_binary = (gt_mask > 0.5).astype(np.uint8)

                        iou = compute_mask_iou(binary_pred, gt_binary)
                        if cls_name_val not in iou_results:
                            iou_results[cls_name_val] = []
                        iou_results[cls_name_val].append(iou)

        print(f"  Finished class: {cls_name}")

    # Organize metrics for this seed
    total_auroc = auroc_metric.compute().item()
    seed_result = {
        'auroc': total_auroc,
        'iou': {}
    }

    all_ious = []
    per_class_mean = {}
    for cls_name_val, iou_list in iou_results.items():
        if len(iou_list) > 0:
            cls_mean = np.mean(iou_list)
            per_class_mean[cls_name_val] = cls_mean
            all_ious.extend(iou_list)

    if len(all_ious) > 0:
        sample_mean_iou = np.mean(all_ious)
        class_mean_iou = np.mean(list(per_class_mean.values()))
        eval_classes = len(per_class_mean)
        total_anomaly_samples = len(all_ious)
    else:
        sample_mean_iou = 0.0
        class_mean_iou = 0.0
        eval_classes = 0
        total_anomaly_samples = 0

    # Refer to eval_segmentation: count the proportion of samples whose IoU exceeds each threshold
    iou_gt_counts = {th: 0 for th in iou_thresholds}
    for iou in all_ious:
        for th in iou_thresholds:
            if iou > th:
                iou_gt_counts[th] += 1

    iou_acc = {th: iou_gt_counts[th] / total_anomaly_samples if total_anomaly_samples > 0 else 0.0
               for th in iou_thresholds}

    seed_result['iou'] = {
        'mean_iou': sample_mean_iou,
        'class_mean_iou': class_mean_iou,
        'eval_classes': eval_classes,
        'total_anomaly_samples': total_anomaly_samples,
        'iou_acc': iou_acc
    }

    return seed_result


def compute_statistics(values):
    """Compute mean, variance, and standard deviation"""
    arr = np.array(values, dtype=np.float64)
    return {
        'mean': float(np.mean(arr)),
        'var': float(np.var(arr)),
        'std': float(np.std(arr))
    }


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
    parser.add_argument("--data_path", type=str, default="/path-to-your-dataset",
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
