import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import MRIADDataset
from memory import build_memory_from_test_good
from metrics import BinaryAUROCIoUMetrics


def _extract_patch_tokens(p, model_name):
    """Reshape patch tokens according to backbone type."""
    if 'ViT' in model_name:
        return p[0, 1:, :]
    else:
        return p[0].view(p.shape[1], -1).permute(1, 0).contiguous()


def _compute_layer_anomaly_map(p, mem_features, cls_name, layer_idx,
                               args, base_gaussian_kernel, h, w, device):
    """
    Compute the anomaly map for a single layer using cosine similarity
    between the test image and memory features.
    """
    feature_dim = p.shape[-1]
    num_patches_1d = int(np.sqrt(p.shape[0]))

    # Reshape to 2D patch grid
    p = p.view(num_patches_1d, num_patches_1d, feature_dim)
    p = p.permute(2, 0, 1).unsqueeze(0)

    gaussian_kernel = base_gaussian_kernel.expand(feature_dim, 1, 3, 3).to(device)
    convolved_tensor = F.conv2d(p, gaussian_kernel, padding=1, groups=feature_dim)
    convolved_tensor = convolved_tensor.view(feature_dim, num_patches_1d * num_patches_1d)
    p_G = convolved_tensor.permute(1, 0)

    mem_G = mem_features[cls_name][layer_idx].view(
        args.k_shot, num_patches_1d * num_patches_1d, feature_dim
    ).permute(1, 0, 2)
    mem_G = mem_G.view(num_patches_1d, num_patches_1d, args.k_shot, feature_dim).permute(2, 3, 0, 1)
    convolved_tensor = F.conv2d(mem_G, gaussian_kernel, padding=1, groups=feature_dim)
    convolved_tensor = convolved_tensor.view(
        args.k_shot, feature_dim, num_patches_1d * num_patches_1d
    ).permute(1, 0, 2).reshape(feature_dim, num_patches_1d * num_patches_1d * args.k_shot)
    mem_G0 = convolved_tensor.permute(1, 0)

    # Compute cosine similarity directly on GPU
    cos_9grid = torch.mm(
        F.normalize(mem_G0.float(), dim=1),
        F.normalize(p_G.float(), dim=1).t()
    )
    height = int(np.sqrt(cos_9grid.shape[1]))
    anomaly_map_few_shot_cos_9grid = (1 - cos_9grid).min(dim=0)[0].view(1, 1, height, height)

    # Resize to original image size
    anomaly_map_few_shot_resized = F.interpolate(
        anomaly_map_few_shot_cos_9grid,
        size=(h, w), mode='bilinear', align_corners=True
    )
    return anomaly_map_few_shot_resized[0]


def evaluate_single_seed(args, model, preprocess, transform, all_classes, base_gaussian_kernel, device, seed):
    """
    Run all specified scenes under a fixed seed and return the metrics for that seed.
    """
    dataset_dir = args.data_path
    save_path = args.save_path
    img_size = args.image_size
    few_shot_features = args.few_shot_features

    metrics = BinaryAUROCIoUMetrics(device)

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
        # to avoid testing on reference images themselves; remaining normal images (including train)
        # continue to be used for testing.
        ref_img_paths = set(e['img_path'] for e in ref_entries)
        test_data.data_all = [d for d in test_data.data_all if d['img_path'] not in ref_img_paths]
        test_data.length = len(test_data.data_all)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        for items in test_dataloader:
            image = items['img'].to(device)
            cls_name_val = items['cls_name'][0]
            path = items['img_path']

            # Determine label: parse specie_name (good or defect) from the path
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
                    p = _extract_patch_tokens(p, args.model)
                    anomaly_map_layer = _compute_layer_anomaly_map(
                        p, mem_features, cls_name_val, idx,
                        args, base_gaussian_kernel, h, w, device
                    )
                    anomaly_maps_few_shot_gpu.append(anomaly_map_layer)

                # Average multi-layer anomaly maps on GPU and transfer to CPU only once
                anomaly_map_few_shot = torch.stack(anomaly_maps_few_shot_gpu).mean(dim=0)  # [1, h, w] on GPU
                anomaly_map_gpu = anomaly_map_few_shot[0]  # [h, w] on GPU

                # ---- Image-level AUROC ----
                # Use topk on GPU to get top-20, avoiding CPU np.sort
                anomaly_map_20 = anomaly_map_gpu.flatten().topk(20)[0].mean()
                metrics.update_auroc(
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
                        metrics.add_iou(anomaly_map, gt_mask, cls_name_val)

        print(f"  Finished class: {cls_name}")

    seed_result = metrics.compute()
    return seed_result
