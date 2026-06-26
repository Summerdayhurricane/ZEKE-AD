import os
import json
import torch
from PIL import Image


def build_memory_from_test_good(args, model, preprocess, transform, cls_name, device):
    """
    Randomly sample k_shot images from test/good and train of the current class
    as references (memory), and compute their patch-token features.

    Returns
    -------
    mem_features_dict : dict
        {cls_name: [layer_features]}
    ref_entries : list
        List of sampled meta entries.
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
