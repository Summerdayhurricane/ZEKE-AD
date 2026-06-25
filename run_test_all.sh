#!/bin/bash

# ============================================
# Parameters：
# ============================================
# --scene_ids 1 2 5      # only test scene 1、2、5
# --seeds 42 1 10 200 500  
# ============================================

for k in 1 2 3 4 5; do
    echo "============================================"
    echo "Running with k_shot = ${k}"
    echo "============================================"
    python test_all_in_one.py \
        --data_path /path-to-MRI-AD \
        --save_path "./results/all_in_one_k${k}" \
        --model ViT-L-14-336 \
        --pretrained openai \
        --few_shot_features 6 12 18 24 \
        --image_size 518 \
        --k_shot "${k}" \
        --gpu 4 \
        --seeds 42 1 10 200 500
done

echo "All k_shot experiments finished."
echo "Results are saved in:"
for k in 1 2 3 4 5; do
    echo "  ./results/all_in_one_k${k}/results.txt"
done
