@echo off
echo Starting C2F-SemiCD TIF Training with 20% Labeled Data...

REM Set CUDA memory management environment variables
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set CUDA_LAUNCH_BLOCKING=1

python train_C2F-SemiCD_tif.py ^
    --paired_csv ./dataset_delhi/paired_dataset.csv ^
    --unpaired_csv ./dataset_delhi/unpaired_pairs.csv ^
    --val_csv ./dataset_delhi/val_dataset.csv ^
    --epoch 100 ^
    --batchsize 4 ^
    --trainsize 128 ^
    --train_ratio 0.2 ^
    --gpu_id 0 ^
    --data_name delhi_20percent ^
    --model_name C2F-SemiCD_20percent ^
    --save_path ./results/

echo Training completed!
pause