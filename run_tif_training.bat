@echo off
echo Training C2F-SemiCD model on TIF images...

python train_C2F-SemiCD_tif.py ^
  --paired_csv .\dataset_delhi\paired_dataset.csv ^
  --unpaired_csv .\dataset_delhi\unpaired_pairs.csv ^
  --val_csv .\dataset_delhi\val_dataset.csv ^
  --epoch 100 ^
  --batchsize 16 ^
  --trainsize 256 ^
  --train_ratio 0.05 ^
  --gpu_id 0 ^
  --data_name TIF_Delhi ^
  --model_name SemiModel_TIF ^
  --save_path .\output\C2F-SemiCD\TIF\

echo Training complete!