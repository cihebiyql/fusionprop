#!/bin/bash
#SBATCH --gpus=1
#SBATCH -x g[0014-0016,0023,0025-0032,0034-0035,0039,0042-0043,0045,0047,0053,0060,0065,0159,0162,0164,0166,0170,0171,0172]
module purge
module load anaconda/2020.11 gcc/9.3
module load cuda/12.1
module load cudnn/8.9.7_cuda12.x
source activate dev240430
# export PATH=/HOME/scz0brz/run/anaconda3/bin:$PATH
export PYTHONUNBUFFERED=1
which python
# ldd /data/apps/cudnn/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn_cnn_train.so.8
# strings /data/apps/cudnn/cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn_cnn_infer.so |grep libcudnn_cnn_infer.so.8
env

python train_22_1_1.py \
    --train_csv s2c2_1_train.csv \
    --test_csv s2c2_1_test.csv \
    --target_column tgt_reg \
    --sequence_column sequence \
    --batch_size 16 \
    --epochs 30 \
    --lr 3e-4 \
    --weight_decay 3e-5 \
    --max_seq_len 600 \
    --hidden_dim 256 \
    --dropout 0.3 \
    --num_folds 5 \
    --patience 10 \
    --seed 3407 \
    --train_mode fusion \
    --fusion_type weighted \
    --use_esmc \
    --use_esm2 \
    --model_save_dir ./train_22_1_1_1 \
    --experiment_name thermostability_prediction \
    --feature_gpu 0 \
    --train_gpu 0 \
    --use_amp \
    --normalize_features \
    --feature_cache_size 3000 \
    --num_workers 6 \
    --normalize_method none \
    > train_22_1_1_1_2.log &

wait
echo "所有脚本运行完成。"


# 模型 4 (Trial 78):
#   Test Pearson: 0.966326 ± 0.004644
#   Test Spearman: 0.950378
#   隐藏层维度: 256, Dropout: 0.30000000000000004
#   批次大小: 16
#   学习率: 0.000317, 权重衰减: 0.00003003
#   标准化方法: none
