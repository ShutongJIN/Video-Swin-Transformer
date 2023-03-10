#!/usr/bin/env bash
#SBATCH --mem  5GB
#SBATCH --gres gpu:8
#SBATCH --cpus-per-task 4
#SBATCH --constrain "khazadum|rivendell|belegost|shire"
#SBATCH --mail-type FAIL
#SBATCH --mail-user shutong@kth.se
#SBATCH --output /Midgard/home/shutong/Video_Swin_Transformer/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/shutong/Video_Swin_Transformer/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc
conda activate test_lab
bash tools/dist_train.sh configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py 8 --cfg-options model.backbone.pretrained=/Midgard/home/shutong/Video-Swin-Transformer/pre_trained_models/swin_tiny_patch244_window877_kinetics400_1k.pth 


