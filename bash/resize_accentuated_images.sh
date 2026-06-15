#!/bin/bash
#SBATCH -p kempner
#SBATCH --mem 32gb
#SBATCH -c 16
#SBATCH -t 4:00:00
#SBATCH -A kempner_binxuwang_lab
#SBATCH -o /n/home12/binxuwang/Github/Closed-loop-visual-insilico/bash/logs/resize_acc_%j.out
#SBATCH -e /n/home12/binxuwang/Github/Closed-loop-visual-insilico/bash/logs/resize_acc_%j.err
#SBATCH -J resize_acc_imgs

mkdir -p /n/home12/binxuwang/Github/Closed-loop-visual-insilico/bash/logs

source ~/.bashrc
mamba activate torch2

cd /n/home12/binxuwang/Github/Closed-loop-visual-insilico
python bash/resize_accentuated_images.py
