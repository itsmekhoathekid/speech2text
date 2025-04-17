#!/bin/bash
#SBATCH --job-name=rnnt
#SBATCH -o /data/npl/Speech2Text/train_out/rnnt_%j.out
#SBATCH --gres=gpu:1    # Yêu cầu 1 GPU bất kỳ
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --nodes=1


python /data/npl/Speech2Text/RNN-Transducer/train.py