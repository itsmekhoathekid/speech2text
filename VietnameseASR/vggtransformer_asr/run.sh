#!/bin/bash
#SBATCH --job-name=vgg_trans
#SBATCH -o /data/npl/Speech2Text/train_out/vgg_%j.out  # Tạo file log với ID job để dễ theo dõi
#SBATCH --gres=gpu:1
#SBATCH -N 1 # số lượng node để chạy
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G



python /data/npl/Speech2Text/VietnameseASR-main/vggtransformer_asr/vggtransformer_train.py /data/npl/Speech2Text/VietnameseASR-main/vggtransformer_asr/hparams/vggtransformer.yaml