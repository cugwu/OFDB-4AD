#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --partition=gpu-l8 # gpu-l8 gpu-low
#SBATCH --account=anom3d
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100-sxm4-80gb:1 # a100_3g.40gb a100-sxm4-80gb v100
#SBATCH --job-name=1p_frac_gray_res50
#SBATCH --output=1p_frac_gray_res50.out
#SBATCH --error=1p_frac_gray_res50.err

module load python/3.10.8-gcc-12.1.0-linux-ubuntu22.04-zen2
module load python/py-pip-22.2.2-gcc-12.1.0-linux-ubuntu22.04-x86_64
source /data/users/cugwu/all_envs/anomalib_env/bin/activate

python main.py --datadir /data/users/cugwu/ad_data/1p-fractals --outdir /data/users/cugwu/ad_data/iccv25/fractals_pretrain \
                               --store_name fractals_gray_resnet50 --arch resnet50 --num_class 1000 --batch_size 128 --epochs 20000

