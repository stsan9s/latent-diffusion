#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

##########
source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load cmake/3.22.3
module load cuda/11.3/11.3.1
module load cudnn/8.2/8.2.4
module load nccl/2.9/2.9.9-1
module load python/3.8/3.8.13
source /home/acd13649ev/summer2022/latent-diffusion/ldm_env/bin/activate

