#!/bin/bash

#$ -l rt_AF=1
#$ -l h_rt=20:00:00
#$ -j y
#$ -cwd

############################## DATASET ###########################
set -eu

[ ! -d $SGE_LOCALDIR ] && (echo "You must run this at a compute node on ABCI."; exit 1)

# ARCHIVE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TARFILE=/groups/gcc50495/dataset/FFHQ/images1024x1024.tar.gz

# Untar files
TARGET_DIR=$SGE_LOCALDIR/
mkdir -p ${TARGET_DIR}
tar xvf $TARFILE -C ${TARGET_DIR}

## symbolic link ##
# [ -d ~/summer2022/latent-diffusion/data/ffhq/images1024x1024 ] && unlink ~/summer2022/latent-diffusion/data/ffhq/images1024x1024
# ln -s $SGE_LOCALDIR/images1024x1024/ ~/summer2022/latent-diffusion/data/ffhq/
echo dataset setupped!
##################################################################

cd ~/summer2022/latent-diffusion/

source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load cmake/3.22.3
module load cuda/11.3/11.3.1
module load cudnn/8.2/8.2.4
module load nccl/2.9/2.9.9-1
module load python/3.8/3.8.13
source /home/acd13649ev/summer2022/latent-diffusion/ldm_env/bin/activate

echo beginning to train model
# python3 ~/summer2022/latent-diffusion/main.py --base ~/summer2022/latent-diffusion/configs/classifier/ffhq-256-emotions-label-smoothing-02-sn.yaml -t --gpus 0,1,2,3,4,5,6,7 --logdir /scratch/acd13649ev/logs/
python3 ~/summer2022/latent-diffusion/main.py --base ~/summer2022/latent-diffusion/configs/classifier/ffhq-256-emotions-mix-up-sn.yaml -t --gpus 0,1,2,3,4,5,6,7 --logdir /scratch/acd13649ev/logs/
