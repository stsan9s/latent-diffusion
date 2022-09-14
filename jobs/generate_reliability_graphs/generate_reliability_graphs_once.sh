#!/bin/bash

#$ -l rt_AG.small=1
#$ -l h_rt=1:00:00
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

echo dataset setupped!

source /etc/profile.d/modules.sh
module load gcc/9.3.0
module load cmake/3.22.3
module load cuda/11.3/11.3.1
module load cudnn/8.2/8.2.4
module load nccl/2.9/2.9.9-1
module load python/3.8/3.8.13
source /home/acd13649ev/summer2022/latent-diffusion/ldm_env/bin/activate

python scripts/confidence_analyzer.py --classifier_config /scratch/acd13649ev/logs/2022-09-01T18-35-24_ffhq-256-emotions/configs/2022-09-01T18-35-24-project.yaml --temperature_scaling True