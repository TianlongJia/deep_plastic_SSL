#!/bin/bash
#SBATCH --job-name="Sw_train_100_epochs"
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=200G
#SBATCH --account=research-ceg-wm
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=T.Jia@tudelft.nl

# Load modules:
module load miniconda3
module load cuda/11.7


# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

conda activate vissl_env
srun python tools/run_distributed_engines.py \
  hydra.verbose=true \
  config=pretrain/swav/swav_1_gpu_RN50_scratch.yaml \
  config.DATA.TRAIN.DATASET_NAMES=[GJO] \
  config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
  config.DATA.TRAIN.DATA_PATHS=["/scratch/tjian/Data/GJO_SSL/images_tiles_224_pretrain/train"] \
  config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=16 \
  config.OPTIMIZER.num_epochs=100 \
  config.CHECKPOINT.DIR="/scratch/tjian/PythonProject/deep_plastic_SSL/checkpoints/train_weights/Self_train_tiles_224/pretrain_25per/RN50_Sw_100e_Scratch/vissl" \
  config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true
  

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate