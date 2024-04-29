#!/bin/bash
#SBATCH --job-name="Vitenam_Sw_train"
#SBATCH --partition=gpu
#SBATCH --time=120:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=100G
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
  config=pretrain/swav/swav_1_gpu_resnet50/C7000_TP_0.1.yaml \
  config.DATA.TRAIN.DATASET_NAMES=[GJO] \
  config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
  config.DATA.TRAIN.DATA_PATHS=["/scratch/tjian/Data/GJO_SSL/images_tiles_224_pretrain/train"] \
  config.OPTIMIZER.num_epochs=100 \
  config.CHECKPOINT.DIR="/scratch/tjian/PythonProject/deep_plastic_SSL/checkpoints/train_weights/Paper_4/Vitenam/Self_train_GJO_Vit_90K/C7000_TP_0.1/vissl" \
  config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
  config.WEIGHTS_INIT.PARAMS_FILE="/scratch/tjian/PythonProject/deep_plastic_SSL/checkpoints/pretrained_model/R-50.pkl" \
  config.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks."
  

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate