#!/bin/bash
#SBATCH --job-name="SSL_2628"
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

conda activate vissl_env1
srun python tools/object_detection_benchmark_1.py \
    --config-file /scratch/tjian/PythonProject/deep_plastic_SSL/configs/config/benchmark/object_detection/COCOInstance/R50_C4_COCO/tiles_224/FasterRCNN_DP_GJO_train_2628.yaml \
     --num-gpus 1 SOLVER.MAX_ITER 65700 TEST.EVAL_PERIOD 657 SOLVER.IMS_PER_BATCH 4 MODEL.BACKBONE.FREEZE_AT 2 MODEL.WEIGHTS /scratch/tjian/PythonProject/deep_plastic_SSL/checkpoints/train_weights/Self_train_tiles_224/RN50_Sw_100e_FTAL/detectron2/RN50_Sw_100e_FTAL.torch OUTPUT_DIR /scratch/tjian/PythonProject/deep_plastic_SSL/checkpoints/train_weights/SSL_models_F2/SW_RN50_100e_FTAL/SSL_train_2628/


/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate