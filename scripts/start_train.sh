# . /scratch/vladimir_albrekht/projects/smollm/nanotron/scripts/start_train.sh

# pkill -f run_train.py kill ghost processes
export HF_HOME=/data/vladimir_albrekht/hf_cache
export TORCH_NCCL_ENABLE_MONITORING=0
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=100000
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /scratch/vladimir_albrekht/projects/smollm/nanotron
eval "$(conda shell.bash hook)"
conda activate smoll_llm_v
CUDA_LAUNCH_BLOCKING=1 CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=8 run_train.py --config-file /scratch/vladimir_albrekht/projects/smollm/nanotron/configs/qwen_3B.yaml