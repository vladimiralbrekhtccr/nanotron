# . /scratch/vladimir_albrekht/projects/smollm/nanotron/scripts/convert_to_hf.sh
eval "$(conda shell.bash hook)"
conda activate smoll_llm_v
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --master-port 29501 --nproc_per_node=1 /scratch/vladimir_albrekht/projects/smollm/nanotron/examples/llama/convert_nanotron_to_hf.py \
--checkpoint_path /scratch/vladimir_albrekht/projects/smollm/output/nanotron_500M/126000 \
--save_path /scratch/vladimir_albrekht/projects/smollm/output/nanotron_500M/126000/hf \
--tokenizer_name /scratch/vladimir_albrekht/projects/smollm/models/tokenizers/kk_tokenizer_qwen_2