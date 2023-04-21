# WORLD_SIZE=1 
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
cd /home/ckq/CHATGPT/alpaca-lora
conda activate
conda activate accel
conda info --envs

WORLD_SIZE=1  CUDA_VISIBLE_DEVICES=1,2  deepspeed finetune.py \
    --base_model '/home/ckq/CHATGPT/0model_pretrained/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --batch_size 256 \
    --micro_batch_size 16 \
    --num_epochs 2 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --deepspeed_path '/home/ckq/CHATGPT/alpaca-lora/config/simple2.yaml' \
    --group_by_length
