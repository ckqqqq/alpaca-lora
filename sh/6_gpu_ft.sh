# WORLD_SIZE=1 
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
cd /home/ckq/CHATGPT/alpaca-lora
conda activate
conda activate accel
conda info --envs

WORLD_SIZE=1  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6  deepspeed finetune.py \
    --base_model '/home/ckq/CHATGPT/0model_pretrained/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --gradient_accumulation_steps_per_gpu 4\
    --micro_batch_size 8 \
    --num_epochs 2 \
    --learning_rate 3e-4 \
    --cutoff_len 1024 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --resume_from_checkpoint  ''\
    --deepspeed_path '/home/ckq/CHATGPT/alpaca-lora/config/6_gpu_ft.yaml' \
    --group_by_length \
    --wandb_project  'llama-lora'
    # '/home/ckq/CHATGPT/alpaca-lora/lora-alpaca/checkpoint-200/global_step200/'
    # --batch_size  32\
