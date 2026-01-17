set -x


export HOME="/data/qiaoshiqi"
wandb login "d3f9838b40b99e2c798867e0f501de72ab37a198"

# export HIP_VISIBLE_DEVICES=6,7
# export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# save_path=$HOME

torchrun --master_port 29501 \
    --nproc_per_node=4 \
    --nnodes=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/openai/gsm8k/main/train-00000-of-00001.parquet \
    data.val_files=$HOME/data/openai/gsm8k/main/test-00000-of-00001.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$HOME/hf_models/Qwen/Qwen2.5-7B-Instruct \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen2.5-7B-instruct \
    trainer.total_epochs=3 \
    trainer.logger='["console","wandb"]'

    # trainer.default_local_dir=$HOME/trained_models/${trainer.project_name}/${trainer.experiment_name}   \
    # --standalone --nnodes=1 --nproc_per_node=4    