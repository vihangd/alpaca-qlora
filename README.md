# Fine-Tuning with QLoRA

This repository can help to instruct-tune Open LLaMA on consumer hardware using QLoRA. It's mostly based on the original alpaca-lora repo which can be found [here](https://github.com/tloen/alpaca-lora). Please note that this has only been tested on Open LLama Models, but should work with other models. Contributions are welcome!

## Training (finetune.py)

This file contains a straightforward application of QLoRA PEFT to the Open LLaMA model, as well as some code related to prompt construction and tokenization. PRs adapting this code to support larger models are always welcome.

**Example usage:**


    python finetune.py \
        --base_model 'openlm-research/open_llama_3b_600bt_preview' \
        --data_path '../datasets/dolly.json' \
        --num_epochs=3 \
        --cutoff_len=512 \
        --group_by_length \
        --output_dir='./dolly-lora-3b' \
        --lora_r=16 \
        --lora_target_modules='[q_proj,v_proj]'

We can also tweak our hyperparameters (similar to alpaca-lora):

    python finetune.py \
        --base_model 'openlm-research/open_llama_3b_600bt_preview \
        --data_path 'yahma/alpaca-cleaned' \
        --output_dir './lora-alpaca' \
        --batch_size 128 \
        --micro_batch_size 4 \
        --num_epochs 3 \
        --learning_rate 1e-4 \
        --cutoff_len 512 \
        --val_set_size 2000 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[q_proj,v_proj]' \
        --train_on_inputs \
        --group_by_length

## Inference (generate.py)
This file reads the foundation model from the Hugging Face model hub and the LoRA weights from trained peft model, and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.

Example usage:    

    python generate.py \
        --load_8bit \
        --base_model 'openlm-research/open_llama_3b_600bt_preview' \
        --lora_weights './lora-alpaca'
    
