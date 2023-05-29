# Fine-Tuning with QLoRA

This repository can help to instruct-tune Open LLaMA or RedPajama models on consumer hardware using QLoRA (Original implementation [here](https://github.com/artidoro/qlora)). It's mostly based on the original alpaca-lora repo which can be found [here](https://github.com/tloen/alpaca-lora). Please note that this has only been tested on Open LLama 3b and RedPajama 3b Models, but should work with other models. Contributions are welcome!

## Training (finetune.py)

This file contains a straightforward application of QLoRA PEFT to the Open LLaMA / RedPajama model, as well as some code related to prompt construction and tokenization. PRs adapting this code to support larger models are always welcome.

**Example usage:**

For Open LLaMa

    python finetune.py \
        --base_model 'openlm-research/open_llama_3b_600bt_preview' \
        --data_path '../datasets/dolly.json' \
        --num_epochs=3 \
        --cutoff_len=512 \
        --group_by_length \
        --output_dir='./dolly-lora-3b' \
        --lora_r=16 \
        --lora_target_modules='[q_proj,v_proj]'

For RedPajama

    python finetune.py   \
    --base_model='togethercomputer/RedPajama-INCITE-Base-3B-v1' \
    --data_path='../datasets/alpaca-codeleet/dolly.json'   \
    --num_epochs=3   \
    --cutoff_len=512   \
    --group_by_length   \
    --output_dir='./dolly-lora-rp-3b-t1' \
    --lora_r=16 \
    --lora_target_modules='["query_key_value"]' 

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

For Open LLaMa

    python generate.py \
        --base_model 'openlm-research/open_llama_3b_600bt_preview' \
        --lora_weights './lora-alpaca'
        
For RedPajama

    python generate.py  \
    --base_model 'togethercomputer/RedPajama-INCITE-Base-3B-v1'  \
    --lora_weights './dolly-lora-rp-3b-t1/'
       

# Acknowledgements

We would like to express our heartfelt gratitude to **Meta** for releasing LLaMA . Without this pioneering technology, the foundations of projects like **Open Llama** and **Alpaca** wouldn't exist. We sincerely appreciate the immense contributions you've made to the field.

Our acknowledgements also extend to the teams behind **Open LLaMA**, **Together Computer**, **Alpaca** and **Alpaca LoRA**.. You can find more about their excellent work on their respective GitHub repositories:

- [Open Llama](https://github.com/openlm-research/open_llama)
- [Together Computer](https://github.com/togethercomputer)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Alpaca LoRa](https://github.com/tloen/alpaca-lora)

Lastly, we would like to express our thanks to the developers of **QLoRA** and **bitsandbytes** Your efforts have been instrumental in advancing the field, and we're grateful for your contributions. More information about these projects can be found at:

- [QLoRA](https://github.com/artidoro/qlora)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)


Thank you all for your commitment to innovation and for making these projects possible.


    
