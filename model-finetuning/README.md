# BLOOM Fine-tuning

## What is LoRA?
**LoRA: Low-Rank Adaptation of Large Language Models** [[paper]](https://arxiv.org/pdf/2106.09685) [[github]](https://github.com/microsoft/LoRA)
> An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at this https URL.

LoRA is for fine-tuning large-scale models with lower resources and limited GPU memory environment. Recently, some researches show that well-trained language models do not need high-rank gradients (or complex weight modification) for fine-tuning to the specific tasks. [P-tuning](https://github.com/THUDM/P-tuning) and [P-tuning v2](https://github.com/THUDM/P-tuning-v2) show that it is sufficient to insert the trainable vectors as a prefix of sequences to change the behavior of transformer models. Even they outperform the fine-tuning.

In this project, we use BLOOM model with LoRA to generate `README.md` content.

## Requirements
* apex
* numpy
* omegaconf
* pytorch_lightning
* tokenizers
* torch
* transformers
* wandb

This repository supports [NVIDIA Apex](https://github.com/NVIDIA/apex). It is optional, but we strongly recommend to use this to accelerate the training. Run the below codes in the terminal to install apex and enable performance boosting:

```bash
$ git clone https://github.com/NVIDIA/apex
$ sed -i "s/or (bare_metal_minor != torch_binary_minor)//g" apex/setup.py
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
$ rm -rf apex
```

## Usage
You can fine-tune the model by using:
```bash
$ python src/train.py config/bloom-1b7.yaml
```
After training, you can extract to huggingface format from the checkpoint by:
```bash
$ python src/export.py config/bloom-1b7.yaml last.ckpt
```

The basic structure of a configuration is like below. Change the hyperparameters and save new configuration if you need.
```yaml
data:
  filenames: examples/*.txt
  max_length: ...
  random_state: 42

model:
  transformer:
    pretrained_model_name_or_path: ...
  lora:
    lora_dim: 8
    lora_scale: 4
    attention_layer_name: query_key_value

optim:
  optimizer:
    lr: ...
    betas: [0.9, 0.999]
    eps: 1e-6
    weight_decay: 0.1
  scheduler:
    name: linear
    num_warmup_steps: ...
    num_training_steps: ...

train:
  batch_size: ...
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  gradient_checkpointing: true
  log_every_n_steps: 10
  save_every_n_train_steps: 2500
```
