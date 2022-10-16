# Sentence generation

This subproject contains several utilities for sentence generation.

* [src/compile.py](src/compile.py): Compile the exported bloom model with PyTorch JIT.
* [src/modeling_fast_bloom.py](src/modeling_fast_bloom.py): Source code of the fast and jit-scriptable version of BLOOM model.
* [src/generation.py](src/generation.py): The entire generation framework for model serving. This code will be used for our [alreadyme-ai-serving](https://github.com/readme-generator/alreadyme-ai-serving) service.
* [src/create_input_prompt.py](src/create_input_prompt.py): A script to create input prompt for model. You can get either JSON-format or full plain text.

## How it works

Transformer models basically work with batched sequences. Since BLOOM-like models use relative position embedding, it is possible to use fixed sequence generation buffer and queue-like FIFO context update. That is, the past key-value pairs and attention masks will be shifted right-to-left to generate new tokens with fixed length. We first initialize the context tensors with fixed `max_length`, and then overwrite with newly requested input prompts. Almost of the generation time is from recursive (auto-regressive) token generation part. Our new framework just predicts the next tokens and then push to the right of the contexts. This makes the required tensor space be assigned at first and then the additional memory consumption will not be happened.

## Experimental project
This subproject is for experiments and the codes can be applied to our serving service, as mentioned above.