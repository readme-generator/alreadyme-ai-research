<h1 align="center">ALREADYME.md AI Research</h1>
<p align="center">
	<a href="https://github.com/readme-generator/alreadyme-ai-research/issues">
		<img alt="GitHub issues" src="https://img.shields.io/github/issues/readme-generator/alreadyme-ai-research">
	</a>
    <a href="https://github.com/readme-generator/alreadyme-ai-research/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/readme-generator/alreadyme-ai-research">
    </a>
<a href="https://app.fossa.com/projects/git%2Bgithub.com%2Freadme-generator%2Falreadyme-ai-research?ref=badge_shield" alt="FOSSA Status"><img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2Freadme-generator%2Falreadyme-ai-research.svg?type=shield"/></a>
    <a href="https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
	<a href="https://www.codefactor.io/repository/github/readme-generator/alreadyme-ai-research">
		<img src="https://www.codefactor.io/repository/github/readme-generator/alreadyme-ai-research/badge" alt="CodeFactor" />
	</a>
    <br/><br/>
    <b>Generate README.md with GPT-3 few-shot learning</b>
</p>

--------


[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Freadme-generator%2Falreadyme-ai-research.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Freadme-generator%2Falreadyme-ai-research?ref=badge_large)

## Introduction

**alreadyme-ai-research** is a core project for generating `README.md` from source codes in any repository. The AI model reads some parts of the source codes and write a corresponding `README.md` document. **ALREADYME.md team** is currently providing a service about this feature, and you can find our results [on this page](https://github.com/readme-generator).

This repository contains several subprojects. You can see the detailed descriptions in the directories.

* **[data-preparation](./data-preparation)**: The source codes for preparing a train dataset.
* **[model-finetuning](./model-finetuning)**: How to fine-tune large-scale language models efficiently.
* **[sentence-generation](./sentence-generation)**: Efficient and scalable way to generate sentences for model serving.

## How it works?
As the large-scale models like [GPT-3](https://github.com/openai/gpt-3) have shown, few-shot learning is the most important key for building the generalized language model. They can understand what they should have to write according to the previous prompt and few-shot examples. Using this features, they can do almost anything without fine-tuning. They can summarize the news, answer the questions, and even make a conversation!

[OpenAI Codex](https://openai.com/blog/openai-codex/) introduced new large-scale langauge model for programming languages by fine-tuning GPT-3. Now we can expect the generalized performance (few-shot learning) on the programming languages. For instance, create a docstring from the source code, write new code from the description (and this is how [Copilot](https://github.com/features/copilot) works), and translate from Python to Java.

We use [BLOOM](https://huggingface.co/bigscience/bloom) which is for open-science and open-access of large-scale language model. BLOOM supports multilingual which are not only natural languages, but the programming languages as well. We designed prompt templates and found best version of them.

```
&&&&&&
$ head -n 30 model-finetuning/src/data.py
from __future__ import annotations

from dataclasses import dataclass

import torch
[...]

&&&&&&
$ head -n 37 model-finetuning/src/train.py
from __future__ import annotations

import argparse
import os
[...]

&&&&&&
$ git config --get remote.origin.url
https://github.com/readme-generator/alreadyme-ai-research.git

&&&&&&
$ cat README.md
[...]
```

All the examples will be separated by `&&&&&&`. We designed to make BLOOM to perform (or simulate) the linux bash command. BLOOM will read some parts of the source codes from the given prompt and generate a proper `README.md` file.

For more details, check out our **[model-finetuning](./model-finetuning)** subproject.

## License
**alreadyme-ai-research** is released under the Apache License 2.0. License can be found in [here](./LICENSE).

## Citations
```bibtex
@misc{https://doi.org/10.48550/arxiv.2005.14165,
	title        = {Language Models are Few-Shot Learners},
	author       = {Brown, Tom B. and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and Agarwal, Sandhini and Herbert-Voss, Ariel and Krueger, Gretchen and Henighan, Tom and Child, Rewon and Ramesh, Aditya and Ziegler, Daniel M. and Wu, Jeffrey and Winter, Clemens and Hesse, Christopher and Chen, Mark and Sigler, Eric and Litwin, Mateusz and Gray, Scott and Chess, Benjamin and Clark, Jack and Berner, Christopher and McCandlish, Sam and Radford, Alec and Sutskever, Ilya and Amodei, Dario},
	year         = 2020,
	publisher    = {arXiv},
	doi          = {10.48550/ARXIV.2005.14165},
	url          = {https://arxiv.org/abs/2005.14165},
	copyright    = {arXiv.org perpetual, non-exclusive license},
	keywords     = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences}
}
```
```bibtex
@misc{https://doi.org/10.48550/arxiv.2107.03374,
	title        = {Evaluating Large Language Models Trained on Code},
	author       = {Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and Ray, Alex and Puri, Raul and Krueger, Gretchen and Petrov, Michael and Khlaaf, Heidy and Sastry, Girish and Mishkin, Pamela and Chan, Brooke and Gray, Scott and Ryder, Nick and Pavlov, Mikhail and Power, Alethea and Kaiser, Lukasz and Bavarian, Mohammad and Winter, Clemens and Tillet, Philippe and Such, Felipe Petroski and Cummings, Dave and Plappert, Matthias and Chantzis, Fotios and Barnes, Elizabeth and Herbert-Voss, Ariel and Guss, William Hebgen and Nichol, Alex and Paino, Alex and Tezak, Nikolas and Tang, Jie and Babuschkin, Igor and Balaji, Suchir and Jain, Shantanu and Saunders, William and Hesse, Christopher and Carr, Andrew N. and Leike, Jan and Achiam, Josh and Misra, Vedant and Morikawa, Evan and Radford, Alec and Knight, Matthew and Brundage, Miles and Murati, Mira and Mayer, Katie and Welinder, Peter and McGrew, Bob and Amodei, Dario and McCandlish, Sam and Sutskever, Ilya and Zaremba, Wojciech},
	year         = 2021,
	publisher    = {arXiv},
	doi          = {10.48550/ARXIV.2107.03374},
	url          = {https://arxiv.org/abs/2107.03374},
	copyright    = {arXiv.org perpetual, non-exclusive license},
	keywords     = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences}
}
```
```bibtex
@misc{https://doi.org/10.48550/arxiv.2106.09685,
	title        = {LoRA: Low-Rank Adaptation of Large Language Models},
	author       = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
	year         = 2021,
	publisher    = {arXiv},
	doi          = {10.48550/ARXIV.2106.09685},
	url          = {https://arxiv.org/abs/2106.09685},
	copyright    = {arXiv.org perpetual, non-exclusive license},
	keywords     = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences}
}
```
```bibtex
@misc{bigscience_2022,
	title        = {Bigscience large open-science openaccess multilingual language model.},
	author       = {BigScience},
	year         = 2022,
	journal      = {bigscience/bloom · Hugging Face},
	url          = {https://huggingface.co/bigscience/bloom}
}
```