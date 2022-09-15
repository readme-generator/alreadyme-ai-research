# Data Preparation 

To train `README.md` generation, we use [GitHub repository dataset](https://huggingface.co/datasets/codeparrot/github-code) which is published from [CodeParrot](https://huggingface.co/codeparrot). It is about 1TB and the compressed size is 300GB, so make sure to download the dataset on more than 300GB disk.

There are three scripts to create a dataset.
* [find_readme_repositories.py](./find_readme_repositories.py): Find repositories which contain `README.md` file.
* [extract_source_codes.py](./extract_source_codes.py): Extract source codes in parquets and write into jsonl files.
* [create_input_examples.py](./create_input_examples.py): Sample the source codes and create input prompt examples. Note that current prompt template is **version 3**.


## Requirements
* pandas
* tqdm
* transformers

## How to create the dataset?

### Download `CodeParrot` GitHub dataset
As mentioned above, we use [GitHub repository dataset](https://huggingface.co/datasets/codeparrot/github-code) of [CodeParrot](https://huggingface.co/codeparrot). Because the dataset is too large, we consider to preprocess from raw-format files. The dataset consists of several parquets and you can check in [here](https://huggingface.co/datasets/codeparrot/github-code/tree/main/data). Both downloading parquets directly with Git LFS and using huggingface `datasets` are available.

```python
from datasets import load_dataset

load_dataset("codeparrot/github-code")
```

After downloading the dataset, make sure the downloaded parquet directory is accessible.

### Get a list of repositories
There are about 10M repositories in the dataset. However, some of them do not have `README.md` file and many of them contain improper document (e.g. empty, too short, and invalid contents). Before extracting source codes of all repositories, it would be better to filter the invalid repositories.

```bash
$ python find_readme_repositories.py /path/to/parquets/*.parquet
```

Make sure `repositories.txt` is generated successfully.

### Extract source codes
Now we need source codes of each repository. `extract_source_codes.py` will extract the codes from parquets and group them by repositories.

```bash
$ python extract_source_codes.py /path/to/parquets/*.parquet
```

After running the above command, you can see `repositories` directory which contains jsonl files.

### Generate input prompt examples
Finally we are going to generate the input prompt examples which will be used to fine-tune the large-scale language model.

```bash
$ python create_input_examples.py --tokenizer [tokenizer] --num-sampling [number of sampling] 
```

Contrary to the previous two scripts, this program provides many options for generation. First, you have to specify the tokenizer which will be used with the target model (e.g. `bigscience/bloom-1b7`). Also the number of sampling is required. Basically, the source codes will be randomly sampled and choiced, so many files will be dropped from the input prompt. Hence you can sample multiple times to make various conditions of the prompt.

If you want to increase/decrease the total length of the sequence (number of tokens in the prompt), then use `--max-total-tokens`. You can also limit the length of `README.md` and source codes with `--max-readme-tokens` and `--max-source-code-tokens` respectively. With `--min-source-code-files`, the repositories with less than the given value will be ignored.

`README.md` with many URLs (e.g. badges, documentations and etc.) can be a problem. It can be controlled with `--max-urls-in-readme`. Also invalid documents like too-long so that should be truncated and without title (starts with `#`) are skipped basically. You can disable them with `--include-no-title` and `--include-longer-readme` respectively.

Change the separator with your own patterns. Just use `--separator` to change it. Default is `&&&&&&`.

## Usage
Here are detailed usage of the scripts.

```bash
$ python find_readme_repositories.py --help
usage: find_readme_repositories.py [-h] [--output OUTPUT] [--min-readme-lines MIN_README_LINES]
                                   [--num-cores NUM_CORES]
                                   parquets [parquets ...]

positional arguments:
  parquets

options:
  -h, --help            show this help message and exit
  --output OUTPUT
  --min-readme-lines MIN_README_LINES
  --num-cores NUM_CORES
```

```bash
$ python extract_source_codes.py --help
usage: extract_source_codes.py [-h] [--output-dir OUTPUT_DIR] [--repository-list REPOSITORY_LIST]
                               [--ignore-path-keywords IGNORE_PATH_KEYWORDS [IGNORE_PATH_KEYWORDS ...]]
                               [--num-cores NUM_CORES]
                               parquets [parquets ...]

positional arguments:
  parquets

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
  --repository-list REPOSITORY_LIST
  --ignore-path-keywords IGNORE_PATH_KEYWORDS [IGNORE_PATH_KEYWORDS ...]
  --num-cores NUM_CORES
```

```bash
$ python create_input_examples.py --help
usage: create_input_examples.py [-h] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR] [--tokenizer TOKENIZER]
                                [--num-sampling NUM_SAMPLING] [--max-total-tokens MAX_TOTAL_TOKENS]
                                [--max-readme-tokens MAX_README_TOKENS] [--max-source-code-tokens MAX_SOURCE_CODE_TOKENS]
                                [--min-source-code-files MIN_SOURCE_CODE_FILES] [--max-urls-in-readme MAX_URLS_IN_README]
                                [--include-no-title] [--include-longer-readme] [--separator SEPARATOR] [--num-cores NUM_CORES]

options:
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
  --output-dir OUTPUT_DIR
  --tokenizer TOKENIZER
  --num-sampling NUM_SAMPLING
  --max-total-tokens MAX_TOTAL_TOKENS
  --max-readme-tokens MAX_README_TOKENS
  --max-source-code-tokens MAX_SOURCE_CODE_TOKENS
  --min-source-code-files MIN_SOURCE_CODE_FILES
  --max-urls-in-readme MAX_URLS_IN_README
  --include-no-title
  --include-longer-readme
  --separator SEPARATOR
  --num-cores NUM_CORES
```