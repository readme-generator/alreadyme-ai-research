from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import re
import shutil
from typing import Optional

import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

os.environ["TOKENIZERS_PARALLELISM"] = "false"

REGEX_URL_PATTERN = (
    r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\."
    r"[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-"
    r"Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
)


def create_example_from_repository(
    repository: str,
    files: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    separator: str,
    max_total_tokens: int,
    max_readme_tokens: int,
    max_source_code_tokens: int,
    min_source_code_files: int,
    max_urls_in_readme: int,
    include_no_title: bool,
    include_longer_readme: bool,
) -> Optional[str]:
    files, readme_content = files.copy(), ""
    for filename in list(files):
        if filename.lower() == "readme.md":
            readme_content = files.pop(filename)
        elif (
            os.path.basename(filename).startswith(".")
            or "test" in filename.lower()
            or filename.endswith(".md")
        ):
            files.pop(filename)

    if len(files) < min_source_code_files:
        return None
    if len(re.findall(REGEX_URL_PATTERN, readme_content)) > max_urls_in_readme:
        return None
    if not include_no_title and not readme_content.lstrip().startswith("#"):
        return None

    readme_content = (
        f"{separator}\n"
        f"$ git config --get remote.origin.url\n"
        f"https://github.com/{repository}.git\n\n"
        f"{separator}\n"
        f"$ cat README.md\n"
        f"{readme_content}"
    )

    # We will only use the first `max_readme_tokens` by truncating the tokenized and
    # encoded sequences. Note that we have to preserve the original raw text format, so
    # we use `return_offsets_mapping` to truncate with the mapped offset position.
    readme_encoding = tokenizer(
        readme_content,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if not include_longer_readme and len(readme_encoding.input_ids) > max_readme_tokens:
        return None

    num_tokens = min(len(readme_encoding.input_ids), max_readme_tokens)
    example_string = readme_content[: readme_encoding.offset_mapping[num_tokens - 1][1]]

    while files and num_tokens < max_total_tokens:
        # Sample the files from the repository and truncate the tokenized and encoded
        # source code sequences. Similar to the case of `README.md` content, we use
        # `return_offsets_mapping` to truncate the original raw text.
        filename = random.choice(list(files))
        max_length = min(max_source_code_tokens, max_total_tokens - num_tokens)

        content = f"\n\n{separator}\n$ head -n $$N$$ {filename}\n{files.pop(filename)}"
        content_encoding = tokenizer(
            content,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
        )
        content = content[: content_encoding.offset_mapping[-1][1]].lstrip()

        if content.count("\n") - 1 == 0:
            break
        content = content.replace("$$N$$", str(content.count("\n") - 1)) + "\n\n"

        # The truncated prompt text (which consists of separator, filename and its
        # source code content) will be added before the current example string. It will
        # make the `README.md` content to be end of the prompt examples.
        num_tokens += len(content_encoding.input_ids)
        example_string = content + example_string

    return example_string


def worker_fn(filenames: list[str], args: argparse.Namespace, return_queue: mp.Queue):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for filename in filenames:
        with open(filename) as fp:
            files, repository = {}, None
            for line in fp:
                file = json.loads(line)
                files[file["path"]] = file["content"]
                repository = file["repository"]

        for _ in range(args.num_sampling):
            example_string = create_example_from_repository(
                repository,
                files,
                tokenizer,
                args.separator,
                args.max_total_tokens,
                args.max_readme_tokens,
                args.max_source_code_tokens,
                args.min_source_code_files,
                args.max_urls_in_readme,
                args.include_no_title,
                args.include_longer_readme,
            )
            if example_string is None:
                break

            output = "".join(random.choices("0123456789ABCDEF", k=16)) + ".txt"
            with open(os.path.join(args.output_dir, output), "w") as fp:
                fp.write(example_string)
        return_queue.put(True)
    return_queue.put(False)


def main(args: argparse.Namespace):
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    filenames = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    workers, return_queue = [], mp.Queue()

    for i in range(args.num_cores):
        worker = mp.Process(
            target=worker_fn,
            args=(filenames[i :: args.num_cores], args, return_queue),
            daemon=True,
        )
        worker.start()
        workers.append(worker)

    num_completes = 0
    with tqdm.tqdm(filenames) as tbar:
        while num_completes < len(workers):
            return_value = return_queue.get()
            tbar.update(1 if return_value else 0)
            num_completes += 1 if not return_value else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="repositories")
    parser.add_argument("--output-dir", default="examples")
    parser.add_argument("--tokenizer", default="bigscience/bloom-1b7")
    parser.add_argument("--num-sampling", type=int, default=5)
    parser.add_argument("--max-total-tokens", type=int, default=2046)
    parser.add_argument("--max-readme-tokens", type=int, default=1024)
    parser.add_argument("--max-source-code-tokens", type=int, default=256)
    parser.add_argument("--min-source-code-files", type=int, default=5)
    parser.add_argument("--max-urls-in-readme", type=int, default=5)
    parser.add_argument("--include-no-title", action="store_true", default=False)
    parser.add_argument("--include-longer-readme", action="store_true", default=False)
    parser.add_argument("--separator", default="&&&&&&")
    parser.add_argument("--num-cores", type=int, default=mp.cpu_count())
    main(parser.parse_args())
