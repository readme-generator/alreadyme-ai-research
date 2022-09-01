from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import shutil

import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_example_from_repository(
    files: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    delimiter: str,
    max_total_tokens: int,
    max_readme_tokens: int,
    max_source_code_tokens: int,
) -> str:
    files, readme_content = files.copy(), ""
    for filename in list(files):
        if filename.lower() == "readme.md":
            readme_content = f"{delimiter}\n{filename}:\n{files.pop(filename)}"
        elif os.path.basename(filename).startswith(".") or "test" in filename.lower():
            files.pop(filename)

    # We will only use the first `max_readme_tokens` by truncating the tokenized and
    # encoded sequences. Note that we have to preserve the original raw text format, so
    # we use `return_offsets_mapping` to truncate with the mapped offset position.
    readme_content_encoding = tokenizer(
        readme_content,
        max_length=max_readme_tokens,
        truncation=True,
        return_offsets_mapping=True,
    )
    num_tokens = len(readme_content_encoding.input_ids)
    example_string = readme_content[: readme_content_encoding.offset_mapping[-1][1]]

    while files and num_tokens + 2 < max_total_tokens:
        # Sample the files from the repository and truncate the tokenized and encoded
        # source code sequences. Similar to the case of `README.md` content, we use
        # `return_offsets_mapping` to truncate the original raw text.
        filename = random.choice(list(files))
        max_length = min(max_source_code_tokens - 2, max_total_tokens - num_tokens - 2)

        content = f"{delimiter}\n{filename}:\n{files.pop(filename)}"
        content_encoding = tokenizer(
            content,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
        )
        content = content[: content_encoding.offset_mapping[-1][1]] + "\n\n"

        # The truncated prompt text (which consists of delimiter, filename and its
        # source code content) will be added before the current example string. It will
        # make the `README.md` content to be end of the prompt examples.
        num_tokens += len(content_encoding.input_ids) + 2
        example_string = content + example_string

    return example_string


def worker_fn(filenames: list[str], args: argparse.Namespace, return_queue: mp.Queue):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for filename in filenames:
        with open(filename) as fp:
            files = {}
            for line in fp:
                file = json.loads(line)
                files[file["path"]] = file["content"]

        for _ in range(args.num_sampling):
            example_string = create_example_from_repository(
                files,
                tokenizer,
                args.delimiter,
                args.max_total_tokens,
                args.max_readme_tokens,
                args.max_source_code_tokens,
            )
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
    parser.add_argument("--tokenizer", default="bigscience/bloom-2b5")
    parser.add_argument("--num-sampling", type=int, default=10)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-readme-tokens", type=int, default=1024)
    parser.add_argument("--max-source-code-tokens", type=int, default=256)
    parser.add_argument("--delimiter", default="================================")
    parser.add_argument("--num-cores", type=int, default=mp.cpu_count())
    main(parser.parse_args())
