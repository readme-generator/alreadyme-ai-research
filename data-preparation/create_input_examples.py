from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import shutil
from typing import Optional

import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def create_example_from_repository(
    repository: str,
    files: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    delimiter: str,
    max_total_tokens: int,
    max_readme_tokens: int,
    max_source_code_tokens: int,
    min_source_code_files: int,
) -> Optional[str]:
    files, readme_content = files.copy(), ""
    for filename in list(files):
        if filename.lower() == "readme.md":
            readme_content = f"{delimiter}\n> README.md of {repository}:\n\n"
            readme_content += files.pop(filename)
        elif (
            os.path.basename(filename).startswith(".")
            or "test" in filename.lower()
            or "example" in filename.lower()
            or "sample" in filename.lower()
            or filename.endswith(".md")
        ):
            files.pop(filename)

    if len(files) < min_source_code_files:
        return None

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

    newline_length = len(tokenizer.tokenize("\n\n"))
    while len(files) > 0:
        # Sample the files from the repository and truncate the tokenized and encoded
        # source code sequences. Similar to the case of `README.md` content, we use
        # `return_offsets_mapping` to truncate the original raw text.
        filename = random.choice(list(files))
        prefix = f"{delimiter}\n> Middle section of {filename}:\n\n"
        prefix_length = len(tokenizer.tokenize(prefix))

        max_length = min(max_source_code_tokens, max_total_tokens - num_tokens)
        max_length = max_length - prefix_length - newline_length
        if max_length <= max_source_code_tokens // 3:
            break

        source = files.pop(filename)
        source_encoding = tokenizer(
            source, add_special_tokens=False, return_offsets_mapping=True
        )

        # Truncate center of the source code if the source code is too long. Because
        # many source codes have comments (or license) and library imports, we have to
        # remove them and get the center-part of the codes to create the prompt with
        # meaningful contents.
        start_index, end_index = 0, -1
        if len(source_encoding.input_ids) > max_length:
            start_index = (len(source_encoding.input_ids) - max_length) // 2
            end_index = start_index + max_length

        start_offset = source_encoding.offset_mapping[start_index][0]
        end_offset = source_encoding.offset_mapping[end_index][1]
        content = prefix + source[start_offset:end_offset] + "\n\n"

        # The truncated prompt text (which consists of delimiter, filename and its
        # source code content) will be added before the current example string. It will
        # make the `README.md` content to be end of the prompt examples.
        num_tokens += prefix_length + newline_length
        num_tokens += len(source_encoding.input_ids[start_index:end_index])
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
                args.delimiter,
                args.max_total_tokens,
                args.max_readme_tokens,
                args.max_source_code_tokens,
                args.min_source_code_files,
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
    parser.add_argument("--num-sampling", type=int, default=4)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-readme-tokens", type=int, default=1024)
    parser.add_argument("--max-source-code-tokens", type=int, default=256)
    parser.add_argument("--min-source-code-files", type=int, default=5)
    parser.add_argument("--delimiter", default="$$$$$$")
    parser.add_argument("--num-cores", type=int, default=mp.cpu_count())
    main(parser.parse_args())
