from __future__ import annotations

import argparse
import glob
import json
import os
import random

from transformers import AutoTokenizer, PreTrainedTokenizerBase

ACCEPTABLE_EXTENSIONS = (
    ".c,.h,.cs,.cpp,.hpp,.c++,.h++,.cc,.hh,.C,.H,.java,.js,.lua,.md,.markdown,.php,"
    ".php3,.php4,.php5,.phps,.phpt,.pl,.pm,.pod,.perl,.py,.rb,.rs,.ts,.tsx,.vb"
).split(",")


def create_input_prompt(
    repository: str,
    files: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    separator: str,
    max_total_tokens: int,
    max_source_code_tokens: int,
) -> str | None:
    files = files.copy()
    for filename in list(files):
        # Remove the hidden (starts with `.`), test, and markdown files from the given
        # file list. They should be ignored because they are not necessary for
        # generating the `README.md` file.
        if (
            os.path.basename(filename).startswith(".")
            or "test" in filename.lower()
            or filename.lower().endswith(".md")
        ):
            files.pop(filename)

    # Create a last example of the prompt which requests the model to generate the
    # content of `README.md` file. In addition, we give the git url to the model.
    prompt = (
        f"{separator}\n"
        f"$ git config --get remote.origin.url\n"
        f"{repository}\n\n"
        f"{separator}\n"
        f"$ cat README.md\n"
    )
    num_tokens = len(tokenizer.tokenize(prompt))

    while files and num_tokens < max_total_tokens:
        # Sample the files from the repository and truncate the tokenized and encoded
        # source code sequences. Similar to the case of `README.md` content, we use
        # `return_offsets_mapping` to truncate the original raw text.
        filename = random.choice(list(files))
        for name in files:
            if name.lower() == "license.md" or name.lower() == "license":
                filename = name
                break
        max_length = min(max_source_code_tokens, max_total_tokens - num_tokens)

        example = f"\n\n{separator}\n$ head -n $$N$$ {filename}\n{files.pop(filename)}"
        encoding = tokenizer(
            example,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        example = example[: encoding.offset_mapping[-1][1]].lstrip()

        # If there is no line of source code, then stop generating example prompt
        # because there is no space to insert new example.
        if example.count("\n") < 1:
            break
        example = example.replace("$$N$$", str(example.count("\n") - 1)) + "\n\n"

        # The truncated prompt text (which consists of separator, filename and its
        # source code content) will be added before the current example string. It will
        # make the `README.md` content to be end of the prompt examples.
        num_tokens += len(encoding.input_ids)
        prompt = example + prompt
    return prompt


def main(args: argparse.Namespace):
    filenames = glob.glob(os.path.join(args.directory, "**", "*.*"), recursive=True)
    filenames = [os.path.normpath(filename) for filename in filenames]
    filenames = [
        filename
        for filename in filenames
        if not any(keyword in filename for keyword in args.ignore_path_keywords)
        and any(filename.lower().endswith(ext) for ext in ACCEPTABLE_EXTENSIONS)
        and not filename.startswith(".")
    ]

    data = {}
    for filename in filenames:
        with open(filename) as fp:
            data[filename] = fp.read()

    with os.popen("git config --get remote.origin.url") as pipe:
        giturl = pipe.read().strip()
    giturl = giturl if giturl.endswith(".git") else f"{giturl}.git"

    if args.create_prompt:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        prompt = create_input_prompt(
            repository=giturl,
            files=data,
            tokenizer=tokenizer,
            separator=args.separator,
            max_total_tokens=args.max_total_tokens,
            max_source_code_tokens=args.max_source_code_tokens,
        )
        print(prompt)
    else:
        print(json.dumps({"githubOriginalUrl": giturl, "data": data}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=".")
    parser.add_argument("--tokenizer", default="bigscience/bloom-1b7")
    parser.add_argument("--ignore-path-keywords", nargs="+", default=["test"])
    parser.add_argument("--create-prompt", default=False, action="store_true")
    parser.add_argument("--max-total-tokens", type=int, default=1348)
    parser.add_argument("--max-source-code-tokens", type=int, default=256)
    parser.add_argument("--separator", default="&&&&&&")
    main(parser.parse_args())
