from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import pandas as pd
import tqdm

ACCEPTABLE_EXTENSIONS = (
    ".c,.h,.cs,.cpp,.hpp,.c++,.h++,.cc,.hh,.C,.H,.java,.js,.lua,.md,.markdown,.php,"
    ".php3,.php4,.php5,.phps,.phpt,.pl,.pm,.pod,.perl,.py,.rb,.rs,.ts,.tsx,.vb"
).split(",")


def worker_fn(
    filename: str,
    args: argparse.Namespace,
    repositories: list[str],
    semaphore: mp.Semaphore,
):
    try:
        # When reading parquet file using pandas, the memory leakage is occurred and it
        # is almost impossible to free the leaked memory (even `gc.collect` does not
        # work). Thus we create the process for each parquet and just terminate for
        # releasing the leaked memory.
        data = pd.read_parquet(filename)
        data = data[data.repo_name.isin(repositories)]
    except Exception:
        semaphore.release()
        return

    for example in data.itertuples():
        if any(keyword in example.path for keyword in args.ignore_path_keywords):
            continue
        if all(not example.path.endswith(ext) for ext in ACCEPTABLE_EXTENSIONS):
            continue

        filename = example.repo_name.replace("/", "-") + ".jsonl"
        with open(os.path.join(args.output_dir, filename), "a") as fp:
            example = {
                "repository": example.repo_name,
                "path": example.path,
                "license": example.license,
                "content": example.content,
            }
            fp.write(json.dumps(example) + "\n")
    semaphore.release()


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.repository_list) as fp:
        repositories = [line.strip() for line in fp]

    sem = mp.Semaphore(args.num_cores)
    for filename in tqdm.tqdm(args.parquets):
        sem.acquire()
        worker = mp.Process(target=worker_fn, args=(filename, args, repositories, sem))
        worker.daemon = True
        worker.start()

    while sem.get_value() < args.num_cores:
        ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquets", nargs="+")
    parser.add_argument("--output-dir", default="repositories")
    parser.add_argument("--repository-list", default="repositories.txt")
    parser.add_argument("--ignore-path-keywords", nargs="+", default=["test"])
    parser.add_argument("--num-cores", type=int, default=mp.cpu_count())
    main(parser.parse_args())
