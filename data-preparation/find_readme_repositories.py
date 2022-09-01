from __future__ import annotations

import argparse
import multiprocessing as mp

import pandas as pd
import tqdm


def worker_fn(filename: str, args: argparse.Namespace, semaphore: mp.Semaphore):
    try:
        # When reading parquet file using pandas, the memory leakage is occurred and it
        # is almost impossible to free the leaked memory (even `gc.collect` does not
        # work). Thus we create the process for each parquet and just terminate for
        # releasing the leaked memory.
        data = pd.read_parquet(filename)
    except Exception:
        semaphore.release()
        return

    with open(args.output, "a") as fp:
        for example in data.itertuples():
            if (
                example.path.lower() == "readme.md"
                and len(example.content.splitlines()) >= args.min_readme_lines
            ):
                fp.write(example.repo_name + "\n")
    semaphore.release()


def main(args: argparse.Namespace):
    semaphore = mp.Semaphore(args.num_cores)
    for filename in tqdm.tqdm(args.parquets):
        semaphore.acquire()
        worker = mp.Process(target=worker_fn, args=(filename, args, semaphore))
        worker.daemon = True
        worker.start()

    while semaphore.get_value() < args.num_cores:
        ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parquets", nargs="+")
    parser.add_argument("--output", default="repositories.txt")
    parser.add_argument("--min-readme-lines", type=int, default=10)
    parser.add_argument("--num-cores", type=int, default=mp.cpu_count())
    main(parser.parse_args())
