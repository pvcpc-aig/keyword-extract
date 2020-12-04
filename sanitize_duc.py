import os
import shutil
from pathlib import Path


def get_file_text(file):
    content = open(file, mode="r", encoding="latin").readlines()
    start = 0
    end = len(content)
    for i, e in enumerate(content):
        if "<TEXT>" == e.strip():
            start = i + 1
        if "</TEXT>" == e.strip():
            end = i
            break
    return ''.join(content[start:end])


if __name__ == "__main__":
    # Copy all of the samples to a single directory with only the
    # content in the <TEXT></TEXT> tag.
    dataset = Path("data/duc2004/dataset/")
    raw_out = Path("data/duc2004/raw")

    for batch in dataset.iterdir():
        if not batch.is_dir():
            continue
        # remove the 't' at the end of the batch directory name
        raw_batch = raw_out.joinpath(batch.name[:-1].upper())
        if not raw_batch.exists():
            raw_batch.mkdir()

        for file in batch.iterdir():
            if file.name.endswith("DS_Store") or file.name.startswith("textrank"):
                continue
            with raw_batch.joinpath(file.name).open(mode="w") as fout:
                fout.write(get_file_text(file))
        
        print(f"wrote {raw_batch}")