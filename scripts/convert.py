from pathlib import Path
from argparse import ArgumentParser

from DynamicTopology.io import read_xml, write_jsonl


def main(input_path: Path, output_path: Path):
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not output_path.exists() and not output_path.suffix != "":
        output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        if not output_path.is_dir():
            output_path = output_path.parent

        for path in input_path.iterdir():
            if path.is_dir() or path.suffix != ".xml":
                continue
            terms = read_xml(path)
            print(f"writing to {output_path / path.name}")
            write_jsonl((output_path / path.name).with_suffix(".jsonl"), terms)
    else:
        print(f"reading from {input_path}")
        terms = read_xml(input_path)
        if output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"writing to {output_path / input_path.name}")
            write_jsonl((output_path / input_path.name).with_suffix(".jsonl"), terms)
        else:
            print(f"writing to {output_path}")
            write_jsonl((output_path).with_suffix(".jsonl"), terms)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    main(args.input, args.output)
