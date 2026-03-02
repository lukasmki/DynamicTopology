from argparse import ArgumentParser
from pathlib import Path

from ase import Atoms, io
from molify import ase2networkx, networkx2ase


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=False)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    if args.output is not None:
        output_dir = Path(args.output).resolve()
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = input_path

    for path in input_path.iterdir():
        if path.suffix != ".xyz":
            continue
        atoms: list[Atoms] = io.read(path, index=":")
        graphs = [ase2networkx(a, False) for a in atoms]
        new_atoms = [networkx2ase(g) for g in graphs]
        io.write(output_dir / path.name, new_atoms, format="extxyz")


if __name__ == "__main__":
    main()
