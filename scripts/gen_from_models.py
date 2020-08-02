from pathlib import Path
import argparse
import subprocess


def argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="synthesis")
    parser.add_argument("--model_path", type=Path, default="./results/synthesis/")
    parser.add_argument("--save_path", type=Path, default="./results/gen_samples")
    parser.add_argument("--seq_len", type=int, default=400)
    parser.add_argument("--bias", type=float, default=10.0, help="bias")
    parser.add_argument("--char_seq", type=str, default="This is real handwriting")
    parser.add_argument("--text_req", action="store_true")
    parser.add_argument("--prime", action="store_true")
    parser.add_argument("--is_map", action="store_true")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--file_path", type=str, help="./app/")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = argparser()
    model_list = list(args.model_path.glob("*.pt"))
    cmd = "python generate.py --model synthesis --model_path {} --save_path {} --bias {args.bias} --char_seq '{args.char_seq}' --prime --text_req --seed {args.seed}"
    for i, model_path in enumerate(model_list):
        print(model_path, args.save_path / (model_path.name))
        subprocess.getstatusoutput(
            cmd.format(model_path, args.save_path / (model_path.name), args=args)
        )

