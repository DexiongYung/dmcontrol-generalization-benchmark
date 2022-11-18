import os
import numpy as np
import pandas as pd
import argparse


def main(args):
    dir_path_list = args.dir_path.split(",")
    data = []
    for dir_path in dir_path_list:
        values_list = []
        for seed in range(args.max_seed + 1):
            csv_fp = os.path.join(dir_path, f"seed_{seed}", args.csv_file_name)

            if args.skip_not_found and not os.path.exists(csv_fp):
                print(f"Path: {csv_fp}, not found")
                continue

            df = pd.read_csv(csv_fp)

            if np.int64 == df[args.column_name].dtype:
                column_value = int(args.column_value)
            else:
                column_value = str(args.column_value)

            values_list.append(
                df.loc[df[args.column_name] == column_value][args.row_value].iloc[0]
            )

        data.append([dir_path, np.mean(values_list), np.std(values_list)])

    pd.DataFrame(data, columns=["path", "mu", "std"]).to_csv(args.output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", required=True, type=str)
    parser.add_argument("--max_seed", default=1, type=int)
    parser.add_argument("--csv_file_name", default="eval.csv", type=str)
    parser.add_argument("--output_file_name", type=str, required=True)
    parser.add_argument("--column_name", type=str, default="step")
    parser.add_argument("--column_value", default=500000)
    parser.add_argument("--row_value", required=True)
    parser.add_argument("--skip_not_found", default=False)
    args = parser.parse_args()
    main(args)
