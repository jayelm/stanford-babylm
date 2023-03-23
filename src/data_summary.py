import argparse
import os

import pandas as pd

# The program counts the total number of words in the
# test_dir, which can be multiple folders

argp = argparse.ArgumentParser()

argp.add_argument(
    "--root_dir",
    default="/home/ubuntu/stanford-babylm/babylm_data/",
    help="(i.e. --root_dir=/home/ubuntu/stanford-babylm/babylm_data/)",
)

argp.add_argument(
    "--test_dir", default=["resample_50k"], help="(i.e. --test_dir 'resample_50k')"
)

argp.add_argument("--out_path", default="out.csv", help="(i.e. --out_path=out.csv)")

args = argp.parse_args()

total_word_dict = {}
for cur_dir in args.test_dir:
    total_word_count = []
    for file_name in os.listdir(os.path.join(args.root_dir, args.cur_dir)):
        num_words = 0
        if file_name.split(".")[-1] == "train":
            result = []
            with open(os.path.join(args.root_dir, args.cur_dir, file_name), "r+") as f:
                print(file_name)
                for line in f:
                    words = line.split()
                    num_words += len(words)
            print("Number of words:", num_words)
            total_word_count.append(num_words)
            total_word_dict[file_name] = num_words
    print("Total count:", sum(total_word_count))

df = pd.DataFrame.from_dict(total_word_dict, orient="index")
print(df)
df.to_csv(args.out_path)
