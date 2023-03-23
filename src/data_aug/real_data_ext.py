import argparse
import os

import pandas as pd

# The program takes files from from_dir and
# extract the same number of words required by
# the custom count_dir

argp = argparse.ArgumentParser()

argp.add_argument(
    "--root_dir",
    default="/home/ubuntu/stanford-babylm/babylm_data/",
    help="(i.e. --root_dir=/home/ubuntu/stanford-babylm/babylm_data/)",
)
argp.add_argument(
    "--from_dir", default="babylm_100M", help="(i.e. --from_dir=babylm_100M)"
)
argp.add_argument("--to_dir", default="real_50k", help="(i.e. --to_dir=real_50k)")
argp.add_argument("--out_path", default="out.csv", help="(i.e. --out_path=out.csv)")

args = argp.parse_args()

# TODO: customize count of words for each experiment
count_dir = {
    "aochildes.train": 214110,
    "bnc_spoken.train": 413218,
    "cbt.train": 288992,
    "children_stories.train": 18820,
    "gutenberg.train": 271561,
    "open_subtitles.train": 231142,
    "qed.train": 332051,
    "simple_wikipedia.train": 1123420,
    "switchboard.train": 88983,
    "wikipedia.train": 398946,
}

from_file = os.path.join(args.root_dir, args.from_dir)
to_file = os.path.join(args.root_dir, args.to_dir)

if not os.path.exists(to_file):
    os.mkdir(to_file)

# Since babylm_100M does not cover (at least sequentially)
# the content from babylm_10M, will take the first sentences
# that are the same from the GECA files
total_word_dict = {}
total_word_count = []
for file_name in os.listdir(from_file):
    num_words = 0
    new_file = []
    if file_name.split(".")[-1] == "train" and file_name in count_dir:
        with open(os.path.join(from_file, file_name), "r") as f:
            print(file_name)
            for line in f:
                words = line.split()
                num_words += len(words)
                new_file.append(line)
                if num_words >= count_dir[file_name]:
                    file1 = open(
                        os.path.join(to_file, args.to_dir + "_" + file_name), "w"
                    )
                    file1.writelines(new_file)
                    file1.close()
                    break
        print("Number of words:", num_words)
        total_word_count.append(num_words)
        total_word_dict[file_name] = num_words
print("Total count:", sum(total_word_count))

df = pd.DataFrame.from_dict(total_word_dict, orient="index")
print(df)
df.to_csv(os.path.join(args.root_dir, args.to_dir, args.out_path))
