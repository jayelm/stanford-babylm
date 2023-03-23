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
    "--from_dir", default="augmented_50k", help="(i.e. --from_dir=augmented_50k)"
)
argp.add_argument("--to_dir", default="GECA_12k", help="(i.e. --to_dir=GECA_12k)")
argp.add_argument(
    "--max_sent",
    default=12000,
    type=int,
    help="(maximum number of sentences in each file, i.e. --max_sent=12000)",
)
argp.add_argument("--out_path", default="out.csv", help="(i.e. --out_path=out.csv)")

args = argp.parse_args()

from_file = os.path.join(args.root_dir, args.from_dir)
to_file = os.path.join(args.root_dir, args.to_dir)

if not os.path.exists(to_file):
    os.mkdir(to_file)

total_word_dict = {}
total_word_count = []
for file_name in os.listdir(from_file):
    num_sent = 0
    num_words = 0
    new_file = []
    if file_name.split(".")[-1] == "train":
        with open(os.path.join(from_file, file_name), "r") as f:
            print(file_name)
            for line in f:
                num_sent += 1
                words = line.split()
                num_words += len(words)
                new_file.append(line)
                if num_sent >= args.max_sent:
                    break
            new_name = args.to_dir + "_" + file_name[10:]
            file1 = open(os.path.join(to_file, new_name), "w")
            file1.writelines(new_file)
            file1.close()
        print("Number of words:", num_words)
        total_word_count.append(num_words)
        total_word_dict[new_name] = num_words
print("Total count:", sum(total_word_count))

df = pd.DataFrame.from_dict(total_word_dict, orient="index")
print(df)
df.to_csv(os.path.join(args.root_dir, args.to_dir, args.out_path))
