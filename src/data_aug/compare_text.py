import argparse
import os

# The program compares text and count for
# repeated sentences in the augmented file
# vs the original training file

argp = argparse.ArgumentParser()

argp.add_argument(
    "--root_dir",
    default="/home/ubuntu/stanford-babylm/babylm_data/",
    help="(i.e. --root_dir=/home/ubuntu/stanford-babylm/babylm_data/)",
)
argp.add_argument(
    "--from_dir", default="babylm_10M", help="(i.e. --from_dir=babylm_10M)"
)
argp.add_argument(
    "--test_dir", default="resample_1k", help="(i.e. --test_dir=resample_1k)"
)

args = argp.parse_args()

total = []
for doc_name in os.listdir(os.path.join(args.root_dir, args.test_dir)):
    in_count = 0
    total_line = 0
    if doc_name.split(".")[-1] == "train":
        orig_path = os.path.join(args.root_dir, args.from_dir, doc_name[12:])
        orig_file = open(orig_path, "r").readlines()
        with open(os.path.join(args.root_dir, args.test_dir, doc_name), "r") as f:
            print(doc_name)
            for line in f:
                if len(line.strip()) != 0:
                    total_line += 1
                if line in orig_file:
                    in_count += 1
                    print(line)
        print("Total number of lines:", total_line)
        print("Total lines in orig:", in_count)
        print("Percentage:", in_count / total_line * 100)
        # Igorning the files that has inconsistent
        # notation on punctuations
        if in_count != 0:
            total.append(in_count / total_line * 100)

print("Total_average:", sum(total) / len(total))
