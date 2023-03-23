"""
Sample generation from model files defined in the `exp` folder.
"""

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

argp = argparse.ArgumentParser()
argp.add_argument(
    "--model_path",
    default="debug-gpt2-xsmall-babylm_10M",
    help="(i.e. --model_path=debug-gpt2-xsmall-babylm_10M)",
)
argp.add_argument(
    "--root_dir",
    default="/home/ubuntu/stanford-babylm/babylm_data/babylm_10M",
    help="(i.e. --root_dir=/home/ubuntu/stanford-babylm/babylm_data/babylm_10M)",
)
argp.add_argument(
    "--final_dir",
    default="/home/ubuntu/stanford-babylm/babylm_data/",
    help="(i.e. --final_dir=/home/ubuntu/stanford-babylm/babylm_data/)",
)
argp.add_argument(
    "--to_dir",
    default="resample_50k",
    help="(i.e. --to_dir=resample_50k)",
)

args = argp.parse_args()


# Load the trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "exp/" + args.model_path + "/" + args.model_path + "-run-42"
)  # adjust suffix if the seed is changed
tokenizer = AutoTokenizer.from_pretrained(
    "exp/" + args.model_path + "/" + args.model_path + "-run-42"
)  # adjust suffix if the seed is changed

# Set pad_token_id to eos_token_id because GPT2 does not have a PAD token
model.generation_config.pad_token_id = model.generation_config.eos_token_id


to_file = os.path.join(args.final_dir, args.to_dir)

if not os.path.exists(to_file):
    os.mkdir(to_file)


count_dir = {
    "aochildes.train": 429271,
    "bnc_spoken.train": 823446,
    "cbt.train": 288992,
    "children_stories.train": 18820,
    "gutenberg.train": 313446,
    "open_subtitles.train": 461994,
    "qed.train": 665159,
    "simple_wikipedia.train": 2172983,
    "switchboard.train": 88983,
    "wikipedia.train": 398946,
}

total_word_count = []
for file_name in os.listdir(args.root_dir):
    num_words = 0
    if file_name.split(".")[-1] == "train" and (file_name in count_dir):
        print("-" * 100)
        print("Filename:", file_name)
        total_file = []
        with open(os.path.join(args.root_dir, file_name), "r") as f:
            new_name = args.to_dir + "_" + file_name
            sent_count = 0
            for line in f:
                if sent_count <= 1000:
                    sent_count += 1
                    continue
                words = line.split()
                if len(words) == 0:
                    continue
                sent_count += 1
                first_word = words[0]
                input_ids = tokenizer(first_word, return_tensors="pt").input_ids
                sample_output = model.generate(
                    input_ids,
                    do_sample=True,
                    max_length=min(len(words), 512),
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
                output = tokenizer.decode(sample_output[0], skip_special_tokens=True)
                # print(output)
                if (len(output.split()) == 0) or (output == "\n"):
                    continue
                num_words += len(output.split())
                total_file.append(output + "\n")

                # Save for every 100 sentences generated
                if sent_count % 100 == 0:
                    file1 = open(os.path.join(to_file, new_name), "w")
                    file1.writelines(total_file)
                    file1.close()
                    print("-" * 100)
                    print("Save path:", os.path.join(to_file, new_name))

                if num_words >= count_dir[file_name]:
                    file1 = open(os.path.join(to_file, new_name), "w")
                    file1.writelines(total_file)
                    file1.close()
                    print("-" * 100)
                    print("Finished path:", os.path.join(to_file, new_name))
                    break

            print("Number of words:", num_words)
            total_word_count.append(num_words)
print("Total count:", sum(total_word_count))
