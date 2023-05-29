### ASSUME DATA IS DOWNLOADED in babylm_data/babylm_test/
import csv
from datasets import Dataset as HFDataset
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('{}/tree_projection_src/'.format(os.getcwd()))
csv.field_size_limit(sys.maxsize)

DATA_DIR=os.getcwd() + '/src/data_utils'
MIN_LENGTH = 3
MAX_LENGTH = 15
def read_data(splits, print_histogram = False):
    in_sentences = []
    string_lengths = []

    index_map = {split: [] for split in splits}

    for file in os.listdir(DATA_DIR + "/babylm_data/babylm_test"):
        with open("{}/babylm_data/babylm_test/{}".format(DATA_DIR,file), "r") as reader:
            d = csv.reader(reader)
            for line in d:
                split_line = [word.split() for word in line]
                string = ' '.join([word for sub_lst in split_line for word in sub_lst])

                string_length = len(string.split())
                string_lengths.append(string_length)

                if string_length > MIN_LENGTH:
                    if string_length > MAX_LENGTH:
                        string = ' '.join([word for word in string.split()[0:MAX_LENGTH]])
                    in_sentences.append(string)

    if print_histogram:
        # Define the maximum bin width
        max_bin_width = 5

        # Calculate the number of bins using the maximum bin width
        num_bins = ((max(string_lengths) - min(string_lengths)) // max_bin_width)  

        # Create a histogram of the line lengths
        plt.hist(string_lengths, bins=num_bins)
        plt.xlim([0,50])
        plt.axvline(x = MIN_LENGTH, color = 'r', label = 'axvline - full height')
        plt.axvline(x = MAX_LENGTH, color = 'r', label = 'axvline - full height')
        plt.xlabel('Line Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Line Lengths')
        plt.savefig('string_lengths_babylm.png')
        percent_under_MAX_LENGTH = sum(MIN_LENGTH <= string <= MAX_LENGTH for string in string_lengths) / len(string_lengths)

    return in_sentences


def build_datasets_babylm():
    splits = ["train", "val", "test", "gen"]
    in_sentences = read_data(splits)
    print("num examples: {}".format(len(in_sentences)))

    return in_sentences
