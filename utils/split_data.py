"""
Example of usage:
    python3 split_data.py -p data/positive_reviews.json
                          -n data/negative_reviews.json
                          -tr data_splitted/laroseda_train.json
                          -te data_splitted/laroseda_test.json
"""

import json
import os

import argparse
import random

from sklearn.model_selection import train_test_split


def read_and_split(f, test_split=1500, sentiment="positive"):
    # Read the data
    data = json.load(f)["reviews"]
    
    # Split in train / test
    train_data, test_data = train_test_split(data, test_size=test_split, random_state=1)

    print("Train size: %d %s samples." % (len(train_data), sentiment))
    print("Test size: %d %s samples." % (len(test_data), sentiment))

    return train_data, test_data


def merge_and_shuffle(data_1, data_2):
    # Merge
    data = data_1 + data_2

    # Randomize
    random.shuffle(data)

    return data


if __name__ == "__main__":
    # Add and parse arguments
    parser = argparse.ArgumentParser(description = "Data split arguments.")
    parser.add_argument(
        "-p", 
        "--pos_file", 
        type=argparse.FileType('r', encoding="utf-8"), 
        required=True, 
        help="Path to the positive reviews file."
    )
    parser.add_argument(
        "-n",
        "--neg_file",
        type=argparse.FileType('r', encoding="utf-8"),
        required=True,
        help="Path to the negative reviews file."
    )
    parser.add_argument(
        "-tr",
        "--train_file",
        type=argparse.FileType('w', encoding="utf-8"),
        required=True,
        help="Path to the train splitted file."
    )
    parser.add_argument(
        "-te",
        "--test_file",
        type=argparse.FileType('w', encoding="utf-8"),
        required=True,
        help="Path to the test splitted reviews file."
    )
    args = parser.parse_args()


    # Read and split the positive and negative samples
    pos_train, pos_test = read_and_split(args.pos_file, sentiment="positive")
    neg_train, neg_test = read_and_split(args.neg_file, sentiment="negative")

    # Merge and shuffle
    train_data = merge_and_shuffle(pos_train, neg_train)
    test_data = merge_and_shuffle(pos_test, neg_test)

    print("Total train: %d." % len(train_data))
    print("Total test: %d." % len(test_data))

    # Initial format
    train_dict = {"reviews": train_data}
    test_dict = {"reviews": test_data}

    # Write in files
    json.dump(train_dict, args.train_file, indent=4)
    json.dump(test_dict, args.test_file, indent=4)

