import os
import random
from .utils import write_into_file, train_val_test_split


def preprocess(dir_path, val_split, test_split):
    positive = []
    with open(os.path.join(dir_path, "positive.txt"), "r") as file:
        for line in file:
            positive.append((line, 2))

    neutral = []
    with open(os.path.join(dir_path, "neutral.txt"), "r") as file:
        for line in file:
            neutral.append((line, 1))

    negative = []
    with open(os.path.join(dir_path, "negative.txt"), "r") as file:
        for line in file:
            negative.append((line, 0))

    dataset = positive + neutral + negative

    del positive
    del neutral
    del negative

    random.shuffle(dataset)

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, val_split, test_split)

    write_into_file(dataset, out_path=os.path.join(dir_path, "data_full.csv"))
    write_into_file(train_dataset, out_path=os.path.join(dir_path, "data_train.csv"))
    write_into_file(val_dataset, out_path=os.path.join(dir_path, "data_val.csv"))
    write_into_file(test_dataset, out_path=os.path.join(dir_path, "data_test.csv"))

