import os
import random

from .utils import write_into_file, train_val_test_split


def preprocess(dir_path, val_split, test_split):
    dataset = []
    with open(os.path.join(dir_path, "gold-posts.txt"), "r") as text_file, open(os.path.join(dir_path, "gold-labels.txt"), "r") as label_file:
        for line, label in zip(text_file, label_file):
            line = line.rstrip()
            label = label.rstrip()

            if label == "n":
                label_nr = 0
            elif label == "0":
                label_nr = 1
            elif label == "p":
                label_nr = 2
            else:
                continue

            dataset.append((line, label_nr))

    random.shuffle(dataset)

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, val_split, test_split)

    write_into_file(dataset, out_path=os.path.join(dir_path, "data_full.csv"))
    write_into_file(train_dataset, out_path=os.path.join(dir_path, "data_train.csv"))
    write_into_file(val_dataset, out_path=os.path.join(dir_path, "data_val.csv"))
    write_into_file(test_dataset, out_path=os.path.join(dir_path, "data_test.csv"))