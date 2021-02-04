import os
import random
import ijson

from src.data.utils import write_into_file, train_val_test_split


def preprocess(dir_path, val_split, test_split, max_sentences=100000):
    dataset = []
    with open(os.path.join(dir_path, "data.json"), "r") as file:
        data = ijson.parse(file)

        i = 0
        for p, e, v in data:
            if i >= max_sentences:
                break

            if p == "item.text":
                dataset.append((v, 0))

                i += 1
    print(dataset[:10])
    random.shuffle(dataset)

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, val_split, test_split)

    write_into_file(dataset, out_path=os.path.join(dir_path, "data_full.csv"))
    write_into_file(train_dataset, out_path=os.path.join(dir_path, "data_train.csv"))
    write_into_file(val_dataset, out_path=os.path.join(dir_path, "data_val.csv"))
    write_into_file(test_dataset, out_path=os.path.join(dir_path, "data_test.csv"))