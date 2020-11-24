from sklearn.utils import class_weight
import numpy as np
import csv
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_full_path", required=True, type=str)
    parser.add_argument("--config_path", required=True, type=str)
    args = parser.parse_args()

    labels = []
    with open(args.dataset_full_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text, label = row["text"], int(row["label"])
            labels.append(label)

    unique_labels = np.unique(labels)

    class_weight = class_weight.compute_class_weight("balanced", unique_labels, labels)
    class_weight = np.around(class_weight, decimals=5)

    print("CLASS WEIGHT")
    print(class_weight)

    with open(args.config_path, "r") as file:
        config = json.load(file)

    with open(args.config_path, "w") as file:
        class_weight_config = {str(label): weight for label, weight in zip(unique_labels, class_weight)}
        config["train"]["class_weight"] = class_weight_config

        json.dump(config, file, indent=2, sort_keys=True)
