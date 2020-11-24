import argparse
import random
import data.review_dataset
import data.facebook_dataset
import data.all_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="mall", type=str)
    args = parser.parse_args()

    random.seed(args.seed)

    dir_path = f"../data/{args.dataset}"

    if args.dataset == "mall" or args.dataset == "csfd":
        data.review_dataset.preprocess(dir_path, val_split=0.15, test_split=0.15)
    elif args.dataset == "facebook":
        data.facebook_dataset.preprocess(dir_path, val_split=0.15, test_split=0.15)
    elif args.dataset == "all":
        in_csv_paths = [
            "../data/facebook/data_full.csv",
            "../data/csfd/data_full.csv",
            "../data/mall/data_full.csv"
        ]

        data.all_dataset.preprocess(in_csv_paths, dir_path, val_split=0.15, test_split=0.15)
    elif args.dataset == "handpicked_test":
        data.review_dataset.preprocess(dir_path, val_split=0, test_split=1)

