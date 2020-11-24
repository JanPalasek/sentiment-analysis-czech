import csv
import tensorflow as tf
from transformers import AutoTokenizer
import os


def _get_data(data_path, tokenizer, classes):
    with open(data_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            text, label = row["text"], int(row["label"])

            text_tokens = tokenizer(text)
            label_res = tf.one_hot(label, depth=classes, dtype=tf.float32)

            yield text_tokens, label_res


def get_train_val_test_dataset(config, tokenizer):
    dir_path = config["path"]
    max_sentence_len = config["model"]["max_sentence_len"]
    classes = config["model"]["classes"]
    data_function = lambda: _get_data(data_path=os.path.join(dir_path, "data_train.csv"), tokenizer=tokenizer, classes=classes)
    train_dataset = tf.data.Dataset.from_generator(data_function, (tf.int32, tf.float32),
                                                   output_shapes=((max_sentence_len,), (classes,)))
    val_function = lambda: _get_data(data_path=os.path.join(dir_path, "data_val.csv"), tokenizer=tokenizer, classes=classes)
    val_dataset = tf.data.Dataset.from_generator(val_function, (tf.int32, tf.float32),
                                                 output_shapes=((max_sentence_len,), (classes,)))
    test_function = lambda: _get_data(data_path=os.path.join(dir_path, "data_test.csv"), tokenizer=tokenizer, classes=classes)
    test_dataset = tf.data.Dataset.from_generator(test_function, (tf.int32, tf.float32),
                                                  output_shapes=((max_sentence_len,), (classes,)))

    return train_dataset, val_dataset, test_dataset


def get_test_dataset(config, tokenizer):
    dir_path = config["path"]
    max_sentence_len = config["model"]["max_sentence_len"]
    classes = config["model"]["classes"]
    test_function = lambda: _get_data(data_path=os.path.join(dir_path, "data_test.csv"), tokenizer=tokenizer,
                                      classes=classes)
    test_dataset = tf.data.Dataset.from_generator(test_function, (tf.int32, tf.float32),
                                                  output_shapes=((max_sentence_len,), (classes,)))

    return test_dataset


def get_tokenizer_function(config):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer_fnc = lambda text: tokenizer.encode(text, padding="max_length", truncation=True, max_length=config.model.max_sentence_len)
    return tokenizer_fnc


def train_val_test_split(dataset, val_split: float, test_split: float):
    dataset_len = len(dataset)
    train_split = 1 - val_split - test_split

    train_size = int(train_split * dataset_len)
    val_size = int(val_split * dataset_len)
    test_size = int(test_split * dataset_len)

    train_start, train_end = 0, train_size
    val_start = train_size
    val_end = train_size + val_size

    test_start, test_end = val_end, val_end + test_size

    return dataset[train_start: train_end], dataset[val_start: val_end], dataset[test_start: test_end]


def write_into_file(dataset, out_path):
    """
    :param dataset: Expect to be list of tuples (string, int), where string is text and int is label number.
    :param out_path:
    :return:
    """
    with open(out_path, "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["text", "label"])
        writer.writeheader()

        for line, label in dataset:
            writer.writerow({"text": line, "label": label})
