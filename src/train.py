import datetime
from transformers import AutoTokenizer
import numpy as np
import tensorflow as tf
import os
import random
import argparse
import json
from model import create_classification_model
import tensorflow_addons as tfa
from typing import Dict


def load_dataset_config(dataset):
    with open(f"../config/{dataset}.json") as file:
        config = json.load(file)
    return config


def load_model_config(model_name) -> Dict:
    with open(f"../config/model/{model_name}.json") as file:
        config = json.load(file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="csfd", type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--dropout", default=None, type=float)
    parser.add_argument("--epochs", default=None, type=float)
    parser.add_argument("--weights_file", default=None, type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    if tf.test.gpu_device_name():
      print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
      print("No GPU available")

    config = load_dataset_config(args.dataset)
    train_config = config["train"]

    model_config_ds = config["model"]
    if args.model is not None:
        model_config_ds["model_config_name"] = args.model
    if args.weights_file is not None:
        model_config_ds["weights_file"] = args.weights_file

    # merge model config from dataset and general
    model_config_general = load_model_config(model_config_ds["model_config_name"])
    model_config = {}
    for k in set(list(model_config_ds.keys()) + list(model_config_general.keys())):
        if k in model_config_ds:
            model_config[k] = model_config_ds[k]
        else:
            model_config[k] = model_config_general[k]
    config["model"] = model_config

    if args.dropout is not None:
        model_config["dropout"] = args.dropout
    if args.learning_rate is not None:
        train_config["optimizer"]["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        train_config["epochs"] = args.epochs

    max_sentence_length = model_config["max_sentence_len"]
    classes = model_config["classes"]
    model_name = model_config["pretrained_name"]

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = lambda text: _tokenizer.encode(text, padding="max_length", truncation=True, max_length=max_sentence_length)

    from data.utils import get_train_val_test_dataset
    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(config, tokenizer)

    log_dir = os.path.join("logs", "{}-{}".format(
            args.dataset,
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = create_classification_model(model_config)

    print(model.summary())

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, 'cp-{epoch:02d}.h5'),
                                           monitor="val_f1_score", save_weights_only=True, save_best_only=False)
    ]

    epochs, batch_size, buffer_size = train_config["epochs"], train_config["batch_size"], train_config["buffer_size"]

    optimizer_config = train_config["optimizer"]
    learning_rate, epsilon = optimizer_config["learning_rate"], optimizer_config["epsilon"]

    class_weight = {int(k): v for k, v in train_config["class_weight"].items()}

    optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = [
        tf.keras.metrics.CategoricalAccuracy("acc"),
        tfa.metrics.F1Score(num_classes=classes, name="f1_score", average="weighted"),
        tf.metrics.Recall(name="recall"),
        tf.metrics.Precision(name="precision")
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.summary()

    print("ARGS")
    print(args)

    print("CONFIG")
    print(config)

    print("MODEL CONFIG")
    print(model_config)

    with open(os.path.join(log_dir, "params.txt"), "w") as file:
        print("CONFIG",  file=file)
        print(f"{config}", file=file)

        print("MODEL CONFIG", file=file)
        print(f"{model_config}", file=file)

    model.fit(train_dataset.shuffle(buffer_size).batch(batch_size),
              epochs=epochs,
              validation_data=val_dataset.batch(batch_size),
              callbacks=callbacks,
              class_weight=class_weight)
