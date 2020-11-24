import tensorflow as tf
from transformers import TFBertModel, TFXLMRobertaModel


def create_classification_model(model_config):
    classes, max_sentence_len, dropout = model_config["classes"], model_config["max_sentence_len"], model_config["dropout"]
    model_pretrained_name = model_config["pretrained_name"]
    tokens_config = model_config["tokens"]

    pad_token_id = tokens_config["pad"]["id"]

    model: tf.keras.Model
    if model_config["model_config_name"] == "bert":
        model = TFBertModel.from_pretrained(model_pretrained_name)
    elif model_config["model_config_name"] == "xlm_roberta":
        model = TFXLMRobertaModel.from_pretrained(model_pretrained_name)
    else:
        raise ValueError()

    subword_ids = tf.keras.layers.Input(shape=(max_sentence_len,), dtype=tf.int32, name="input_ids")

    attention_masks = tf.keras.layers.Lambda(lambda x: tf.cast(x != pad_token_id, tf.int32))(subword_ids)

    subword_embeddings = model([subword_ids, attention_masks])[0]

    layer = subword_embeddings
    layer = tf.keras.layers.Flatten()(layer)
    dropout = tf.keras.layers.Dropout(rate=dropout)(layer)
    output = tf.keras.layers.Dense(units=classes, activation="softmax")(dropout)

    model = tf.keras.models.Model(inputs=subword_ids, outputs=output)

    if "weights_file" in model_config:
        model.load_weights(model_config["weights_file"])

    return model
