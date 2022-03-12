"""
"""

import json
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from aim_task.fetch_data import DATA_PATH
from itertools import chain
from nltk.corpus import stopwords

try:
    STOP_WORDS_BASIC = stopwords.words("arabic")
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS_BASIC = stopwords.words("arabic")
finally:
    pass
    # FIXME should this be here?
    # STOP_WORDS_BASIC = stopwords.words("arabic")

from nltk.tokenize import RegexpTokenizer
from preprocess_data import (
    build_tokeniser,
    init_preprocess,
    MAX_LENGTH,
    onehot_encode_labels,
    tokenise_pad_texts,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


def tokeniser(string: str) -> list:
    """https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists

    Parameters:
    -----------

    Returns:
    --------

    TODO complete pydoc
    """
    regex_c37 = r"(?u)(?=(\w{3}))" + "".join(
        [f"(?=(\\w{{,{i}}}))" for i in range(4, 8)]
    )
    regex_w26 = r"(?u)(?=(?!\W)(?=((?:\W*\b\w+\b){2}))" + "".join(
        [f"(?!\\W)(?=((?:\\W*\\b\w+\\b){{,{i}}}))" for i in range(3, 7)]
    )
    token_c37 = RegexpTokenizer(regex_c37).tokenize(string)
    token_w26 = RegexpTokenizer(regex_w26).tokenize(string)
    tokens = set(chain(*(token_c37 + token_w26)))
    return sorted(tokens)


def build_naive_bayes_model(*args, **kwargs):
    """builds and returns an sklearn pipeline


    TODO complete the pydoc
    """
    if "stop_words" not in kwargs:
        kwargs["stop_words"] = STOP_WORDS_BASIC
    model = MultinomialNB(alpha=kwargs.get("alpha", 1))
    if "alpha" in kwargs:
        del kwargs["alpha"]
    vectoriser = CountVectorizer(**kwargs)

    return make_pipeline(vectoriser, model)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = init_preprocess()
    scoring = [
        "neg_log_loss",
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
    ]

    model_classic = build_naive_bayes_model()
    _ = model_classic.fit(X_train, y_train)
    y_pred = model_classic.predict(X_test)
    clf_report = classification_report(y_test, y_pred, labels=y_train.unique())
    print(clf_report)

    model_classic_path = os.path.join(DATA_PATH, "model/model_classic.pkl")
    pickle.dump(model_classic, open(model_classic_path, "wb"))

    corpus = np.r_[X_train.values, X_test.values]
    tknsr = build_tokeniser(corpus)
    corpus = tokenise_pad_texts(corpus)
    labels = np.r_[y_train, y_test]
    labels = onehot_encode_labels(labels)

    model_rnn = tf.keras.models.Sequential(name="model_rnn")
    model_rnn.add(
        tf.keras.layers.Embedding(
            len(tknsr.word_counts) + 1,
            128,
            embedding_initializer=tf.keras.initializers.RandomUniform(),
            input_length=MAX_LENGTH,
        )
    )
    model_rnn.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.25)))
    model_rnn.add(tf.kera.layers.Dropout(0.5))
    model_rnn.add(tf.keras.Dense(64, activation="relu"))
    model_rnn.add(tf.keras.Dense(labels.shape[1], activation="softmax"))

    model_rnn.compile(
        "adam", "categorical_crossentropy", metrics=["categorical_accuracy"]
    )

    _ = model_rnn.fit(
        corpus, labels, batch_size=32, epochs=2, verbose=0, validation_size=0.2
    )

    model_rnn_path = os.path.join(DATA_PATH, "models/model_rnn.json")
    model_rnn_json = model_rnn.to_json()
    json.dump(json.loads(model_rnn_json), open(model_rnn_path, "w"))
    model_rnn.save_weights(os.path.join(DATA_PATH, "weights/model_rnn_weights.h5"))
