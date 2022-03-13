"""
"""

# flake8: noqa

import json
import os
import pickle
from itertools import chain

import nltk
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords

from aim_task.fetch_data import DATA_PATH
from aim_task.preprocess_data import (
    MAX_LENGTH,
    build_tokeniser,
    init_preprocess,
    onehot_encode_labels,
    tokenise_pad_texts,
)

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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline

labels = []


def tokeniser(string: str) -> list:
    """a function to be passed to `CountVectorizer` instead of the built-in
    types to allow for a combination of character-word n-grams

    https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
    https://stackoverflow.com/questions/5466451/how-can-i-print-literal-curly-brace-characters-in-a-string-and-also-use-format

    Parameters:
    -----------
    string: the string to be tokenised

    Returns:
    --------
    out: list
        a sorted list of the tokens found in tokenising the `string`
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


def build_naive_bayes_model(*args, **kwargs) -> Pipeline:
    """builds and returns an sklearn pipeline, where the stages are a
    `CountVectorizer` followed by a `MultinomialNB` model. The `` pairs
    are expected to be

    Parameters:
    -----------
    kwargs: the keyword arguments for `CountVectorizer` and possibly a keyword
    argument `alpha` for `MultinomialNB`

    Returns:
    --------
    out: sklearn.pipeline.Pipeline
        the pipeline ready to train
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

    labels = y_train.cat.categories.values.tolist()
    labels_path = os.path.join(DATA_PATH, "raw/labels.json")
    json.dump(labels, open(labels_path, "w"))

    model_classic = build_naive_bayes_model()
    _ = model_classic.fit(X_train, y_train)
    y_pred = model_classic.predict(X_test)
    clf_report = classification_report(y_test, y_pred, labels=y_train.unique())
    print(clf_report)

    model_classic_path = os.path.join(DATA_PATH, "model/model_classic.pkl")
    pickle.dump(model_classic, open(model_classic_path, "wb"))

    corpus = np.r_[X_train.values, X_test.values]
    tknsr = build_tokeniser(corpus)

    tokeniser_json = tknsr.to_json()
    tokeniser_path = os.path.join(DATA_PATH, "models/tokeniser.json")
    json.dump(open(tokeniser_path, "w"))

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
