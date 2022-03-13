"""
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from aim_task.fetch_data import load_full_dataset, DATA_PATH
from sklearn.model_selection import train_test_split
from typing import Tuple, Union

# from nltk.tokenize import TweetTokenizer

REGEX_HANDLER = r"(?u)@\w+"
REGEX_HYPERLINK = r"(?u)https?://\w+"
REGEX_HASHTAG = r"(?u)#(\w+)"
# to capture letters that repeat over 3 times
REGEX_CHAR_3_PLUS = r"(?u)(?=(\w))\1{3,}"

# to capture non-arabic, & non-farsi unicodes
REGEX_NOT_ARA_IRA = r"(?u)\b[^\u0600-\u06ff\ufe70-\ufefc_]+\b"

# to capture most common arabic glyphs
REGEX_RANGE_ARA = r"\ufe70-\ufefc\u0621-\u063a\u0640-\u0652\u060c\u061f"
REGEX_NOT_ARA = r"(?u)\b[^" + REGEX_RANGE_ARA + "]+\b"

# converting farsi glyphs into arabic
REGEX_IRA2ARA = {
    r"\u06fd": "\u0621",
    r"[\u0676\u0677]": "\u0624",
    r"[\u0678\u06d3]": "\u0626",
    r"[\u0622\u0623\u0625\u0671-\u0675]": "\u0627",
    r"[\u067b\u067e\u0680]": "\u0628",
    r"[\u067a\u067c\u067d\u067f]": "\u062a",
    r"[\u0686\u0687]": "\u062c",
    r"[\u0688-\u068b\u068d]": "\u062f",
    r"[\u068c\u068e-\u0690]": "\u0630",
    r"[\u066b\u0691-\u0695]": "\u0631",
    r"[\u0696-\u0699]": "\u0632",
    r"[\u069a\u069b]": "\u0633",
    r"\u069c": "\u0634",
    r"\u060f": "\u0639",
    r"\u06d4": "\u0640",
    r"[\u06a4\u06a5]": "\u0641",
    r"\u06a8\u06a6": "\u0642",
    r"[\u063b\u063c\u06a9-\u06b4]": "\u0643",
    r"[\u06b5-\u06b8]": "\u0644",
    r"\u06fe": "\u0645",
    r"[\u06b9-\u06bd]": "\u0646",
    r"[\u0629\u06be\u06c0-\u06c3\u06ff]": "\u0647",
    r"[\u06c4-\u06cb\u06cf]": "\u0648",
    r"[\u0620\u063d-\u063f\u0649\u06cc-\u06ce\u06d0-\u06d2\u06e6\u06e7]": "\u064a",
    r"[\u06d9\ufef5\ufef7\ufef9]": "\ufefb",
}

# FIXME set this by EDA
MAX_LENGTH = 1024


def store_validation_set(
    df: pd.DataFrame, set_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """stores a copy of validation set on disk

    FIXME complete pydoc
    """
    y = df["dialect"]
    train_df, valid_df = train_test_split(
        df, test_size=set_size, random_state=42, stratify=y
    )
    valid_set_path = os.path.join(DATA_PATH, "raw/valid.pkl")
    pickle.dump(valid_df, open(valid_set_path, "wb"))
    return train_df, valid_df


def regex_substitute(
    series: Union[list, np.ndarray, pd.Series], regex_holder_dict: dict
) -> pd.Series:
    """given an array-like, & a dictionary of RegExp:`placeholder`, the series
    is processed to replace each RegExp with the corresponding `placeholder`

    Parameters:
    -----------

    Returns:
    --------

    Raises:
    -------

    TODO complete the pydoc
    """
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series, dtype=str)
        except:
            raise ValueError("Invalid series: not array-like")
    for regex, holder in regex_holder_dict.items():
        series = series.apply(lambda s: re.sub(regex, holder, s))
    return series


def preprocess(data: Union[list, np.ndarray, pd.Series, pd.DataFrame]) -> pd.Series:
    """The actual preprocessing subroutine

    TODO complete pydoc

    """
    if isinstance(data, pd.DataFrame):
        if "tweets" not in data.columns:
            raise Exception("FIXME write a proper exception")
        # extract data as series
        data = data["tweets"]
    elif isinstance(data, list) or isinstance(data, np.ndarray):
        data = pd.Series(data, dtype=str)
    elif not isinstance(data, pd.Series):
        raise Exception("FIXME write a proper exception")

    # removing twitter specific features
    regex_twitter = {
        REGEX_HANDLER: r" ",
        REGEX_HYPERLINK: r" ",
        r"\d+": r" ",
        REGEX_HASHTAG: r"\1",
        r"_": r" ",
    }
    data = regex_substitute(data, regex_twitter)

    # normalising the characters
    data = regex_substitute(data, REGEX_IRA2ARA)

    # removing any non-useful tokens
    extra_tokens = {
        REGEX_NOT_ARA: r" ",
        r"(?u)[\d\s_]+": r" ",
        REGEX_CHAR_3_PLUS: r"\1\1",
    }
    ara_ira = regex_substitute(data, extra_tokens)
    data = ara_ira.str.strip()

    # FIXME could use lemmatisation?

    return data


def init_preprocess(
    test_size: float = 0.1,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """initialisation point for the training of the model

    TODO complete the pydoc
    """
    full_df = load_full_dataset()
    # keep a validation set aside
    df, _ = store_validation_set(full_df)

    df = preprocess(df.copy())

    # split data
    X = df["tweets"]
    y = df["dialect"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # TODO do more preprocess

    return X_train, X_test, y_train, y_test


def build_tokeniser(
    corpus: Union[list, np.ndarray, pd.Series]
) -> tf.keras.preprocessing.text.Tokenizer:
    """instantiate and fits a `tensorflow.keras.preprocessing.Tokenizer` object
    on the given corpus, and returns the tokeniser

    Parameters:
    -----------

    TODO complete pydoc
    """
    tokeniser = tf.keras.preprocessing.text.Tokenizer()
    _ = tokeniser.fit_on_texts(corpus)
    return tokeniser


def tokenise_pad_texts(
    corpus: Union[list, np.ndarray, pd.Series],
    tokeniser: tf.keras.preprocessing.text.Tokenizer = None,
    maxlen: int = MAX_LENGTH,
    padding: str = "post",
):
    """tokenises the documents in `corpus`, then pads the documents to the
    `maxlen` length

    Parameters:
    -----------

    TODO complete pydoc
    """
    if not tokeniser:
        tokeniser = build_tokeniser(corpus)
    elif not isinstance(tokeniser, tf.keras.preprocessing.text.Tokenizer):
        raise ValueError(
            "The tokeniser must be of type tensorflow.keras.preprocessing.Tokenizer"
        )
    tokens = tokeniser.texts_to_sequences(corpus)
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(
        tokens, maxlen=maxlen, padding=padding
    )
    return padded_tokens


def onehot_encode_labels(labels: Union[list, np.ndarray, pd.Series]) -> np.ndarray:
    """performs one hot encoding on labels

    Parameters:
    -----------

    TODO complete pydoc
    """
    labels = pd.Series(labels, dtype="category")
    return tf.keras.utils.to_categorical(labels.cat.codes)
