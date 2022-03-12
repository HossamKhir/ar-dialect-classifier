"""
"""

import os
import pickle
import pandas as pd
import requests
import json
from typing import Callable

AIM_URI = "https://recruitment.aimtechnologies.co/ai-tasks"
ID_LIMIT = 1000
DATA_PATH = "./data/"
FILENAME = "dialect_dataset"


def fetch_tweets_by_id(ids: list, object_hook: Callable = None) -> dict:
    """preforms a POST request to the AIM_URI, and returns the dictionary of
        id:tweet on success, raises an Exception on failure. The function
        handles the limit for the AIM_URI using `ID_LIMIT` constant

    Parameters
    ----------
    ids: array-like
        a list or array holding the IDs to retrieve, each is of type str

    Returns
    -------
    out: dict
        a dictionary where the keys are the given IDs and the corresponding
        values are the documents

    Raises
    ------
    FIXME add proper raises
    """
    tweets = {}
    for i in range(0, len(ids), ID_LIMIT):
        data = json.dumps(ids[i : i + ID_LIMIT])
        res = requests.post(AIM_URI, data)
        if res.ok:
            res.encoding = "utf8"
            i_docs = json.loads(res.text, object_hook=object_hook)
            tweets.update(i_docs)
        else:
            raise Exception("FIXME: check status code of response")
    return tweets


def check_cached_data() -> bool:
    """checks if the data has been loaded before and cached

    FIXME complete the pydoc
    """
    filename = FILENAME + ".pkl"
    for _, _, filenames in os.walk(os.path.join(DATA_PATH, "raw/")):
        if filename in filenames:
            return True
    return False


def load_local_dataset() -> pd.DataFrame:
    """loads the local dataset

    FIXME complete the pydoc
    """
    file_path = os.path.join(DATA_PATH, f"raw/{FILENAME}.csv")
    df = pd.read_csv(file_path, index_col=["id"])
    df["dialect"] = df["dialect"].astype("category")
    return df


def fetch_remote_dataset(ids: list, object_hook: Callable = None) -> pd.DataFrame:
    """fetches the dataset by POSTing to URI

    FIXME complete the pydoc
    """
    tweets = fetch_tweets_by_id(ids, object_hook=object_hook)
    return pd.DataFrame(
        tweets.values(), index=[int(k) for k in tweets.keys()], columns=["tweets"]
    )


def load_full_dataset(
    force: bool = False, cache: bool = True, object_hook: Callable = None
) -> pd.DataFrame:
    """loads the full dataset

    # FIXME complete the pydoc
    """
    cached_file_path = os.path.join(DATA_PATH, f"raw/{FILENAME}.pkl")
    if not force and check_cached_data():
        return pickle.load(open(cached_file_path, "rb"))
    else:
        local_df = load_local_dataset()
        ids = local_df.index.values.astype(str).tolist()
        remote_df = fetch_remote_dataset(ids, object_hook=object_hook)
        full_df = local_df.join(remote_df, on=local_df.index)
        if cache:
            pickle.dump(full_df, open(cached_file_path, "wb"))
        return full_df


def load_classic_model(name: str = "benchmark"):
    """loads a classic ML model saved as `.pkl` pickled file

    Parameters:
    -----------

    Returns:
    --------

    Raises:
    -------

    TODO complete the pydoc
    """
    path = os.path.join(DATA_PATH, f"models/{name}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find model {name}!")
    return pickle.load(open(path, "rb"))


if __name__ == "__main__":
    _ = load_full_dataset(force=True)
