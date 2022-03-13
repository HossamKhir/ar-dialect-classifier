"""
"""
# flake8: noqa

import json
import os
import pickle
from typing import Callable

import pandas as pd
import requests
from sklearn.pipeline import Pipeline

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

    Returns:
    --------
    `True` if the data pickled file is found under the data path
    """
    filename = FILENAME + ".pkl"
    for _, _, filenames in os.walk(os.path.join(DATA_PATH, "raw/")):
        if filename in filenames:
            return True
    return False


def load_local_dataset() -> pd.DataFrame:
    """loads the local dataset from disk

    Returns:
    -------
    out: a dataframe with `id` as index, and a categorical column `dialect`
    """
    file_path = os.path.join(DATA_PATH, f"raw/{FILENAME}.csv")
    df = pd.read_csv(file_path, index_col=["id"])
    df["dialect"] = df["dialect"].astype("category")
    return df


def fetch_remote_dataset(ids: list, object_hook: Callable = None) -> pd.DataFrame:
    """fetches the dataset by POSTing to the AIM_URI. This subroutine
    internally handles the limit set by the AIM_URI of 1000 IDs per call

    Parameters:
    -----------
    ids: list
        the list of IDs of tweets to be fetched from the URL
    object_hook: function
        a callable function to be applied to the object returned by the
        response

    Returns:
    --------
    out: `pandas.DataFrame`
        the fetched tweets, with the given `ids` as their index
    """
    tweets = fetch_tweets_by_id(ids, object_hook=object_hook)
    return pd.DataFrame(
        tweets.values(), index=[int(k) for k in tweets.keys()], columns=["tweets"]
    )


def load_full_dataset(
    force: bool = False, cache: bool = True, object_hook: Callable = None
) -> pd.DataFrame:
    """loads the full dataset from both local storage, and remote storage

    Parameters:
    -----------
    force: bool
        whether to fetch the data regardless if it is cached locally or not
        defaults to `False`
    cache: bool
        whether to keep a cached copy of the full dataset after loading it
        defaults to `True`
    object_hook: function
        a callable subroutine to be applied on the returned tweets from
        remote fetching

    Returns:
    --------
    out: `pandas.DataFrame`
        a dataframe with `id` for index, `tweets`, `dialect` for columns
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


def load_classic_model(name: str = "benchmark") -> Pipeline:
    """loads a classic ML model saved as `.pkl` pickled file

    Parameters:
    -----------
    name: string
        the name of the model by which it was saved

    Returns:
    --------
    out: `sklearn.pipeline.Pipeline`
        a pipeline object ready for train/test

    Raises:
    -------
    FileNotFoundError
        if the given model name is not found under saved models
    """
    path = os.path.join(DATA_PATH, f"models/{name}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Could not find model {name}!")
    return pickle.load(open(path, "rb"))


if __name__ == "__main__":
    _ = load_full_dataset(force=True)
