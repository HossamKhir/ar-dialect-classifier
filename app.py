"""
"""

import json
import numpy as np
import os
from aim_task.fetch_data import DATA_PATH, load_classic_model
from aim_task.preprocess_data import init_preprocess, preprocess
from aim_task.model_training import (
    build_naive_bayes_model,
    tokenise_pad_texts,
)
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer
from flask import Flask, abort, jsonify, request

DEBUG = False
DOCS_LIMIT = 1000
LABELS = os.path.join(DATA_PATH, "raw/labels.json")

app = Flask(__name__)


@app.errorhandler(405)
def method_not_allowed(e):
    """error handler for 405 error `Method Not Allowed`

    TODO complete pydoc
    """
    return jsonify(error=str(e)), 405


@app.errorhandler(400)
def bad_request(e):
    """error handler for 400 error `bad request`

    TODO complete pydoc
    """
    return jsonify(error=str(e)), 400


@app.errorhandler(500)
def internal_server_error(e):
    """error handler for 500 error `internal server error`

    TODO complete pydoc
    """
    return jsonify(error=str(e)), 500


@app.route("/predict/", methods=["GET", "POST"])
def predict():
    """predicts the dialect of a list/array of text

    Returns:
    --------

    Raises:
    -------

    TODO complete pydoc
    """
    global model_classic, model_rnn, tknsr
    if request.method != "POST":
        return abort(405)
    try:
        payload = request.json
        docs = json.loads(payload)
        if len(docs) > DOCS_LIMIT:
            return abort(
                400, error=f"The payload length exceeds the {DOCS_LIMIT} limit"
            )
        docs_processed = preprocess(docs)
        model = request.args.get("model", "classic").lower()
        if model in ["ml", "classic"]:
            predictions = model_classic.predict(docs_processed)
        elif model in ["dl", "nn", "dnn", "rnn"]:
            docs_processed = tokenise_pad_texts(docs_processed)
            predictions = model_rnn.predict(docs_processed)
            predictions = LABELS[np.argmax(predictions, axis=1)]
        else:
            return abort(
                400,
                error="available models are: `ml`, or `dl`; default is ml",
            )
        return jsonify(dict(zip(docs, predictions)))
    except Exception as err:
        print(err)
        return abort(500)


if __name__ == "__main__":
    try:
        LABELS = np.array(json.load(open(LABELS)))
    except FileNotFoundError:
        if "X_train" not in dir():
            X_train, X_test, y_train, y_test = init_preprocess()
        LABELS = y_train.cat.categories

    # load pre-trained classical model
    try:
        model_classic = load_classic_model("model_c37")
    except FileNotFoundError:
        # load benchmark classical model
        try:
            model_classic = load_classic_model()
        except FileNotFoundError:
            # build a benchmark model
            if "X_train" not in dir():
                X_train, X_test, y_train, y_test = init_preprocess()
            model_classic = build_naive_bayes_model()
            X = preprocess(np.r_[X_train, X_test])
            y = np.r_[y_train, y_test]
            _ = model_classic.fit(X, y)

    # load the DNN model
    model_rnn_path = os.path.join(DATA_PATH, "models/model_rnn.json")
    model_rnn_json = json.load(open(model_rnn_path))
    model_rnn = model_from_json(json.dumps(model_rnn_json))

    # load the tokeniser
    try:
        tokeniser_path = os.path.join(DATA_PATH, "models/tokeniser.json")
        tknsr_json = json.load(open(tokeniser_path))
        tknsr = tokenizer_from_json(json.dumps(tknsr_json))
    except FileNotFoundError:
        # fit a tokeniser
        if "X_train" not in dir():
            X_train, X_test, y_train, y_test = init_preprocess()
        tknsr = Tokenizer()
        _ = tknsr.fit_on_texts(np.r_[X_train, X_test])

    # load the weights of the model
    try:
        weights_path = os.path.join(DATA_PATH, "weights/model_rnn_weights.h5")
        model_rnn.load_weights(weights_path)
    except FileNotFoundError:
        # fit an RNN model
        model_rnn.compile(
            "adam", "categorical_crossentropy", metrics=["categorical_accuracy"]
        )
        if "X_train" not in dir():
            X_train, X_test, y_train, y_test = init_preprocess()
        model_rnn.fit(
            np.r_[X_train, X_test],
            np.r_[y_train, y_test],
            batch_size=32,
            validation_split=0.2,
            epochs=2,
            verbose=0,
        )
    except Exception:
        # TODO should handle exceptions for incompatible weights
        pass

    app.run(debug=DEBUG)
