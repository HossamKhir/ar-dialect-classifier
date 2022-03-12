"""
"""

import json
import os
from fetch_data import DATA_PATH, load_classic_model
from preprocess_data import preprocess
from tensorflow.keras.models import model_from_json
from flask import Flask, abort, jsonify, request

DEBUG = True
DOCS_LIMIT = 1000

app = Flask(__name__)

try:
    model_classic = load_classic_model("model_c37")
except FileNotFoundError:
    try:
        model_classic = load_classic_model()
    except FileNotFoundError:
        print(
            """Default model not found.
        Kindly train a model and try again.
        """
        )
        quit(-1)

try:
    model_rnn_path = os.path.join(DATA_PATH, "models/model_rnn.json")
    model_rnn_json = json.load(open(model_rnn_path))
    model_rnn = model_from_json(json.dumps(model_rnn_json))
except FileNotFoundError:
    print(
        """RNN model not found.
        Kindly build an RNN model and try again.
        """
    )
    quit(-1)
try:
    model_rnn.load_weights(os.path.join(DATA_PATH, "weights/model_rnn_weights.h5"))
except FileNotFoundError:
    print(
        """The weights for the RNN model not found.
        Kindly train the RNN model, then try again
        """
    )
except:
    # TODO should handle exceptions for incompatible weights
    pass


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
    print("handling request")
    if request.method != "POST":
        return abort(405)
    print("method is post")
    try:
        payload = request.json
        print(payload)
        docs = json.loads(payload)
        if len(docs) > DOCS_LIMIT:
            return abort(
                400, error=f"The payload length exceeds the {DOCS_LIMIT} limit"
            )
        docs_processed = preprocess(docs)
        print("NOTE:\tpreprocess done")
        model = request.args.get("model", "classic").lower()
        if model in ["ml", "classic"]:
            predictions = model_classic.predict(docs_processed)
        elif model in ["dl", "nn", "dnn", "rnn"]:
            pass
            # predictions = model_rnn.predict(docs_processed)
        else:
            return abort(
                400,
                error="available models are: `ml`, or `dl`; default is ml",
            )
        return jsonify(dict(zip(docs, predictions)))
    except:
        return abort(500)


if __name__ == "__main__":
    app.run(debug=DEBUG)
