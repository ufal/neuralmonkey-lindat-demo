#!/usr/bin/env python

import argparse
import json
import os
import requests

import flask
from flask import Flask
from flask import request
import numpy as np
from skimage.transform import resize

import nltk
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')
from nltk.tokenize.moses import MosesTokenizer

JSON_HEADER = {'Content-type': 'application/json'}

APP = Flask(__name__)
APP.sentiment_en_address = None
APP.sentiment_cs_address = None

EN_MOSES_TOKENIZER = MosesTokenizer("en")
CS_MOSES_TOKENIZER = MosesTokenizer("cs")


@APP.route('/')
def index():
    return "{}\n"


def root_dir():  # pragma: no cover
        return os.path.abspath(os.path.dirname(__file__))


def get_file(file_name):
    try:
        src = os.path.join(root_dir(), file_name)
        return open(src).read()
    except IOError as exc:
        return str(exc)


@APP.route('/web', methods=['GET'])
def show_web():
    content = get_file("web.html")
    return flask.Response(content, mimetype="text/html")


@APP.route('/sentiment_en', methods=['POST'])
def sentiment_en():
    if APP.sentiment_en_address is None:
        response = flask.Response(
            "English sentiment analysis was not registered.",
            content_type='application/json; charset=utf-8')
        response.status_code = 503
        return response

    raw_text = request.form["text"]
    tok_text = [w.lower() for w in EN_MOSES_TOKENIZER.tokenize(raw_text)]
    request_json = {"text": [raw_text.lower().split()]}
    monkey_response = requests.post(
        "http://" + APP.sentiment_en_address + "/run",
        json=request_json, headers=JSON_HEADER)

    if monkey_response.status_code != 200:
        response = flask.Response(
            "Sentiment analysis service failed.",
            content_type='application/json; charset=utf-8')
        response.status_code = 500
        return response

    stars = int(monkey_response.json()["score"][0][0])

    json_response = json.dumps({"stars": stars})
    response = flask.Response(json_response,
                              content_type='application/json; charset=utf-8')
    response.headers.add('content-length', len(json_response.encode('utf-8')))
    response.status_code = 200
    return response


@APP.route('/sentiment_cs', methods=['POST'])
def sentiment_cs():
    if APP.sentiment_cs_address is None:
        response = flask.Response(
            "Czech sentiment analysis was not registered.",
            content_type='application/json; charset=utf-8')
        response.status_code = 503
        return response

    raw_text = request.form["text"]
    tok_text = [w.lower() for w in EN_MOSES_TOKENIZER.tokenize(raw_text)]

    # TODO regex pre-processing

    request_json = {"text": [raw_text.lower().split()]}
    monkey_response = requests.post(
        "http://" + APP.sentiment_cs_address + "/run",
        json=request_json, headers=JSON_HEADER)

    if monkey_response.status_code != 200:
        response = flask.Response(
            "Sentiment analysis service failed.",
            content_type='application/json; charset=utf-8')
        response.status_code = 500
        return response

    sentiment = monkey_response.json()["score"][0][0]

    json_response = json.dumps({"sentiment": sentiment})
    response = flask.Response(json_response,
                              content_type='application/json; charset=utf-8')
    response.headers.add('content-length', len(json_response.encode('utf-8')))
    response.status_code = 200
    return response


@APP.route('/captioning', methods=['POST'])
def captioning():
    if APP.resnet_address is None:
        response = flask.Response(
            "Image analysis using ResNet was not registered.",
            content_type='application/json; charset=utf-8')
        response.status_code = 503
        return response

    lng = request.form['lng']
    captioning_address = None
    if lng == 'en':
        captioning_address = APP.captioning_en_address
    elif lng == 'de':
        captioning_address = APP.captioning_de_address
    elif lng == 'cs':
        captioning_address = APP.captioning_cs_address

    if captioning_address is None:
        response = flask.Response(
            "Captioning for language {} is not registered.".format(lng),
            content_type='application/json; charset=utf-8')
        response.status_code = 503
        return response

    img_data = np.array([
        int(i) for i in request.form['img_data'].split(",")])
    orig_w = int(request.form['img_w'])
    orig_h = int(request.form['img_h'])

    img = img_data.reshape([orig_w, orig_h, 4])
    preprocessed = resize(img[:, :, :3] / 255, (229, 229))

    resnet_request_json = {"images": [preprocessed.tolist()]}
    resnet_monkey_response = requests.post(
        "http://" + APP.resnet_address + "/run",
        json=resnet_request_json, headers=JSON_HEADER)

    if resnet_monkey_response.status_code != 200:
        response = flask.Response(
            "Image analysis using ResNet failed.",
            content_type='application/json; charset=utf-8')
        response.status_code = 500
        return response

    resnet_fetures = resnet_monkey_response.json()[
        "resnet_features"][0]['imagenet/StopGradient:0']

    captioning_request_json = {"images": [resnet_fetures]}
    captioning_monkey_response = requests.post(
        "http://" + captioning_address + "/run",
        json=captioning_request_json, headers=JSON_HEADER)

    if captioning_monkey_response.status_code != 200:
        response = flask.Response(
            "Captioning service failed.",
            content_type='application/json; charset=utf-8')
        response.status_code = 500
        return response

    output_text = " ".join(captioning_monkey_response.json()["target"][0])
    json_response = json.dumps({"caption": output_text})
    response = flask.Response(json_response,
                              content_type='application/json; charset=utf-8')
    response.headers.add('content-length', len(json_response.encode('utf-8')))
    response.status_code = 200
    return response


@APP.route('/translation_encs', methods=['POST'])
def sentiment_en():
    if APP.translation_encs is None:
        response = flask.Response(
            "English-to-Czech MT was not registered.",
            content_type='application/json; charset=utf-8')
        response.status_code = 503
        return response

    raw_text = request.form["text"]
    tok_text = [w.lower() for w in EN_MOSES_TOKENIZER.tokenize(raw_text)]

    request_json = {"text": [raw_text.lower().split()]}
    monkey_response = requests.post(
        "http://" + APP.sentiment_en_address + "/run",
        json=request_json, headers=JSON_HEADER)

    if monkey_response.status_code != 200:
        response = flask.Response(
            "Machine translation service failed.",
            content_type='application/json; charset=utf-8')
        response.status_code = 500
        return response

    # TODO retrieve translation instead of score
    stars = int(monkey_response.json()["score"][0][0])

    json_response = json.dumps({"stars": stars})
    response = flask.Response(json_response,
                              content_type='application/json; charset=utf-8')
    response.headers.add('content-length', len(json_response.encode('utf-8')))
    response.status_code = 200
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the demo server')
    parser.add_argument(
        "--port", help="Port the server runs on", type=int, default=5000)
    parser.add_argument(
        "--host", help="IP address the server will run at",
        type=str, default="127.0.0.1")
    parser.add_argument(
        "--sentiment-en", help="Service with English sentiment analysis.",
        type=str, default=None)
    parser.add_argument(
        "--sentiment-cs", help="Service with Czech sentiment analysis.",
        type=str, default=None)
    parser.add_argument(
        "--resnet", help="Service with image feature extraction via resnet.",
        type=str, default=None)
    parser.add_argument(
        "--captioning-en", help="Service with English image captioning.",
        type=str, default=None)
    parser.add_argument(
        "--captioning-cs", help="Service with Czech image captioning.",
        type=str, default=None)
    parser.add_argument(
        "--translation-encs", help="Service with English-to-Czech MT.",
        type=str, default=None)
    args = parser.parse_args()

    APP.sentiment_en_address = args.sentiment_en
    APP.sentiment_cs_address = args.sentiment_cs
    APP.resnet_address = args.resnet
    APP.captioning_en_address = args.captioning_en
    APP.captioning_cs_address = args.captioning_cs
    APP.translation_encs = args.translation_encs

    APP.run(debug=True, host=args.host, port=args.port)
