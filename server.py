#!/usr/bin/env python

import argparse
import json
import os
import requests
import unicodedata
import sys

import flask
from flask import Flask
from flask import request
import numpy as np
from skimage.transform import resize

from mosestokenizer import (
        MosesTokenizer, MosesPunctuationNormalizer, MosesSentenceSplitter,
        MosesDetokenizer)

JSON_HEADER = {'Content-type': 'application/json'}

APP = Flask(__name__)
APP.sentiment_en_address = None
APP.sentiment_cs_address = None

EN_MOSES_TOKENIZER = MosesTokenizer("en")
CS_MOSES_TOKENIZER = MosesTokenizer("cs")
EN_MOSES_PUNCT_NORM = MosesPunctuationNormalizer("en")
CS_MOSES_PUNCT_NORM = MosesPunctuationNormalizer("cs")
EN_MOSES_SENT_SPLITTER = MosesSentenceSplitter("en")
EN_MOSES_DETOKENIZER = MosesDetokenizer("en")
CS_MOSES_DETOKENIZER = MosesDetokenizer("cs")

ALPHANUMERIC_CHARSET = set(
    chr(i) for i in range(sys.maxunicode)
    if (unicodedata.category(chr(i)).startswith("L")
        or unicodedata.category(chr(i)).startswith("N")))


def root_dir():  # pragma: no cover
        return os.path.abspath(os.path.dirname(__file__))


def get_file(file_name):
    try:
        src = os.path.join(root_dir(), file_name)
        return open(src).read()
    except IOError as exc:
        return str(exc)


def service_runs(address):
    if address is None:
        return False

    try:
        monkey_response = requests.get("http://" + address)
        return monkey_response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


@APP.route('/', methods=['GET'])
def show_web():
    web_files = ["web-header.html"]
    if service_runs(APP.translation_encs):
        web_files.append("translation_encs.html")
    if service_runs(APP.sentiment_en_address):
        web_files.append("sentiment_en.html")
    if service_runs(APP.sentiment_cs_address):
        web_files.append("sentiment_cs.html")
    if (service_runs(APP.resnet_address) and
            service_runs(APP.captioning_en_address) and
            service_runs(APP.captioning_cs_address)):
        web_files.append("captioning.html")
    web_files.append("web-footer.html")

    content = "\n".join([get_file(f) for f in web_files])
    return flask.Response(content, mimetype="text/html")


@APP.route('/sentiment_en', methods=['POST'])
def sentiment_en():
    if APP.sentiment_en_address is None:
        response = flask.Response(
            "English sentiment analysis was not registered.",
            content_type='application/json; charset=utf-8')
        response.status_code = 503
        return response

    raw_text = request.form["text"].replace("\n", " ")
    norm_text = EN_MOSES_PUNCT_NORM(raw_text)
    tok_text = [w.lower() for w in EN_MOSES_TOKENIZER(norm_text)]
    request_json = {"text": tok_text}
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

    raw_text = request.form["text"].replace("\n", " ")
    norm_text = CS_MOSES_PUNCT_NORM(raw_text)
    tok_text = [w.lower() for w in CS_MOSES_TOKENIZER(norm_text)]

    # TODO regex pre-processing

    request_json = {"text": tok_text}
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
        "resnet_features"][0]

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

    if lng == "cs":
        output_text = CS_MOSES_DETOKENIZER(
            captioning_monkey_response.json()["target"][0])
    else:
        output_text = EN_MOSES_DETOKENIZER(
            captioning_monkey_response.json()["target"][0])
    json_response = json.dumps({"caption": output_text})
    response = flask.Response(json_response,
                              content_type='application/json; charset=utf-8')
    response.headers.add('content-length', len(json_response.encode('utf-8')))
    response.status_code = 200
    return response


def t2t_tokenize(text):

    tokens = []
    is_alnum = [ch in ALPHANUMERIC_CHARSET for ch in text]
    current_token_start = 0

    for pos in range(1, len(text)):
        # Boundary of alnum and non-alnum character groups
        if is_alnum[pos] != is_alnum[pos - 1]:
            token = text[current_token_start:pos]

            # Drop single space if it's not on the beginning
            if token != " " or current_token_start == 0:
                tokens.append(token)

            current_token_start = pos

    # Add a final token (even if it's a single space)
    final_token = text[current_token_start:]
    tokens.append(final_token)

    return tokens


@APP.route('/translation_encs', methods=['POST'])
def translation_encs():
    if APP.translation_encs is None:
        response = flask.Response(
            "English-to-Czech MT was not registered.",
            content_type='application/json; charset=utf-8')
        response.status_code = 503
        return response

    raw_text = request.form["text"]
    sentences = EN_MOSES_SENT_SPLITTER([raw_text])

    request_json = {"source": [t2t_tokenize(s) for s in sentences]}
    monkey_response = requests.post(
        "http://" + APP.translation_encs + "/run",
        json=request_json, headers=JSON_HEADER)

    if monkey_response.status_code != 200:
        response = flask.Response(
            "Machine translation service failed.",
            content_type='application/json; charset=utf-8')
        response.status_code = 500
        return response

    target = "\n".join([
        EN_MOSES_DETOKENIZER(sent) for sent in monkey_response.json()["target"]])

    json_response = json.dumps({"target": target})
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
        "--translation-encs", help="Service with English-to-Czech MT.", type=str, default=None)
    args = parser.parse_args()

    APP.translation_encs = args.translation_encs
    APP.sentiment_en_address = args.sentiment_en
    APP.sentiment_cs_address = args.sentiment_cs
    APP.resnet_address = args.resnet
    APP.captioning_en_address = args.captioning_en
    APP.captioning_cs_address = args.captioning_cs

    APP.run(debug=False, host=args.host, port=args.port)
