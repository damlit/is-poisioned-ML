import os
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request
from train import train
from predict import predict


app = Flask(__name__)


@app.route('/')
def start():
    return render_template('mushroom.html', trained=False)


@app.route('/train')
def train_route():
    model_file = Path("./models/model.pkl")
    if not model_file.is_file():
        train()
    return render_template('mushroom.html', trained=True)


@app.route('/results', methods=['GET'])
def predict_route():
    data_from_form = map_attributes(request)
    is_poisonous, prob = predict(data_from_form)
    percent_prob = round((prob * 100), 2)
    return render_template('results.html', is_poisonous=is_poisonous, prob=percent_prob)


def map_attributes(r_request):
    d = {'a1': [r_request.args.get('cap-shape')], 'a2': [r_request.args.get('cap-surface')], 'a3': [r_request.args.get('cap-color')],
         'a4': [r_request.args.get('bruises')], 'a5': [r_request.args.get('odor')], 'a6': [r_request.args.get('gill-attachment')],
         'a7': [r_request.args.get('gill-spacing')], 'a8': [r_request.args.get('gill-size')], 'a9': [r_request.args.get('gill-color')],
         'a10': [r_request.args.get('stalk-shape')], 'a11': [r_request.args.get('stalk-root')], 'a12': [r_request.args.get('stalk-surface-above-ring')],
         'a13': [r_request.args.get('stalk-surface-below-ring')], 'a14': [r_request.args.get('stalk-color-above-ring')], 'a15': [r_request.args.get('stalk-color-below-ring')],
         'a16': [r_request.args.get('veil-type')], 'a17': [r_request.args.get('veil-color')], 'a18': [r_request.args.get('ring-number')],
         'a19': [r_request.args.get('ring-type')], 'a20': [r_request.args.get('spore-print-color')], 'a21': [r_request.args.get('population')],
         'a22': [r_request.args.get('habitat')]}
    df = pd.DataFrame(data=d)
    return df.values[:, 0:22].reshape(1, 22)


if __name__ == '__main__':
    app.run()
