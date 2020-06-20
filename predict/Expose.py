import flask
from flask import request, abort

from predict.InformationCollector import *
from predict.Predict import Predict
from predict.Train import *



if __name__ == "__main__":
    pass

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/contentPicture', methods=['GET'])
def pictureContent():
    name = request.args.get('name', None)
    if name is None:
        abort(404)

    information_collector = InformationCollector()
    labels = information_collector.get_labels()

    predict = Predict()

    return predict.predict(labels, name)

app.run()


