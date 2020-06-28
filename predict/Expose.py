import json
from json import JSONEncoder
import numpy
import flask
from flask import request, abort

from PIL import Image
from predict.InformationCollector import *
from predict.Predict import Predict
from predict.Train import *
import base64

# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, numpy.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)


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


    val = predict.predict(labels, name)


    # Serialization
    # numpyData = {"array": val}
    # encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)

    return val

app.run()


