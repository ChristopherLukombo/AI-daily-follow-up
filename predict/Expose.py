
#  Author : LUKOMBO Christopher, DELIESSCHE Angelo
#  Version : 17


import flask
from flask import request, abort
from predict.InformationCollector import *
from predict.Predict import Predict
from flask_cors import CORS, cross_origin


if __name__ == "__main__":

    pass

app = flask.Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True


@app.route('/contentPicture', methods=['GET'])
@cross_origin()
def pictureContent():
    nameInRequest = request.args.get('name', None)
    if nameInRequest is None:
        abort(404)

    information_collector = InformationCollector()
    labels = information_collector.get_labels()

    predict = Predict()

    matchedResults = predict.predict(labels, nameInRequest)
    listToReturn = []

    for i in matchedResults:
        listToReturn.append(i.decode("utf8"))
    jsonToReturn = {"datas": listToReturn}
    return jsonToReturn

app.run(host="0.0.0.0")

