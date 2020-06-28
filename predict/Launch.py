from predict.Train import *

from predict.InformationCollector import InformationCollector

if __name__ == "__main__":
    pass

information_collector = InformationCollector()

training_data = information_collector.create_training_data()

information_collector.mix(training_data)

X, y = information_collector.fill_X_and_y(training_data)

X = information_collector.reshape(X)

information_collector.save_infos(X, y)

# Train

train = Train()

X, y = train.load_X_and_Y()

model = train.build_model(X)

train.compile_model(model)

result = train.train_model(model, X, y)

train.save_model(model)

train.display_infos(result)

classes = information_collector.get_classes()

labels = information_collector.get_labels()

# Predict

# predict = Predict()
#
# predict.predict_display(labels)
