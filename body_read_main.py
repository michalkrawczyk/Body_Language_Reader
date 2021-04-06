from utils import softmax
from configuration import read_weights, read_classes
import numpy as np

#TODO find good naming conventions for dicts and lists


def main():
    gestures, meanings = read_weights()
    classes, _ = read_classes()

    #TODO: Change that when yolo part is ready - Test Data
    predicted_gestures = [0, 4]
    predicted_conf = [0.86, 0.5]
    predicted_conf = np.array(predicted_conf)
    ###

    prediction_weights = np.zeros(len(meanings))

    for gesture_id, gesture in enumerate(predicted_gestures):
        print("Predicted Gesture: {}".format(classes[gesture]))

        temp = gestures[gesture].copy()
        temp *= predicted_conf[gesture_id]

        prediction_weights = np.add(prediction_weights, temp)

    prediction_weights = softmax(prediction_weights)

    for idx, meaning in enumerate(meanings):
        print("{} - {:.4f} %".format(meaning, prediction_weights[idx]))


main()
