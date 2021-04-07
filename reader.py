from util import softmax
from configuration import read_weights, read_classes
import numpy as np
import argparse
from warnings import warn

from detection import detect_gestures_from_img

# TODO find good naming conventions for dicts and lists


def read_from_img(img):
    """
    Performs Reading Body Language from image
    :param img: image file
    :return: list of pairs [meaning_id, probability]
    """
    if img == "None":
        warn("Please provide first path to image by adding argument --img")
        return None

    gestures, meanings = read_weights()
    classes, _ = read_classes()

    detected_gestures = detect_gestures_from_img(img)
    # detected_gestures = [ [0, 0, 0, 0, 0.1, 0], [2, 2, 3, 3, 0.5, 4]]   # Tests

    if detected_gestures is None:
        return None                 # for now

    prediction_weights = np.zeros(len(meanings))

    for gesture in detected_gestures:
        _, _, _, _, conf, gesture_id = gesture
        # print("Predicted Gesture: {}".format(classes[gesture_id]))

        temp = gestures[gesture_id].copy()
        temp *= conf

        prediction_weights = np.add(prediction_weights, temp)

    prediction_weights = softmax(prediction_weights)

    results = []
    for idx, meaning in enumerate(meanings):
        # print("{} - {:.4f} %".format(meaning, prediction_weights[idx]))
        results.append([idx, prediction_weights[idx]])

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='None', help='path to image')
    opt = parser.parse_args()
    print(read_from_img(opt.img))
