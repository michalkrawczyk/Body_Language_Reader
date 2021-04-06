from utils import read_yaml
from .definitions import WEIGHTS_YAML, CLASSES_YAML
import numpy as np
import pandas as pd


def read_weights():
    """
    Reads yaml with weights and meanings
    :return gesture_weight_dict - weights list for specific gesture (by gesture's id)
            meaning_id_dict - meanings with their ids
    """
    data_dict = read_yaml(WEIGHTS_YAML)

    gesture_weights_dict = {}

    for gesture in data_dict['weights']:
        value = next(iter(gesture.values()))
        key = next(iter(gesture.keys()))

        assert (len(value) == len(data_dict['meanings']))  # vector of meanings don't match
        value = np.array(value)

        gesture_weights_dict[key] = value

    meanings_id_dict = (pd.Categorical(data_dict['meanings']))
    return gesture_weights_dict, meanings_id_dict


def read_classes():
    """
    Reads yaml with classes
    That .yaml file is also used for training YOLO network
    :return classes_names - list with class names (names of detected gestures)
            meaning_id_dict - meanings with their ids
    """
    data_dict = read_yaml(CLASSES_YAML)

    classes_names = data_dict['names']
    classes_count = data_dict['nc']

    assert (isinstance(classes_count, int))
    assert (classes_count == len(classes_names))

    return classes_names, classes_count
