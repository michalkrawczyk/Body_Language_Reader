import yaml
from os import path


def read_yaml(filename: str) -> dict:
    """
    Reads Yaml File
    :param filename: path to .yaml file
    :return: Dictionary with data from .yaml
    """
    assert(path.isfile(filename))
    data = None

    with open(filename) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    assert(isinstance(data, dict))

    return data


