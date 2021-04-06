import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'weight_vectors.yaml')
CLASSES_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data.yaml')