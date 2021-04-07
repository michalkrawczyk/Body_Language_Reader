import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_YAML = os.path.join(ROOT_DIR, 'weight_vectors.yaml')

CLASSES_YAML = os.path.join(ROOT_DIR, 'data.yaml')

MODEL_OR_WEIGHTS = os.path.join(ROOT_DIR, 'best.pt')
