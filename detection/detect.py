import torch
from PIL import Image

from configuration import MODEL_OR_WEIGHTS

#TODO: Test when model will be trained


def model_init(model_or_checkpoint: str):
    """
    Initialize (Load) Yolo model
    :param model_or_checkpoint: Yolo v5 model or checkpoint (trained model)
    E.g models from Yolo v5 ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
    :return: configured Yolo model
    """

    model = torch.hub.load('ultralytics/yolov5',
                           'custom',
                           path_or_model=model_or_checkpoint)

    return model

########


MODEL = model_init(MODEL_OR_WEIGHTS)


########

def parse_results(result_tensor):
    """
    Converts result from tensor to list for further processing
    :param result_tensor: tensor with result from Yolo detection
    :return: list of detected gestures constisting of:
    - x1, x2, y1, y2 are cords of objects
    - confidance - confidance of prediction
    - gesture - id of predicted gesture
    """
    results = []

    for detected in result_tensor:
        x1 = detected[0].item()     # for clarity
        y1 = detected[1].item()
        x2 = detected[2].item()
        y2 = detected[3].item()
        confidence = detected[4].item()
        gesture = detected[5].item()
        results.append([x1, y1, x2, y2, confidence, gesture])

    return results


def detect_gestures_from_img(img):
    """
    Performs detection of gestures from given image
    :param img: image file to proceed
    :return: list of detected gestures with format [x1, y1, x2, y2, confidance, gesture_id] if found
            otherwise - None
    """
    if img is not None:
        image = Image.open(img).convert('RGB')
        # image.filename = img.filename

        MODEL.eval()

        with torch.no_grad():
            result = MODEL(image)

            if result is not None:
                gestures = parse_results(result.xyxy[0])
                return gestures

        return None
    raise ValueError("Invalid Image File - Cannot be None")
