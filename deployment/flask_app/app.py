from flask import Flask, request
from flask_restful import Api, Resource
from pyngrok import ngrok
from PIL import Image
import numpy as np
import concurrent.futures

# from ultralytics import YOLO
import cv2
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor


MOST_MEDICAL_MODELS = r"D:\Projects\Python Devolpment\Data field\python\data scintict\medicals-models\_models\yolo_4_midecal.pt"
ACNE_MODEL = r"D:\Projects\Python Devolpment\Data field\python\data scintict\medicals-models\_models\ance3Classes.h5"
ECZEMA_MODEL = r"D:\Projects\Python Devolpment\Data field\python\data scintict\medicals-models\_models\eczemaH5v2.h5"
HEALTHY_SKIN_MODEL = r"D:\Projects\Python Devolpment\Data field\python\data scintict\medicals-models\_models\HealthySkin.h5"
ROSACEA_MODEL = r"D:\Projects\Python Devolpment\Data field\python\data scintict\medicals-models\_models\rosacea200.h5"
PSORIASIS_MODEL = r"D:\Projects\Python Devolpment\Data field\python\data scintict\medicals-models\_models\psoriasis.h5"
psoriasis_rosacea_MODEL = r"D:\Projects\Python Devolpment\Data field\python\data scintict\medicals-models\_models\psoriasis_rosacea.h5"
MODELS = {
    "all": {
        "path": "",
        "type": "h5",
        "classes": [
            "Acne and Rosacea Photos",
            "Healthy-skin",
            "Psoriasis",
            "eczema",
            "rosacea200",
            "scalp infection scrapping",
            "skin acne",
        ],
    }
}
# MODELS = {
#     'rosacea': {
#         'path': ROSACEA_MODEL,
#         'type': 'h5',
#         'classes': ['rosacea']
#     },
#     'acne': {
#         'path': ACNE_MODEL,
#         'type': 'h5',
#         'classes': ['normal skin', 'skin acne', 'skin psoriasis']
#     },
#     'eczemaH5': {
#         'path': ECZEMA_MODEL,
#         'type': 'h5',
#         'classes': ['Asteatotic Eczema', 'Chronic Eczema', 'Hand Eczema', 'Nummular Eczema', 'Subacute Eczema']
#     },
#     'HealthySkin': {
#         'path': HEALTHY_SKIN_MODEL,
#         'type': 'h5',
#         'classes': ['Healthy-skin', 'Measles', 'Rubella']
#     },
#     'psoriasis': {
#         'path': PSORIASIS_MODEL,
#         'type': 'h5',
#         'classes': ['Healthy-skin', 'Measles', 'Rubella']
#     },
#     'psoriasis_rosacea_MODEL': {
#         'path': psoriasis_rosacea_MODEL,
#         'type': 'h5',
#         'classes': ['normal', 'psoriasis', 'rosacea']
#     }
# }


def load_trained_model(model_dir):
    """
    this function take the model dir load it then from it know the input and the output shape
    Args
      model_dir => path of the model
    Return
      model, input_shape,classes in list(take len of it)

    """

    model = tf.keras.models.load_model(model_dir)
    # return model
    # new edit
    input_shape = list(model.layers[0].input_shape[0])
    output_len = list(range(0, model.layers[-1].output.shape[1]))
    input_shape = input_shape[1:-1]
    return model, input_shape[::-1], output_len  # input_shape[1:-1]


def classifier(img, model, shape) -> int:
    """"""
    if img is None:
        raise ValueError("Image should not be None")

    # img=cv2.imread(img)
    img = cv2.resize(img, dsize=shape)
    # img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # [[224,224,3]]
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred)
    print("model output: ", pred)

    # values={k:i for i,k in train_batches.class_indices.items()}
    # print(values[idx])
    return idx, pred


def yolo_prediction(img, model):
    model = YOLO(model)
    results = model.predict(
        source=img,
        # save_crop=True,
        # save_txt=True,
        device="cpu",
        # conf_thres=0.25,
    )
    classes = results[0].names
    conf = results[0].boxes.conf.tolist()
    print(results[0].boxes.conf)
    print(results[0].names)
    print("__" * 50)
    if len(conf) == 0:
        return 0, classes
    return max(conf), classes


def process_model(model, img):
    # if MODELS[model]['type'] == 'yolo':
    #     return None
    #     conf, c = yolo_prediction(img, MODELS[model]['path'])
    #     return model, {'conf': conf, 'classes': c}
    if MODELS[model]["type"] == "h5":
        model_obj, shape, _ = load_trained_model(MODELS[model]["path"])
        # print('model: ', model)
        idx, pred = classifier(img, model_obj, shape)
        # print('pred:', pred)
        # print('idx:', idx)
        # print('__'*50)

        return model, {
            "conf": pred,
            "classes": MODELS[model]["classes"],
            "class_int": int(idx),
        }


def get_best_model(results):
    best_model = None
    best_confidence = -1
    best_class = None
    max_acc=0
    for model, result in results.items():
        max_acc = result["conf"].max()
        if max_acc > best_confidence and max_acc != 1:
            best_model = model
            best_confidence = max_acc
            best_class = result["classes"][result["conf"].argmax()]
    print(results)
    if max_acc < 0.6:
        return "no prediction"
    else:
        return best_class


"""
[[2.0900794e-22, 1.0000000e+00, 1.7757469e-18, 2.2246467e-24, 
        3.7089987e-20, 2.9936779e-18, 1.5934527e-25]],
        dtype=float32), 'classes': ['Acne and Rosacea Photos', 'Healthy-skin', 'Psoriasis', 'eczema', 'rosacea200', 'scalp infection scrapping', 'skin acne'], 'class_int': 5}
        'classes': ['Acne and Rosacea Photos', 'Healthy-skin', 'Psoriasis', 'eczema', 'rosacea200', 'scalp infection scrapping', 'skin acne'], 'class_int': 5}
"""

# best_model, best_class = get_best_model(x)
# print(f'Best model: {best_model}, Class: {best_class}')


def predict_class(img):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_model = {
            executor.submit(process_model, model, img): model for model in MODELS
        }
        for future in concurrent.futures.as_completed(future_to_model):
            model = future_to_model[future]
            try:
                model_name, result = future.result()
                results[model_name] = result
            except Exception as exc:
                print(f"{model} generated an exception: {exc}")
    # print('__'*50)

    return get_best_model(results)


app = Flask(__name__)
api = Api(app)


class ClassificationAPI(Resource):
    def post(self):
        img = Image.open(request.files["image"])
        image = np.array(img)
        return predict_class(image)


api.add_resource(ClassificationAPI, "/predict", methods=["POST"])

if __name__ == "__main__":
    url = ngrok.connect(5000).public_url
    print(" ***** Tunnel URL:", url)
    app.run(debug=False, host="0.0.0.0", port=5000)
