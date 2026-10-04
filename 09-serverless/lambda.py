import json
from io import BytesIO
from urllib import request

import numpy as np
import onnxruntime as ort
from PIL import Image

model = ort.InferenceSession("hair_classifier_empty.onnx", providers=["CPUExecutionProvider"])
input_name = model.get_inputs()[0].name


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    return Image.open(BytesIO(buffer))


def prepare_image(img, target_size):
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(target_size, Image.NEAREST)


def preprocess(img, target_size=(200, 200)):
    img = prepare_image(img, target_size)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.expand_dims(x.transpose(2, 0, 1), axis=0)


def handler(event, context):
    url = event["url"]
    img = download_image(url)
    x = preprocess(img)
    logit = model.run(None, {input_name: x})[0][0][0]
    return {"prediction": float(logit)}

if __name__ == "__main__":
    print(handler({"url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"}, None))