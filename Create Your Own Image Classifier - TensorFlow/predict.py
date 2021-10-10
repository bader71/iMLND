import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
from PIL import Image
import warnings
import json

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Image classifier, prediction part")

parser.add_argument("--input", default = "./test_images/hard-leaved_pocket_orchid.jpg", action = "store", type = str, help = "image path")
parser.add_argument("--model", default = "./test_model.h5", action = "store", type = str, help = "checkpoint file path/name")
parser.add_argument("--top_k", default = 3,dest="top_k", action = "store", type = int, help = "return top k")
parser.add_argument("--category_names", default = "label_map.json",dest = "category_names", action = "store", help = "mapping to names")

arg_parser = parser.parse_args()
image_path = arg_parser.input
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [224, 224])
    image = image/255
    return image.numpy()


def predict(image, model, k):
    image = Image.open(image)
    image_np = np.asarray(image)
    processed_image = process_image(image_np)
    pred_image = np.expand_dims(processed_image, axis=0)
    preds = model.predict(pred_image)[0]
    idx = np.argpartition(preds, -k)[-k:]
    indices = idx[np.argsort((-preds)[idx])]
    top_k_labels = [str(k) for k in indices]
    return preds[indices], top_k_labels


if __name__ == "__main__":
    print("begin...")
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    model_2 = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    probs, classes = predict(image_path, model_2, topk)
    label_names = [class_names[str(int(idd)+1)] for idd in classes]
    print("probs:", probs)
    print("classes:", classes)
    print("label names:", label_names)
    print("end prediction")
    
    