import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_efficientnetv2s_sawit_finetuned.keras")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")


def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def load_sawit_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def predict_image(model, img_path):
    class_names = load_class_names()

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    all_probabilities = {
        class_names[i]: float(predictions[i]) * 100
        for i in range(len(class_names))
    }

    return predicted_class, confidence, all_probabilities


if __name__ == "__main__":
    model = load_sawit_model()

    test_image_path = r"D:\Skripsi\EfficientNetV2S_Sawit\sample_test\tes2.jpg"

    predicted_class, confidence, probabilities = predict_image(model, test_image_path)

    print("Hasil Prediksi:", predicted_class)
    print("Confidence:", f"{confidence:.2f}%")
    print("Probabilitas:")

    for class_name, prob in probabilities.items():
        print(f"- {class_name}: {prob:.2f}%")