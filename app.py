from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from gtts import gTTS
from googletrans import Translator

app = Flask(__name__)

model = load_model("model/model.h5")
translator = Translator()

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

disease_info = {
    "Tomato_Target_Spot": {
        "cause": "Fungal infection caused by Corynespora cassiicola.",
        "solution": "Use fungicides and avoid overhead irrigation."
    },
    "Tomato_Early_blight": {
        "cause": "Caused by Alternaria solani fungus.",
        "solution": "Remove infected leaves and apply fungicide."
    },
    "Tomato_Late_blight": {
        "cause": "Caused by Phytophthora infestans fungus.",
        "solution": "Remove infected plants and apply recommended fungicides."
    },
    "Tomato_Bacterial_spot": {
        "cause": "Caused by Xanthomonas bacteria.",
        "solution": "Use certified seeds and copper-based sprays."
    },
    "Tomato_healthy": {
        "cause": "No disease detected.",
        "solution": "Maintain proper irrigation and nutrients."
    },
    "Potato___Early_blight": {
        "cause": "Fungal infection caused by Alternaria solani.",
        "solution": "Use resistant varieties and fungicides."
    },
    "Potato___Late_blight": {
        "cause": "Caused by Phytophthora infestans.",
        "solution": "Avoid excess moisture and apply fungicide."
    },
    "Potato___healthy": {
        "cause": "No disease detected.",
        "solution": "Maintain proper crop care."
    },
    "Pepper__bell___Bacterial_spot": {
        "cause": "Bacterial infection by Xanthomonas campestris.",
        "solution": "Use copper-based sprays and disease-free seeds."
    },
    "Pepper__bell___healthy": {
        "cause": "No disease detected.",
        "solution": "Maintain proper crop management."
    }
}


UPLOAD_FOLDER = "static/uploads"
AUDIO_FOLDER = "static/audio"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return redirect("/upload")
    return render_template("login.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    cause = disease_info.get(predicted_class, {}).get("cause", "Information not available.")
    solution = disease_info.get(predicted_class, {}).get("solution", "Information not available.")

    # English base text
    base_text = (
        f"Disease detected is {predicted_class}. "
        f"Confidence is {confidence} percent. "
        f"Cause is {cause}. "
        f"Prevention method is {solution}."
    )

    languages = {
        "english": "en",
        "telugu": "te",
        "hindi": "hi",
        "tamil": "ta",
        "kannada": "kn",
        "malayalam": "ml"
    }

    audio_files = {}

    for lang, code in languages.items():
        if lang == "english":
            translated_text = base_text
        else:
            translated_text = translator.translate(base_text, dest=code).text

        audio_path = os.path.join(AUDIO_FOLDER, f"{lang}.mp3")
        gTTS(text=translated_text, lang=code).save(audio_path)
        audio_files[lang] = f"{lang}.mp3"

    return render_template(
        "result.html",
        disease=predicted_class,
        confidence=confidence,
        cause=cause,
        solution=solution,
        image_path=filepath,
        audio_files=audio_files
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)
