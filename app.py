import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, render_template, request
from torchvision import transforms
from sklearn.tree import DecisionTreeClassifier
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Device dan model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("models/cnn_resnet50_skin_classifier_scripted.pt", map_location=device)
model.eval()

# Transformasi untuk CNN
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_skin_type(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        classes = ["Acne", "Dry", "Normal", "Oily"]
        return classes[predicted.item()]

def build_decision_tree():
    skincare_data = {
        "Jenis Kulit": ["Dry", "Oily", "Acne", "Normal"] * 3,
        "Cuaca": ["Panas", "Lembap", "Dingin"] * 4,
        "Alergi": ["Tidak", "Tidak", "Ya"] * 4,
        "Produk Sebelumnya": ["Cocok", "Tidak Cocok", "Belum Pernah"] * 4,
        "Rekomendasi": [
            "Gel moisturizer ringan", "Toner dengan niacinamide", "Krim soothing untuk acne",
            "Sunscreen dengan pelembap", "Gel ringan oil-free", "Krim pelembap untuk kulit kering",
            "Cleanser lembut & serum calming", "Pelembap dengan panthenol", "Produk basic tanpa iritan",
            "Krim ringan hydrating", "Produk dengan centella asiatica", "Moisturizer ceramide"
        ],
        "Dihindari": [
            "Sabun berbusa tinggi", "Krim berat berbasis minyak", "Essential oil, alkohol tinggi",
            "Fragrance tinggi", "Scrub kasar, alkohol", "Produk occlusive",
            "Alkohol, pewangi", "Exfoliator keras", "Produk trial-unknown",
            "Krim tebal", "Bahan eksfoliasi tinggi", "Minyak mineral"
        ],
        "Bahan Disarankan": [
            "Hyaluronic acid", "Niacinamide", "Panthenol, Centella",
            "Green tea extract", "Salicylic acid", "Ceramide",
            "Chamomile", "Aloe vera", "Squalane",
            "Oat extract", "Cica extract", "Vitamin E"
        ]
    }

    df = pd.DataFrame(skincare_data)
    X = pd.get_dummies(df[["Jenis Kulit", "Cuaca", "Alergi", "Produk Sebelumnya"]])
    y = df[["Rekomendasi", "Dihindari", "Bahan Disarankan"]]
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf, X.columns

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/scan")
def scan():
    return render_template("scan.html")

@app.route("/form", methods=["POST"])
def form():
    image_data = request.form.get("image_data")
    if not image_data:
        return "No image received", 400

    try:
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return "Invalid image format", 400

    return render_template("form.html", image_data=request.form["image_data"])

@app.route("/predict", methods=["POST"])
def predict():
    image_data = request.form.get("image_data")

    if image_data and image_data.startswith("data:image"):
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    else:
        return "Invalid image data", 400

    # Ambil input dari form
    alergi_status = request.form["alergi_status"]
    alergi_bahan = request.form["alergi_bahan"]
    produk_nama = request.form["produk_nama"]
    produk_status = request.form["produk_status"]

    # Prediksi jenis kulit
    skin_type = predict_skin_type(image)

    # Dummy cuaca
    weather = "Lembap"
    temp = 27
    humidity = 85

    # Prediksi skincare
    clf, feature_columns = build_decision_tree()
    input_data = pd.DataFrame([[skin_type, weather, alergi_status, produk_status]],
                              columns=["Jenis Kulit", "Cuaca", "Alergi", "Produk Sebelumnya"])
    input_data = pd.get_dummies(input_data)
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0
    input_data = input_data[feature_columns]

    prediction = clf.predict(input_data)
    rekomendasi, dihindari, bahan_disarankan = prediction[0]

    if alergi_status == "Ya" and alergi_bahan.strip() != "":
        dihindari = f"{alergi_bahan}, {dihindari}"

    return render_template("result.html",
                           location="Tangerang Selatan",
                           weather=weather,
                           temp=temp,
                           humidity=humidity,
                           skin_type=skin_type,
                           alergi_status=alergi_status,
                           alergi_bahan=alergi_bahan,
                           produk_nama=produk_nama,
                           produk_status=produk_status,
                           rekomendasi=rekomendasi,
                           dihindari=dihindari,
                           bahan_disarankan=bahan_disarankan)

if __name__ == "__main__":
    app.run(debug=True)
