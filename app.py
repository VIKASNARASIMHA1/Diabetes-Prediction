"""Gradio web app for the diabetes classifier.

    python app.py

Loads models/diabetes_model.joblib (created by `python train.py` or by the notebook); trains it automatically if it is missing.
Leave a field empty (or 0) when a measurement is unknown: the model's imputer fills it in.
"""
import joblib
import gradio as gr
import numpy as np
import pandas as pd

from train import FEATURES, MODEL_PATH, ZERO_MEANS_MISSING, train


def load_bundle():
    if not MODEL_PATH.exists():
        print("No saved model found, training one now (takes a few seconds)...")
        train(save=True, verbose=False)
    return joblib.load(MODEL_PATH)


BUNDLE = load_bundle()
MODELS, BEST = BUNDLE["models"], BUNDLE["best"]


def card(title, subtitle, colors, text):
    return f"""
    <div style='background: linear-gradient(135deg, {colors[0]}, {colors[1]}); padding:20px; border-radius:15px;
                text-align:center; box-shadow: 0 4px 10px rgba(0,0,0,0.15);'>
        <h2 style='color:{text}; margin:0;'>{title}</h2>
        <p style='color:{text}; margin-top:8px;'>{subtitle}</p>
        <p style='color:{text}; margin-top:8px; font-size:0.85em; opacity:0.85;'>
            Educational demo, not medical advice. Please consult a doctor for any health concern.</p>
    </div>"""


def predict_diabetes(algorithm, pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age):
    row = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]], columns=FEATURES)
    row = row.apply(pd.to_numeric, errors="coerce")                      # empty field -> NaN
    row[ZERO_MEANS_MISSING] = row[ZERO_MEANS_MISSING].replace(0, np.nan)  # 0 is not a real measurement here
    model = MODELS[algorithm]
    positive = int(model.predict(row)[0]) == 1
    detail = f"Model used: {algorithm}"
    if hasattr(model, "predict_proba"):
        detail += f" | estimated probability of diabetes: {model.predict_proba(row)[0, 1]:.0%}"
    if positive:
        return card("⚠️ Likelihood of Diabetes Detected", detail, ("#ff9a9e", "#fecfef"), "#7a0000")
    return card("✅ No Diabetes Detected", detail, ("#a8ff78", "#78ffd6"), "#004d00")


demo = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Dropdown(choices=list(MODELS), value=BEST, label="🧠 Choose Algorithm"),
        gr.Number(label="🤰 Pregnancies", value=1),
        gr.Number(label="🍬 Glucose (mg/dL)", value=120),
        gr.Number(label="💓 Blood Pressure (mm Hg)", value=70),
        gr.Number(label="📏 Skin Thickness (mm)", value=20),
        gr.Number(label="💉 Insulin (mu U/ml)", value=79),
        gr.Number(label="⚖️ BMI", value=25.0),
        gr.Number(label="🧬 Diabetes Pedigree Function", value=0.5),
        gr.Number(label="🎂 Age", value=30),
    ],
    outputs=gr.HTML(),
    title="🩺 Diabetes Prediction Web App",
    description="Choose an algorithm and enter the medical measurements. Leave a field empty if the value is unknown.",
    examples=[[BEST, 5, 166, 72, 19, 175, 25.8, 0.587, 51], [BEST, 1, 85, 66, 29, None, 26.6, 0.351, 31]],
)

if __name__ == "__main__":
    demo.launch()
