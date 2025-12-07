# web_inference.py

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = Flask(__name__)

# 모델 경로
MODEL_PATH = "./model_output"  # 학습된 모델이 저장된 디렉토리
NUM_LABELS = 5  # 실제 레이블 수로 변경

# 모델과 토크나이저 로드
try:
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        print("학습된 모델 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        model_ready = True
    else:
        raise FileNotFoundError
except:
    print("학습된 모델이 없으므로 임시 더미 모델 사용")
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=NUM_LABELS)
    model.eval()
    model_ready = False  # 학습 완료 여부 플래그

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    return preds.tolist()

@app.route("/")
def index():
    return render_template("index.html", model_ready=model_ready)

@app.route("/predict", methods=["POST"])
def web_predict():
    if not model_ready:
        return jsonify({"error": "모델 학습 완료 후 사용 가능합니다."})
    
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "텍스트를 입력해주세요."})
    
    preds = predict(text)
    return jsonify({"predictions": preds})

if __name__ == "__main__":
    app.run(debug=True)