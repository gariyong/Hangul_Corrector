# web_inference.py

import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer

# ========================
# 전역 변수
# ========================
model = None
tokenizer = None
mlb = None
model_ready = False

# ========================
# 경로 설정
# ========================
MODEL_PATH = "./model_output/checkpoint-625"  # 학습 체크포인트 경로
LABELS_PATH = "./processed/mlb_classes.npy"   # mlb 클래스 정보
NUM_LABELS = 5  # 임시 더미 모델 사용 시

current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, '..', 'templates')
print(f"Flask 템플릿 경로: {template_dir}")

# ========================
# Flask 앱 생성
# ========================
app = Flask(__name__, template_folder=template_dir)

# ========================
# Helper: 마지막 체크포인트 검색
# ========================
def get_last_checkpoint(folder):
    if not os.path.exists(folder):
        return None
    checkpoints = [d for d in os.listdir(folder)
                   if d.startswith('checkpoint-') and os.path.isdir(os.path.join(folder, d))]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[1]), reverse=True)
    return os.path.join(folder, checkpoints[0])

# ========================
# 모델 로드
# ========================
def load_model():
    global model, tokenizer, mlb, model_ready
    print("모델 로드 중...")
    try:
        checkpoint_dir = get_last_checkpoint("./model_output")
        if checkpoint_dir:
            model_dir = checkpoint_dir
            print(f"Checkpoint 발견: {model_dir}")
        elif os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin")):
            model_dir = MODEL_PATH
            print(f"최종 모델 로드: {MODEL_PATH}")
        else:
            raise FileNotFoundError("학습된 모델이 없습니다.")

        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model.eval()

        if os.path.exists(LABELS_PATH):
            mlb_classes = np.load(LABELS_PATH, allow_pickle=True)
            mlb = MultiLabelBinarizer()
            mlb.fit([mlb_classes])
            print(f"{len(mlb.classes_)}개의 클래스 로드 완료")
        else:
            print("mlb_classes.npy가 없습니다. 멀티라벨 디코딩 불가")
            mlb = None

        model_ready = True
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {e}", file=sys.stderr)
        print("임시 더미 모델 사용")
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=NUM_LABELS)
        model.eval()
        model_ready = False
        mlb = None

# ========================
# 예측 함수
# ========================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
    return preds.tolist()

# ========================
# 임시 교정 함수
# ========================
def simple_correction(sentence, labels):
    if labels:
        return f"[오류 분류 완료] {sentence}"
    return sentence

# ========================
# Flask 라우트
# ========================
@app.route("/")
def index():
    return render_template("index.html", model_ready=model_ready)

@app.route("/correct", methods=["POST"])
def correct():
    if not model_ready:
        return jsonify({
            "corrected_sentence": "오류: 모델이 아직 로드되지 않았거나 학습되지 않았습니다.",
            "labels": []
        }), 503

    try:
        data = request.json
        sentence = data.get("sentence", "")

        if not sentence:
            return jsonify({
                "corrected_sentence": "오류: 입력 문장이 비어 있습니다.",
                "labels": []
            }), 400

        # 토큰화 및 추론
        inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
        predictions = (probs > 0.5).astype(int)

        # 멀티라벨 디코딩
        if mlb:
            decoded_labels = mlb.inverse_transform(predictions.reshape(1, -1))
            label_list = list(decoded_labels[0])
        else:
            label_list = ["모델 로드 오류 (클래스 정보 없음)"]

        corrected_sentence = simple_correction(sentence, label_list)

        return jsonify({
            "corrected_sentence": corrected_sentence,
            "labels": label_list
        })

    except Exception as e:
        return jsonify({
            "corrected_sentence": f"서버 처리 오류: {e}",
            "labels": []
        }), 500

# ========================
# 메인
# ========================
if __name__ == "__main__":
    load_model()
    app.run(debug=True)