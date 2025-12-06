# app.py

from flask import Flask, request, render_template
import joblib
import os
from konlpy.tag import Okt
import numpy as np # 예측 결과를 다루기 위해 추가

app = Flask(__name__)

#  모델 설정 
MODEL_DIR = 'saved_model'

#  1. 전처리 도구 정의
okt = Okt()
def okt_tokenizer(text):
    """Konlpy Okt를 사용한 형태소 기반 토크나이저"""
    if not isinstance(text, str):
        return []
    return okt.morphs(text)

#  2. 모델 로드 
LOADED_MODEL, LOADED_TFIDF, LOADED_ENCODER = None, None, None
MODEL_READY = False

try:
    LOADED_MODEL = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'))
    LOADED_TFIDF = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    LOADED_ENCODER = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    MODEL_READY = True
    print("모델 로드 성공: 애플리케이션 준비 완료.")
except FileNotFoundError:
    print(f"경고: 모델 파일이 '{MODEL_DIR}'에 없습니다. 학습 완료 후 재시작하세요.")

#  3. 실제 예측 함수 
def predict_error_type(sentence):
    if not MODEL_READY:
        return "모델이 로드되지 않아 예측할 수 없습니다. train_model.py를 실행하여 학습을 먼저 진행해주세요."

    try:
        # TF-IDF 변환 (학습 시 사용한 토크나이저 사용)
        sentence_vector = LOADED_TFIDF.transform([sentence])

        # 모델 예측 및 라벨 디코딩
        predicted_label_index = LOADED_MODEL.predict(sentence_vector)[0]
        predicted_type = LOADED_ENCODER.inverse_transform([predicted_label_index])[0]
        
        return predicted_type
    except Exception as e:
        return f"예측 중 오류 발생: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_sentence = ""
    
    if request.method == 'POST':
        input_sentence = request.form.get('sentence', '').strip()
        if input_sentence:
            predicted_type = predict_error_type(input_sentence) 
            # 결과를 HTML에 표시할 문자열로 정리
            prediction_result = f"입력 문장: **{input_sentence}**<br>분류 결과: **{predicted_type}**"

    return render_template('index.html', result=prediction_result, input_sentence=input_sentence)

if __name__ == '__main__':
    # 웹 앱 실행 전, saved_model 폴더가 있는지 확인하고 없으면 생성
    os.makedirs(MODEL_DIR, exist_ok=True)
    app.run(debug=True)