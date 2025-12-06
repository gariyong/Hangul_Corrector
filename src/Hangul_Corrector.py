import pandas as pd
import numpy as np
import os
import joblib
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 설정 변수
DATA_PATH = 'sample_data.csv'  # 샘플링된 데이터 파일 경로
MODEL_DIR = 'saved_model'      # 학습된 모델을 저장할 디렉토리
FEATURE_COL = '오류 문장'         # 모델 입력으로 사용할 텍스트 컬럼 이름
LABEL_COL = '오류 유형'           # 모델 라벨 컬럼 이름

# 1. 전처리 도구 정의
okt = Okt()
def okt_tokenizer(text):
    """Konlpy Okt를 사용한 형태소 기반 토크나이저"""
    if not isinstance(text, str):
        return []
    return okt.morphs(text)

# 2. 데이터 로드 및 정리
def load_and_preprocess_data(path):
    print(f"데이터 로드 중: {path}")
    try:
        # 실제 데이터셋의 인코딩에 따라 'utf-8' 또는 다른 인코딩 사용
        df = pd.read_csv(path, encoding='utf-8')
    except FileNotFoundError:
        print(f"오류: 데이터 파일 '{path}'을 찾을 수 없습니다. 경로를 확인해주세요.")
        return None, None, None

    # 필수 컬럼이 있는지 확인
    if FEATURE_COL not in df.columns or LABEL_COL not in df.columns:
        print(f"오류: 데이터셋에 '{FEATURE_COL}' 또는 '{LABEL_COL}' 컬럼이 없습니다.")
        return None, None, None

    # 결측치 제거 및 문자열 타입 확인
    df.dropna(subset=[FEATURE_COL, LABEL_COL], inplace=True)
    df[FEATURE_COL] = df[FEATURE_COL].astype(str)
    
    print(f"총 데이터 수: {len(df)}")
    return df

# 3. 모델 학습 및 저장
def train_and_save_model(df):
    
    # 3.1 라벨 인코딩 (문자열 라벨을 숫자로 변환)
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL])
    
    # 3.2 TF-IDF 벡터화 (피처 추출)
    print("TF-IDF 벡터화 시작...")
    # TF-IDF 객체 초기화 (미리 정의한 okt_tokenizer 사용)
    tfidf = TfidfVectorizer(tokenizer=okt_tokenizer, 
                            ngram_range=(1, 2), 
                            max_features=10000) # 피처 수를 늘려 난이도 확보
    X = tfidf.fit_transform(df[FEATURE_COL])
    print(f"피처 벡터 크기: {X.shape}")

    # 3.3 학습/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,            
        random_state=42, 
        stratify=y                # 라벨 분포를 유지하며 분할
    )
    
    # 3.4 모델 학습 (Logistic Regression)
    print("모델 학습 중...")
    model = LogisticRegression(max_iter=5000, solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    print("모델 학습 완료!")

    # 3.5 모델 성능 평가
    y_pred = model.predict(X_test)
    print("\n--- 모델 성능 평가 (테스트 데이터)")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 3.6 모델 및 전처리 도구 저장
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'))
    joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print(f"\n모델 및 전처리 도구가 '{MODEL_DIR}' 폴더에 저장되었습니다.")
    
if __name__ == "__main__":
    df = load_and_preprocess_data(DATA_PATH)
    if df is not None and not df.empty:
        train_and_save_model(df)
    else:
        print("데이터 로드 또는 전처리 실패. 학습을 건너뜁니다.")