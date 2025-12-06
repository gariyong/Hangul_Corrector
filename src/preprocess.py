# src/preprocess.py

import zipfile
import json
import os
from sklearn.model_selection import train_test_split

# 1. ZIP 파일 경로
zip_path = os.path.join("data", "TL_TX_CA_2.zip")

# 2. 멀티라벨 데이터셋 로드
def load_and_process_labels(zip_path):
    dataset = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        # zip 안 모든 파일 순회
        for file_name in z.namelist():
            if not file_name.endswith(".json"):
                continue

            with z.open(file_name) as f:
                # UTF-8 BOM 처리
                data = json.loads(f.read().decode("utf-8-sig"))

            # 문장과 레이블 추출
            sentence = data.get("ko", "").strip()
            errors = data.get("error", [])
            labels = []

            for err in errors:
                err_type = err.get("errorType")
                if err_type:
                    labels.append(err_type)

            # 중복 제거
            labels = list(set(labels))
            if sentence and labels:
                dataset.append((sentence, labels))

    return dataset

# 3. 데이터셋 로드
dataset = load_and_process_labels(zip_path)
print(f"총 샘플 수: {len(dataset)}")
print("샘플 5개:", dataset[:5])

# 4. 학습/검증 분리
train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

# 5. CSV 형태로 저장 (선택)
import csv

def save_to_csv(data, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "labels"])
        for sentence, labels in data:
            writer.writerow([sentence, ";".join(labels)])

os.makedirs("processed", exist_ok=True)
save_to_csv(train_data, "processed/train.csv")
save_to_csv(val_data, "processed/val.csv")

print("전처리 완료! processed/train.csv, processed/val.csv 생성됨")
