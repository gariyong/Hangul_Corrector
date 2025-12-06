#src/sampler.py

import json
import zipfile
from collections import defaultdict


# 사용할 라벨 목록 정의 (프로젝트 전반에서 동일하게 사용)
LABELS = ["spac", "typo", "grammar", "spellgram", "mis_recog"]


def load_and_merge_labels(zip_path):
    merged = defaultdict(set)

    with zipfile.ZipFile(zip_path, 'r') as z:
        for filename in z.namelist():
            if filename.endswith(".json"):
                with z.open(filename) as f:
                    data = json.loads(f.read().decode("utf-8-sig"))
                    text = data["ko"]
                    err_list = data.get("error", [])

                    for e in err_list:
                        err_type = e["errorType"]
                        if err_type in LABELS:
                            merged[text].add(err_type)

    # 집합을 정렬된 리스트로 바꾸기
    result = []
    for text, label_set in merged.items():
        result.append({
            "text": text,
            "labels": sorted(list(label_set))
        })

    return result


def save_dataset_json(output_path, dataset):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    zip_path = r"C:\Users\happy\OneDrive\바탕 화면\Hangul_Corrector\data\TL_TX_CA_2.zip"

    dataset = load_and_merge_labels(zip_path)

    print("총 샘플 수:", len(dataset))
    print("예시 5개:", dataset[:5])

    save_dataset_json("merged_dataset.json", dataset)
