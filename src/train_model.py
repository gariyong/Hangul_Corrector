import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import f1_score

# 1. CSV 로드
train_df = pd.read_csv("processed/train.csv")
val_df = pd.read_csv("processed/val.csv")

# 2. labels 분리
train_df['labels'] = train_df['labels'].apply(lambda x: x.split(";"))
val_df['labels'] = val_df['labels'].apply(lambda x: x.split(";"))

# 3. 멀티라벨 인코딩
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_df['labels'])
val_labels = mlb.transform(val_df['labels'])

# 4. Hugging Face Dataset 변환
train_dataset = Dataset.from_dict({
    "text": train_df["sentence"].tolist(),
    # 레이블을 float으로 변환하여 리스트로 저장 (numpy의 .astype(float) 사용)
    "labels": train_labels.astype(float).tolist() 
})
val_dataset = Dataset.from_dict({
    "text": val_df["sentence"].tolist(),
    "labels": val_labels.astype(float).tolist() 
})

# 5. Tokenizer
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    # 'labels'는 이미 float으로 변환되었으므로 토큰화만 진행
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# 6. PyTorch 포맷 설정
train_dataset.set_format(
    type="torch", 
    columns=["input_ids", "attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch", 
    columns=["input_ids", "attention_mask", "labels"],
)

# 7. 모델 초기화 (멀티라벨 분류)
num_labels = len(mlb.classes_)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# 8. TrainingArguments 설정 
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy='epoch',
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    eval_strategy='epoch',
    eval_steps=200,
)

# 9. Trainer 초기화
def compute_metrics(pred):
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    
    return {"f1": f1_score(labels.astype(int), preds, average="micro")}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 10. 학습 시작
trainer.train()