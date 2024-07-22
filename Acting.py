import torch
from transformers import BertForSequenceClassification,BertTokenizer,Trainer
import pandas as pd

model = BertForSequenceClassification.from_pretrained("./trained_model")
tokenizer = BertTokenizer.from_pretrained("./trained_model")

test_data = pd.read_csv("./COLDataset/COLDataset/test.csv")
test_data = test_data.head(1000)
test_texts = test_data["TEXT"].tolist()
test_labels = test_data["label"].tolist()

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return torch.argmax(predictions, dim=1).item()

acc = 0
for i in range(1000):
    prediction = predict(test_texts[i])
    if prediction == test_labels[i] :
        acc+=1
    print("预测标签:", "攻击性" if prediction == 1 else "温和性","正确" if prediction == test_labels[i] else "错误")

print(f"正确率为：{acc/1000}.4f")