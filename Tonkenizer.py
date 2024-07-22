import pandas as pd
import torch
import torch.utils
import torch.utils.data
from transformers import BertTokenizer,BertForSequenceClassification,Trainer,TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv("./COLDataset/COLDataset/train.csv")
test_data = pd.read_csv("./COLDataset/COLDataset/test.csv")
train_text = train_data['TEXT'].tolist()
train_label = train_data['label'].tolist()
test_text = test_data['TEXT'].tolist()
test_label = test_data['label'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts,padding=True, truncation=True, max_length=512)

train_encodings = tokenize_function(train_text)
test_encodings = tokenize_function(test_text)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        item = {key:torch.tensor(val[index])for key,val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item
    
    def __len__(self):
        return len(self.labels)
    
train_dataset = CustomDataset(train_encodings,train_label)
test_dataset = CustomDataset(test_encodings,test_label)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels =2)
model = model.to(device)
train_args = TrainingArguments(
    output_dir= "./result",
    num_train_epochs= 5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps= 1000,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy= "epoch",
    load_best_model_at_end=True,
)
metric = load_metric("accuracy")

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return metric.compute(predictions=preds, references=p.label_ids)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
