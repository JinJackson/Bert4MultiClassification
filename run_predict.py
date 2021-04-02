from transformers import BertTokenizer
from model.BertClassificationModel import BertForSequenceClassification
import torch
import sys


checkpoint_dir = 'bert-base-chinese'


tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
model = BertForSequenceClassification.from_pretrained(checkpoint_dir)

def predict(model, tokenizer, text):
    tokenized_dict = tokenizer.encode_plus(text=text, truncation=True, padding='max_length', max_length=500)
    input_ids = torch.tensor([tokenized_dict['input_ids']]).long()
    token_type_ids = torch.tensor([tokenized_dict['token_type_ids']]).long()
    attention_mask = torch.tensor([tokenized_dict['attention_mask']]).long()
    _, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    pred_label = torch.argmax(logits, 1).data
    return logits, pred_label
    #pred_label = torch.argmax(logits)


if __name__ == '__main__':
    labels = {'0':'喜悦', '1':'愤怒', '2':'厌恶', '3':'低落'}
    text = sys.argv[1]
    logits, pred_label = predict(model, tokenizer, text)
    print('四种感情的打分分别为：', str(logits.data))
    print('预测的情感结果为：',labels[str(int(pred_label.data))] )