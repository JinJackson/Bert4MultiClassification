from transformers import BertTokenizer
from model.BertClassificationModel import BertForSequenceClassification
import torch
checkpoint_dir = 'bert-base-chinese'


tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
model = BertForSequenceClassification.from_pretrained(checkpoint_dir)

def predict(model, tokenizer, text):
    tokenized_dict = tokenizer.encode_plus(text=text, truncation=True, padding='max_length', max_length=500)
    input_ids = torch.tensor([tokenized_dict['input_ids']]).long()
    token_type_ids = torch.tensor([tokenized_dict['token_type_ids']]).long()
    attention_mask = torch.tensor([tokenized_dict['attention_mask']]).long()
    _, logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    pred_label = torch.argmax(logits, 1)
    return logits, pred_label
    #pred_label = torch.argmax(logits)


if __name__ == '__main__':
    text = '我今天真的好开心啊'
    logits, pred_label = predict(model, tokenizer, text)
    print(logits, pred_label)