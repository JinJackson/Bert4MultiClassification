from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm


def readDatas(data_file):
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        texts = []
        labels = []
        for i in range(1, len(lines)):
            line = lines[i]
            line = line.replace(',', ' [SEP] ', 1)
            a_data = line.split(' [SEP] ')
            label, text = a_data
            texts.append(text)
            labels.append(label)
    return texts, labels

def SplitTrainTestData(texts, labels, test_size=0.2):
    train_X, test_X, train_Y, test_Y = train_test_split(texts, labels, test_size=test_size, random_state=1024)
    print(len(train_X), len(test_X), len(train_Y), len(test_Y))
    return train_X, test_X, train_Y, test_Y


class TrainData(Dataset):
    def __init__(self, train_X, train_Y, max_length, tokenizer):
        self.train_X = train_X
        self.train_Y = train_Y
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.train_X[index]
        label = int(self.train_Y[index])

        tokenized_dict = self.tokenizer(text=text, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = tokenized_dict['input_ids']
        token_type_ids = tokenized_dict['token_type_ids']
        attention_mask = tokenized_dict['attention_mask']
        return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask), np.array([label])

    def __len__(self):
        return len(self.train_X)
# if __name__ == '__main__':
#     data_file = 'data/simplifyweibo_4_moods.csv'
#     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#     texts, labels = readDatas(data_file)
#     train_X, test_X, train_Y, test_Y = SplitTrainTestData(texts, labels, test_size=0.2)
#     train_data = TrainData(test_X,
#                            train_Y,
#                            300,
#                            tokenizer=tokenizer)
#     train_dataloader = DataLoader(dataset=train_data,
#                                   batch_size=2,
#                                   shuffle=True)
#     for batch in train_dataloader:
#         print(batch)
#         break
    # max_length = 0
    # for text in tqdm(texts):
    #     tokenized_dict = tokenizer.encode_plus(text)
    #     input_ids = tokenized_dict['input_ids']
    #     max_length = max_length if len(input_ids) < max_length else len(input_ids)
    # print(max_length)