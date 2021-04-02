from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from data_preprocess import readDatas, SplitTrainTestData
from data_preprocess import TrainData
import os
import torch
from tqdm import tqdm
import numpy as np
from utils.logger import getLogger
import glob
from model.BertClassificationModel import BertForSequenceClassification


data_file = 'data/simplifyweibo_4_moods.csv'
texts, labels = readDatas(data_file)



do_train = True
epochs = 4
learning_rate = 2e-5
adam_epsilon=1e-8
warmup = 0.1
max_length = 400
batch_size = 2
save_dir = 'result/model'
bert_model = 'bert-base-chinese'
logger = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, tokenizer, checkpoint):

    texts, labels = readDatas(data_file)
    train_X, test_X, train_Y, test_Y = SplitTrainTestData(texts, labels, test_size=0.2)
    train_data = TrainData(train_X,
                           train_Y,
                           max_length=max_length,
                           tokenizer=tokenizer)
    train_dataLoader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    # 初始化 optimizer，scheduler
    t_total = len(train_dataLoader) * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)

    warmup_steps = warmup * t_total
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )


    # 读取断点 optimizer、scheduler
    checkpoint_dir = save_dir + "/checkpoint-" + str(checkpoint)
    if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))

    # 开始训练
    logger.debug("***** Running training *****")
    logger.debug("  Num examples = %d", len(train_dataLoader))
    logger.debug("  Num Epochs = %d", epochs)
    logger.debug("  Batch size = %d", batch_size)

    # 没有历史断点，则从0开始
    if checkpoint < 0:
        checkpoint = 0
    else:
        checkpoint += 1
    logger.debug("  Start Batch = %d", checkpoint)
    for epoch in range(checkpoint, epochs):
        model.train()
        epoch_loss = []


        for batch in tqdm(train_dataLoader, desc="Iteration"):
            model.zero_grad()
            # 设置tensor gpu运行
            batch = tuple(t.to(device) for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch

            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            labels=labels)

            loss = outputs[0]

            loss.backward()  # 计算出梯度
            epoch_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

            # 保存模型
        output_dir = save_dir + "/checkpoint-" + str(epoch)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        # torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.debug("Saving model checkpoint to %s", output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.debug("Saving optimizer and scheduler states to %s", output_dir)

        # TODO to be done
        # eval test
        test_loss, test_acc = test(model, tokenizer, test_X, test_Y, checkpoint=epoch)


        logger.info('【TEST】Train Epoch %d: train_loss=%.4f, acc=%.4f' % (epoch, test_loss, test_acc))

def test(model, tokenizer, test_X, test_Y, checkpoint):
        # eval数据处理：eval可能是test或者dev?
    test_data = TrainData(train_X=test_X,
                          train_Y=test_Y,
                          max_length=max_length,
                          tokenizer=tokenizer
                        )

    test_dataLoader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    logger.debug("***** Running evaluation {} *****".format(checkpoint))
    logger.debug(" Num examples = %d", len(test_dataLoader))
    logger.debug(" Batch size = %d", batch_size)

    loss = []

    all_labels = None
    all_logits = None
    model.eval()

    for batch in tqdm(test_dataLoader, desc="testing"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, token_type_ids, attention_mask, labels = batch

        with torch.no_grad():
            outputs = model(input_ids=input_ids.long(),
                            token_type_ids=token_type_ids.long(),
                            labels=labels)

            eval_loss, logits = outputs[:2]

            loss.append(eval_loss.item())

            if all_labels is None:
                all_labels = labels.detach().cpu().numpy()
                all_logits = logits.detach().cpu().numpy()

            else:
                all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

    all_pred = np.argmax(all_logits, 1)
    all_labels = all_labels.squeeze(axis=1)
    results = (all_pred == all_labels)
    acc = results.sum() / len(all_pred)
    return np.array(loss).mean(), acc


if __name__ == "__main__":

    # 创建存储目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger = getLogger(__name__, os.path.join(save_dir, 'log.txt'))
    if do_train:
        # train： 接着未训练完checkpoint继续训练
        checkpoint = -1
        for checkpoint_dir_name in glob.glob(save_dir + "/*"):
            try:
                checkpoint = max(checkpoint, int(checkpoint_dir_name.split('/')[-1].split('-')[1]))
            except Exception as e:
                pass
        checkpoint_dir = save_dir + "/checkpoint-" + str(checkpoint)
        if checkpoint > -1:
            logger.debug("Load Model from {}".format(checkpoint_dir))
        tokenizer = BertTokenizer.from_pretrained(bert_model if checkpoint == -1 else checkpoint_dir)
        model = BertForSequenceClassification.from_pretrained(bert_model if checkpoint == -1 else checkpoint_dir)
        model.to(device)
        # 训练
        train(model, tokenizer, checkpoint)