import torch
from model import fn_cls
import util
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser("Text-cls training task.")
parser.add_argument("--epoch", type=int, default= 10)
logger = util.create_logger("Bert Training", log_file='./logfile')
args = parser.parse_args()


def main():
    logger.info("Loading data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_loader = util.profile('train.txt', tokenizer)
    # test_loader = util.profile('test.txt', tokenizer)
    dev_loader = util.profile('dev.txt', tokenizer)
    logger.info("Build model...")
    bert = BertModel.from_pretrained('bert-base-chinese')
    model = fn_cls(bert)
    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    max_epoch = args.epoch
    model.cuda()
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(1, max_epoch + 1):
        tacc = []
        losslis = []
        # train epoch
        train_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()
            mask = []
            for sample in data:
                mask.append([1 if i != 0 else 0 for i in sample])
            mask = torch.tensor(mask).cuda()
            output = model(data, attention_mask=mask)
            pred = torch.argmax(output, 1)
            tacc.append(np.mean((pred == torch.argmax(target, 1)).cpu().numpy()))
            loss = criterion(sigmoid(output).view(-1, 10), target)
            losslis.append(loss.item())
            loss.backward()
            optim.step()
            optim.zero_grad()
        logger.info('Train Epoch: {} \tLoss:{:.6f} Acc:{}'.format(
            epoch, np.mean(losslis), 100 * np.mean(tacc)
        ))
        logger.info('time:{}'.format((time.time() - train_time)))
        # evaluate epoch
        model.eval()
        li = []
        acc = []
        for batch_idx, (data, target) in enumerate(dev_loader):
            data, target = torch.tensor(data).cuda(), torch.tensor(
                target).cuda()
            mask = []
            for sample in data:
                mask.append([1 if i != 0 else 0 for i in sample])
            mask = torch.tensor(mask).cuda()

            output = model(data, attention_mask=mask)
            pred = torch.argmax(output, 1)
            acc.append(np.mean((pred == target).cpu().numpy()))
            eval_loss = criterion(sigmoid(output).view(-1, 10), target)
            li.append(eval_loss.item())
        logger.info('-' * 50)
        logger.info('Validate Epoch: {} acc:{:.2f}%\tLoss:{:.6f}'.format(
            1, np.mean(acc) * 100, np.mean(li)))


if __name__ == "__main__":
    main()
