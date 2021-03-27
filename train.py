import torch
from model import fn_cls
import util
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np


def main():
    train_loader = util.profile('train.txt')
    test_loader = util.profile('test.txt')
    dev_loader = util.profile('dev.txt')

    bert = BertModel.from_pretrained('bert-base-chinese')
    model = fn_cls(bert)
    criterion = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    max_epoch = 10
    model.cuda()
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(1, max_epoch + 1):
        tacc = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data.cuda()
            target.cuda()
            mask = []
            for sample in data:
                mask.append([1 if i != 0 else 0 for i in sample])
            mask = torch.tensor(mask).cuda()
            output = model(data, attention_mask=mask)
            pred = torch.argmax(output, 1)
            tacc.append(np.mean((pred == torch.argmax(target, 1)).cpu().numpy()))
            loss = criterion(sigmoid(output).view(-1, 10), target)
            loss.backward()
            optim.step()
            optim.zero_grad()
        print(np.mean(tacc))


if __name__ == "__main__":
    main()
