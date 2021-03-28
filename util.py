import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def encod(x, tokenizer):
    tokenized_text = [tokenizer.tokenize(i) for i in x]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    for j in range(len(input_ids)):
        # 将样本数据填充至长度为 50
        i = input_ids[j]
        if len(i) != 50:
            input_ids[j].extend([0] * (50 - len(i)))
    return input_ids


def decod(x, tokenizer):
    def cstr(t):
        n = list(t).index(0)
        return t[:n]

    token_lis = [tokenizer.convert_ids_to_tokens(cstr(i)) for i in x]
    return token_lis


def get_dummies(l, size=10):
    res = list()
    for i in l:
        tmp = [0] * size
        tmp[int(i)] = 1
        res.append(tmp)
    return np.array(res)


def profile(filepath, tokenizer):
    x = []
    y = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, l in enumerate(f.readlines()):
            l = l.strip('\n').split('\t')
            x.append('[CLS]' + l[0] + '[SEP]')
            y.append(l[1])
        x = np.array(x)
        y = np.array(y)
        sset = TensorDataset(torch.LongTensor(encod(x,tokenizer)),
                             torch.FloatTensor(get_dummies(y)))
        loader = DataLoader(dataset=sset,
                            batch_size=20,
                            shuffle=True)
        return loader
