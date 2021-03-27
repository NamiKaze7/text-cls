import torch.nn as nn



class fn_cls(nn.Module):
    def __init__(self, bert):
        super(fn_cls, self).__init__()
        self.model = bert
        self.dropout = nn.Dropout(0.3)
        self.l1 = nn.Linear(768, 10)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        x = outputs[0][:, 0, :]  # 取池化后的结果 batch * 768
        x = x.view(-1, 768)
        x = self.dropout(x)
        x = self.l1(x)
        return x
