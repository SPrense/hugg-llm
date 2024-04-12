import torch.nn as nn

class MutiCls(nn.Module):
    def __init__(
        self,
        input_size=768,
        n_labels = 10
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_size,n_labels)
        
    def forward(self,x):
        x = self.linear1(x)
        x = x.softmax(dim=1)
        return x