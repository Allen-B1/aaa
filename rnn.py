import torch
import torch.nn as nn
import rnn_preprocess

class MeasurePredictor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.lstm = nn.LSTM(120, 384, batch_first=True)
        self.hidden2next = nn.Linear(384, 120)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.hidden2next(x)
        return x
