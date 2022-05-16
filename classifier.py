import torch
from torch import nn
from torch.utils.data import DataLoader
import time

import dataset

class MusicClassifier(nn.Module):
    def __init__(self):
        super(MusicClassifier, self).__init__()
        self.linrelu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048*88, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linrelu(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = MusicClassifier().to(device)

train_set = dataset.Cache(dataset.MaestroDataset("train", dataset.NotesRepr2(1/32, 2048)))
test_set = dataset.MaestroDataset("test", dataset.NotesRepr2(1/32, 2048))

BATCH_SIZE = 4
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

print('starting training loop...')
optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-4)
for idx, (x, y) in enumerate(train_dataloader):
    start_time = time.time()
    
    y_onehot = torch.zeros(BATCH_SIZE, 64)
    for i, j in enumerate(y):
        y_onehot[i][j.item()] = 1
    x = torch.unsqueeze(x, 1)
    pred = classifier(x)
    loss: torch.Tensor = nn.MSELoss()(pred, y_onehot)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if idx % 1 == 0:
        acc = torch.sum(torch.argmax(pred, dim=1) == y) / BATCH_SIZE
        print("[%d] loss: %f | acc: %f | time: %fs" % (idx, loss.item(), acc*100, time.time() - start_time))
