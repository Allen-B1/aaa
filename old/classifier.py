import itertools
import torch
from torch import nn
from torch.utils.data import DataLoader
import time

import notes
import notes.repr

class MusicClassifier(nn.Module):
    def __init__(self):
        super(MusicClassifier, self).__init__()
        self.linrelu = nn.Sequential(
            nn.Conv2d(1, 2, (6, 88)),
            nn.MaxPool2d((2, 1), 1),
            nn.Conv2d(2, 2, (12, 1)),
            nn.MaxPool2d((4, 1), 2),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linrelu(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
classifier = MusicClassifier().to(device)

FILE = "maestro-NotesRepr2_invstep2_max512.pt"
train_set = notes.repr.SavedMaestroDataset(FILE, "train")
test_set = notes.repr.SavedMaestroDataset(FILE, "test")

BATCH_SIZE = 64
train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

print('starting training loop...')
optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-7)
for idx, (x, y) in enumerate(train_dataloader):
    start_time = time.time()
    
    y_onehot = torch.zeros(len(y), 64)
    for i, j in enumerate(y):
        y_onehot[i][j.item()] = 1
    x = torch.unsqueeze(x, 1)
    x = torch.tensor(x, dtype=torch.float32)
    pred = classifier(x)
    loss: torch.Tensor = nn.CrossEntropyLoss()(pred, y_onehot)

#    print(pred)
#    print(y_onehot)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if idx % 1 == 0:
        acc = torch.sum(torch.argmax(pred, dim=1) == y) / BATCH_SIZE
        print("[%d] loss: %f | acc: %f%% | time: %fs" % (idx, loss.item(), acc*100, time.time() - start_time))

print("\033[33;1m-- Testing --\033[0m")

test_correct = 0
for (x, y) in test_dataloader:
    y_onehot = torch.zeros(len(y), 64)
    for i, j in enumerate(y):
        y_onehot[i][j.item()] = 1
    x = torch.unsqueeze(x, 1)
    x = torch.tensor(x, dtype=torch.float32)
    pred = classifier(x)
    test_correct += int(torch.sum(torch.argmax(pred, dim=1) == y).item())

print("Total Correct: %d / %d [%f%%]" % (test_correct, len(test_set), test_correct / len(test_set) * 100))