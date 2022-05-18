from tkinter import Y
from typing import Sequence
import torch.nn.functional as F
import torch.nn, torch.optim
from torch.utils.data import DataLoader, Dataset
import notes.tensor
from notes.note import Part
import preprocess
import torch

class Classifier(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.conv1 = torch.nn.Conv2d(1, 4, (4, 22))
        self.pool1 = torch.nn.MaxPool2d((2, 2), (2, 2))
        self.dense1 = torch.nn.LazyLinear(120)
        self.dense2 = torch.nn.Linear(120, 19)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.flatten(x)
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x
    
data = list(preprocess.load("preprocessed.pt"))

n_composers = len(set(map(lambda x: x[2], data)))
print("composers: %d" % n_composers)

import random
shuffler = random.Random(50)
shuffler.shuffle(data)

train_data = data[:int(len(data)*2/3)]
test_data = data[int(len(data)*2/3):]
print("train samples: %d | test samples: %d" % (len(train_data), len(test_data)))

classifier = Classifier()
optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-5)
for sample_num, (name, composer, composer_idx, measures) in enumerate(train_data):
    x_tensor = measures.unsqueeze(0)
    y_tensor = F.one_hot(torch.tensor(composer_idx), n_composers).to(torch.float32)

    pred = classifier(x_tensor)
#    print("x: %s y: %s %s y-hat: %s %s" % (x_tensor.shape, y_tensor.shape, y_tensor.dtype, pred.shape, pred.dtype))
    loss = F.cross_entropy(pred.unsqueeze(0), y_tensor.unsqueeze(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if sample_num % 10 == 0:
        print("[%d] loss: %f" % (sample_num, loss))

correct = 0
for sample_num, (name, composer, composer_idx, measures) in enumerate(test_data):
    pred = classifier(measures.unsqueeze(0))
    correct += int(torch.argmax(pred).item() == composer_idx)
print("total accuracy: %d/%d [%f%%]" % (correct, len(test_data), correct/len(test_data)*100))