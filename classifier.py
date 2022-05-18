from typing import List, Sequence, Tuple
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
#        self.conv1 = torch.nn.Conv2d(1, 2, (4, 22))
#        self.pool1 = torch.nn.MaxPool2d((2, 2), (2, 2))
        self.dense1 = torch.nn.LazyLinear(100)
        self.dense2 = torch.nn.Linear(100, 100)
        self.dense3 = torch.nn.Linear(100, 19)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        x = self.conv1(x)
#        x = self.pool1(x)
        x = torch.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
    
data = list(preprocess.load("preprocessed-all.pt"))

# split into measures
data = [(a, b, c, measure) for (a, b, c, measures) in data for measure in measures]

print(data[0][3].shape)

n_composers = len(set(map(lambda x: x[2], data)))
print("composers: %d" % n_composers)

import random
shuffler = random.Random(55)
shuffler.shuffle(data)

train_data = data[:int(len(data)*7/8)]
test_data = data[int(len(data)*7/8):]
print("train measures: %d | test measures: %d" % (len(train_data), len(test_data)))

classifier = Classifier()
optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-7)
for sample_num, (name, composer, composer_idx, measure) in enumerate(train_data):
    x_tensor = measure.unsqueeze(0)
    y_tensor = F.one_hot(torch.tensor(composer_idx), n_composers).to(torch.float32)

    pred = classifier(x_tensor)
#    print("x: %s y: %s %s y-hat: %s %s" % (x_tensor.shape, y_tensor.shape, y_tensor.dtype, pred.shape, pred.dtype))
    loss = F.cross_entropy(pred.unsqueeze(0), y_tensor.unsqueeze(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if sample_num % 100 == 0 or sample_num == len(train_data) - 1 :
        print("[%d] loss: %f" % (sample_num, loss), end = "\n" if sample_num % 5000 == 0 or sample_num == len(train_data) - 1 else '\r')

def accuracy_on(data: List[Tuple[str, str, int, torch.Tensor]], label: str, color: str):
    correct = 0
    for sample_num, (name, composer, composer_idx, measure) in enumerate(data):
        pred = classifier(measure.unsqueeze(0))
        correct += int(torch.argmax(pred).item() == composer_idx)
    print("accuracy on %s data: %s%d/%d [%f%%]\033[0m" % (label, color, correct, len(data), correct/len(data)*100))

accuracy_on(test_data, "test", "\033[1m\033[34m")
accuracy_on(train_data, "train", "")