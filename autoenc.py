from abc import ABC, abstractmethod
from typing import List, Tuple, cast
import torch.nn as nn
import torch.nn.functional as F
import torch
import preprocess
import argparse
import random
import time
import os

VERSION = 13
SAVE_FOLDER = "saves/autoenc/trial-" + str(VERSION)

class AutoEncoder(nn.Module, ABC):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: pass
    @abstractmethod
    def decode_regularize(self, x: torch.Tensor) -> torch.Tensor: pass
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: pass
    @abstractmethod
    def version(self) -> int: pass

class AutoEncoderV13(AutoEncoder):
    def __init__(self):
        nn.Module.__init__(self)

        # [49, 88]
        # (48/4, 8) ; (48/12, 1)
        self.conv1 = nn.Conv2d(1, 8, (3, 3))
        self.conv2 = nn.Conv2d(8, 4, (4, 4))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(4 * 43 * 83, 120)
        self.dedense = nn.Linear(120, 4 * 43 * 83)
        self.deflatten = nn.Unflatten(1, (4, 43, 83))
        self.deconv1 = nn.ConvTranspose2d(4, 8, (4, 4))
        self.deconv2 = nn.ConvTranspose2d(8, 1, (3, 3))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
#        print(x.shape)
        x = F.elu(self.conv2(x))
#        print(x.shape)
        x = self.flatten(x)
        x = F.elu(self.dense(x))
#        print(x.shape)
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.dedense(x))
#        print(x.shape)
        x = self.deflatten(x)
        x = F.elu(self.deconv1(x))
#        print(x.shape)
        x = self.deconv2(x)
#        print(x.shape)
        return x
    
    def decode_regularize(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.decode(x))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x

    def version(self) -> int:
        return 13

class AutoEncoderV12(AutoEncoder):
    def __init__(self):
        nn.Module.__init__(self)

        # (48/4, 8) ; (48/12, 1)
        self.conv1 = nn.Conv2d(1, 8, (4, 4))
        self.conv2 = nn.Conv2d(8, 4, (4, 3), stride=(2, 1))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(4 * 22 * 83, 120)
        self.dedense = nn.Linear(120, 4 * 22 * 83)
        self.deflatten = nn.Unflatten(1, (4, 22, 83))
        self.deconv1 = nn.ConvTranspose2d(4, 8, (4, 3), stride=(2, 1))
        self.deconv2 = nn.ConvTranspose2d(8, 1, (4, 4))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
#        print(x.shape)
        x = F.elu(self.conv2(x))
#        print(x.shape)
        x = self.flatten(x)
        x = F.elu(self.dense(x))
#        print(x.shape)
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.dedense(x))
#        print(x.shape)
        x = self.deflatten(x)
        x = F.elu(self.deconv1(x))
#        print(x.shape)
        x = self.deconv2(x)
#        print(x.shape)
        return x
    
    def decode_regularize(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.decode(x))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x

    def version(self) -> int:
        return 12

class AutoEncoderV10(AutoEncoder):
    def __init__(self):
        nn.Module.__init__(self)

        self.hidden1 = nn.Linear(49 * 88, 512)
        self.code = nn.Linear(512, 120)
        self.hidden2 = nn.Linear(120, 512)
        self.output = nn.Linear(512, 49 * 88)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.Flatten()(x)
        x = F.leaky_relu(self.hidden1(x))
        x = F.leaky_relu(self.code(x))
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.hidden2(x))
        x = self.output(x)
        return x 

    def decode_regularize(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.decode(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def version(self) -> int:
        return 10

class AutoEncoderV11(AutoEncoder):
    def __init__(self):
        nn.Module.__init__(self)

        self.hidden1 = nn.Linear(49 * 88, 512)
        self.code = nn.Linear(512, 120)
        self.hidden2 = nn.Linear(120, 512)
        self.output = nn.Linear(512, 49 * 88)

        self.dropout = nn.Dropout(p=0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.Flatten()(x)
        x = F.leaky_relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.code(x))
        return x
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x
    
    def decode_regularize(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.decode(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.dropout(x)
        x = self.decode(x)
        return x

    def version(self) -> int:
        return 11
    
def new_version(version: int) -> AutoEncoder:
    if version == 10:
        return AutoEncoderV10()
    elif version == 11:
        return AutoEncoderV11()
    elif version == 12:
        return AutoEncoderV12()
    elif version == 13:
        return AutoEncoderV13()
    else:
        raise Exception("unknown autoencoder version")

def load(file: str, device: str='cuda') -> Tuple[AutoEncoder, int]:
    save = torch.load(file, map_location=torch.device(device))
    version = save['version']
    autoenc = new_version(version).to(device)
    autoenc.load_state_dict(save["model"])
    epoch = save["epoch"]
    return (autoenc, epoch)

def save(autoenc: AutoEncoder, epoch: int, file: str):
    torch.save({
        "model": autoenc.state_dict(),
        "epoch": epoch,
        "version": autoenc.version()
    }, file)


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    class AutoEncDataset(Dataset):
        def __init__(self, split: str):
            self.data = preprocess.load("saves/preprocessed.pt")
            filter = (lambda x: x % 11 != 0) if split == "train" else (lambda x: x % 11 == 0)
            self.data = [(a, b, c, measure.to("cuda" if torch.cuda.is_available() else "cpu"))
                for (a, b, c, measures) in self.data
                for measure_num, measure in enumerate(measures)
                if filter(measure_num)]

        def __len__(self) -> int:
            return len(self.data)
        
        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.data[idx][3]

    try:
        os.makedirs(SAVE_FOLDER + "/stats")
    except FileExistsError: pass

    parser = argparse.ArgumentParser(description='Run the AutoEncoder')
    parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
    parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
    parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
    parser.add_argument("--learning-rate", type=float, help="Learning rate of Adam optimizer", default=1e-4)
    args = parser.parse_args()

    if args.in_label is not None:
        autoenc, epoch_num = load(SAVE_FOLDER + "/" + args.in_label + ".pt", "cuda" if torch.cuda.is_available() else "cpu")
        assert autoenc.version() == VERSION
        print("Loading: " + args.in_label)
        print("Epoch: " + str(epoch_num))
    else:
        autoenc, epoch_num = new_version(VERSION), 0
        autoenc.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing new autoencoder")

    ds = AutoEncDataset("train")
    ds_test = AutoEncDataset("test")

    start_time = time.perf_counter()

    # training loop
    optimizer = torch.optim.Adam(autoenc.parameters(), lr=args.learning_rate)
    losses: List[Tuple[int, float, float]] = []
    for i in range(args.epochs):
        # train
        autoenc.train()
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        losses_within_epoch = []
        for batch_num, measures in enumerate(dl):
            pred = autoenc(measures)
            pred = torch.reshape(pred, (-1, 49, 88))
            loss = F.mse_loss(pred, measures)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_within_epoch.append(loss.item())

            print("[E%02d][B%03d] loss: %f" % (epoch_num + 1, batch_num, loss), end="\n" if batch_num % 100 == 0 else '\r')

        avg_train_loss = sum(losses_within_epoch) / len(losses_within_epoch)

        # test loss
        autoenc.eval()
        dl_test = DataLoader(ds_test)
        test_losses_within_epoch = []
        for measures in dl_test:
            pred = autoenc(measures)
            pred = torch.reshape(pred, (-1, 49, 88))
            loss = F.mse_loss(pred, measures)
            test_losses_within_epoch.append(loss.item())

        avg_test_loss = sum(test_losses_within_epoch) / len(test_losses_within_epoch)
        losses.append((epoch_num + 1, avg_train_loss, avg_test_loss))

        epoch_num += 1

    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    print("\n\033[32m\033[1mFinished training %d epochs in %fs\033[0m" % (args.epochs, time_elapsed))
    print("Epoch: " + str(epoch_num))
    save(autoenc, epoch_num, SAVE_FOLDER + "/" + args.out_label + ".pt")
    print("Saving to: " + args.out_label)

    # save stats file
    import pandas
    df = pandas.DataFrame({"loss": losses_within_epoch})
    df.to_csv(SAVE_FOLDER + "/stats/epoch-" + str(epoch_num) + ".csv")
    print("Saved stats file in stats/epoch-" + str(epoch_num) + ".csv")

    epochs = list(map(lambda t: t[0], losses))
    train_losses = list(map(lambda t: t[1], losses))
    test_losses = list(map(lambda t: t[2], losses))
    df = pandas.DataFrame({"epoch": epochs, "loss": train_losses, "test_loss": test_losses})
    df.set_index("epoch")
    multi_csv_file = "/stats/epochs-" + str(epoch_num - args.epochs + 1) + "-to-" + str(epoch_num) + ".csv"
    df.to_csv(SAVE_FOLDER + multi_csv_file)