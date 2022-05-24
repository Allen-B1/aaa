from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import preprocess
import argparse
import random
import time
import os

VERSION = 11
SAVE_FOLDER = "saves/autoenc/trial-" + str(VERSION)

class AutoEncoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        if VERSION == 10:
            self.hidden1 = nn.Linear(49 * 88, 512)
            self.code = nn.Linear(512, 120)
            self.hidden2 = nn.Linear(120, 512)
            self.output = nn.Linear(512, 49 * 88)
        elif VERSION == 11:
            self.hidden1 = nn.Linear(49 * 88, 512)
            self.code = nn.Linear(512, 120)
            self.hidden2 = nn.Linear(120, 512)
            self.output = nn.Linear(512, 49 * 88)

            self.dropout = nn.Dropout(p=0.1)


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
        if VERSION == 11:
            x = self.dropout(x)
        x = self.decode(x)
        return x

def load(file: str, device: str='cuda') -> Tuple[AutoEncoder, int]:
    save = torch.load(file, map_location=torch.device(device))
    if save['version'] != VERSION:
        raise Exception("expected v" + str(VERSION) + ", got v" + str(save['version']))
    
    autoenc = AutoEncoder().to(device)
    autoenc.load_state_dict(save["model"])
    epoch = save["epoch"]
    return (autoenc, epoch)

def save(autoenc: AutoEncoder, epoch: int, file: str):
    torch.save({
        "model": autoenc.state_dict(),
        "epoch": epoch,
        "version": VERSION
    }, file)


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    class AutoEncDataset(Dataset):
        def __init__(self):
            self.data = preprocess.load("saves/preprocessed.pt")
            self.data = [(a, b, c, measure.to("cuda" if torch.cuda.is_available() else "cpu")) for (a, b, c, measures) in self.data for measure in measures]

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
    args = parser.parse_args()

    if args.in_label is not None:
        autoenc, epoch_num = load(SAVE_FOLDER + "/" + args.in_label + ".pt")
        autoenc.train()
        print("Loading: " + args.in_label)
        print("Epoch: " + str(epoch_num))
    else:
        autoenc, epoch_num = AutoEncoder(), 0
        autoenc.to("cuda")
        print("Initializing new autoencoder")

    ds = AutoEncDataset()

    start_time = time.perf_counter()

    # training loop
    optimizer = torch.optim.Adam(autoenc.parameters(), lr=1e-4)
    losses_within_epoch: List[float] = []
    losses_epochs: List[Tuple[int, float]] = []
    for i in range(args.epochs):
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

        avg_loss = sum(losses_within_epoch) / len(losses_within_epoch)
        losses_epochs.append((epoch_num + 1, avg_loss))

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

    epochs = list(map(lambda t: t[0], losses_epochs))
    losses = list(map(lambda t: t[1], losses_epochs))
    df = pandas.DataFrame({"epoch": epochs, "loss": losses})
    multi_csv_file = "/stats/epochs-" + str(epoch_num - args.epochs + 1) + "-to-" + str(epoch_num) + ".csv"
    df.to_csv(SAVE_FOLDER + multi_csv_file)