from typing import List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch
import preprocess
import argparse
import random
import time
import os

SAVE_FOLDER = "saves/autoenc/trial-6"

class AutoEncoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Linear(49 * 88, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 120),
        )

        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(120, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 49 * 88)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(F.dropout(self.encoder(torch.flatten(x))))

    def get_code(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(torch.flatten(x)))
    
if __name__ == "__main__":
    try:
        os.makedirs(SAVE_FOLDER + "/stats")
    except FileExistsError: pass

    parser = argparse.ArgumentParser(description='Run the AutoEncoder')
    parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
    parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
    parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
    args = parser.parse_args()

    data = preprocess.load("saves/preprocessed.pt")
    data = [(a, b, c, measure.to("cuda" if torch.cuda.is_available() else "cpu")) for (a, b, c, measures) in data for measure in measures]

    autoenc = AutoEncoder().to("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num: int = 0 # number of epochs, before the process
    if args.in_label is not None:
        save = torch.load(SAVE_FOLDER + "/" + args.in_label + ".pt")
        autoenc.load_state_dict(save["model"])
        epoch_num = save["epoch"]
        print("Loading: " + args.in_label)
        print("Epoch: " + str(epoch_num))

    start_time = time.perf_counter()

    optimizer = torch.optim.Adam(autoenc.parameters(), lr=1e-4)
    losses_within_epoch: List[float] = []
    losses_epochs: List[Tuple[int, float]] = []
    for i in range(args.epochs):
        random.shuffle(data)
        losses_within_epoch = []
        for iter_num, (name, composer, c, measure) in enumerate(data):
            pred = autoenc(measure)
            pred = torch.reshape(pred, (49, 88))
            loss = F.mse_loss(pred, measure)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_within_epoch.append(loss.item())

            if iter_num % 100 == 0:
                print("[E%d][%d] loss: %f" % (epoch_num + 1, iter_num, loss), end="\n" if iter_num % 1000 == 0 else '\r')

        avg_loss = sum(losses_within_epoch) / len(losses_within_epoch)
        losses_epochs.append((epoch_num + 1, avg_loss))

        epoch_num += 1

    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    print("\n\033[32m\033[1mFinished training %d epochs in %fs\033[0m" % (args.epochs, time_elapsed))
    print("Epoch: " + str(epoch_num))
    torch.save({"model": autoenc.state_dict(), "epoch": epoch_num}, SAVE_FOLDER + "/" + args.out_label + ".pt")
    print("Saving to: " + args.out_label + ".pt")

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