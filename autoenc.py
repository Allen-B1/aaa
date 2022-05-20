import torch.nn as nn
import torch.nn.functional as F
import torch
import preprocess
import argparse
import random
import time

SAVE_FOLDER = "saves/autoenc/trial-2"

class AutoEncoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(
            nn.Linear(49 * 88, 384),
            nn.ReLU(),
            nn.Linear(384, 120),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(120, 384),
            nn.ReLU(),
            nn.Linear(384, 49 * 88)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(torch.flatten(x)))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the AutoEncoder')
    parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
    parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
    parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
    args = parser.parse_args()

    data = preprocess.load("saves/preprocessed-all.pt")
    data = [(a, b, c, measure.to("cuda" if torch.cuda.is_available() else "cpu")) for (a, b, c, measures) in data for measure in measures]

    autoenc = AutoEncoder().to("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num: int = 0 # number of epochs, before the process
    if args.in_label is not None:
        save = torch.load(SAVE_FOLDER + "/" + args.in_label + ".pt")
        autoenc.load_state_dict(save["model"])
        epoch_num: int = save["epoch"]
        print("Loading: " + args.in_label)
        print("Epoch: " + str(epoch_num))

    start_time = time.perf_counter()

    optimizer = torch.optim.Adam(autoenc.parameters(), lr=1e-4)
    for i in range(args.epochs):
        random.shuffle(data)
        losses = []
        for iter_num, (name, composer, c, measure) in enumerate(data):
            pred = autoenc(measure)
            pred = torch.reshape(pred, (49, 88))
            loss = F.mse_loss(pred, measure)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_num % 100 == 0:
                losses.append(loss.item())
                print("[E%d][%d] loss: %f" % (epoch_num + 1, iter_num, loss), end="\n" if iter_num % 1000 == 0 else '\r')

        epoch_num += 1


    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    print("\n\033[32m\033[1mFinished training %d epochs in %fs\033[0m" % (args.epochs, time_elapsed))
    print("Epoch: " + str(epoch_num))
    torch.save({"model": autoenc.state_dict(), "epoch": epoch_num}, SAVE_FOLDER + "/" + args.out_label + ".pt")
    print("Saving to: " + args.out_label + ".pt")