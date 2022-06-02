from typing import List, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim

SAVE_FOLDER = "saves/multiautoenc/trial-1"

class MultiAutoEncoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.measure_encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3)),
            nn.ELU(),
            nn.Conv2d(8, 4, (4, 4)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(4 * 44 * 83, 120)
        )

        self.section_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120 * 16, 384),
            nn.ELU(),
            nn.Linear(384, 96),
            nn.Sigmoid()
        )

        self.section_decoder = nn.Sequential(
            nn.Linear(96, 384),
            nn.ELU(), 
            nn.Linear(384, 120 * 16),
            nn.ELU(),
            nn.Unflatten(1, (16, 120)),
        )

        self.measure_decoder = nn.Sequential(
            nn.Linear(120, 4 * 44 * 83),
            nn.Unflatten(1, (4, 44, 83)),
            nn.ConvTranspose2d(4, 8, (4, 4)),
            nn.ConvTranspose2d(8, 1, (3, 3))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        measures = torch.split(x, 1, dim=1)
        measure_codes = [self.measure_encoder(measure) for measure in measures]
        codes = self.section_encoder(torch.stack(measure_codes, dim=1))
        measure_codes = self.section_decoder(codes)
        return torch.stack([self.measure_decoder(codes.squeeze(1)) for codes in torch.split(measure_codes, 1, dim=1)], dim=1).squeeze(2)

def load(file: str, device:str='cuda') -> Tuple[MultiAutoEncoder, int]:
    save = torch.load(file, map_location=torch.device(device))
    autoenc = MultiAutoEncoder().to(device)
    autoenc.load_state_dict(save["model"])
    epoch = save["epoch"]
    return (autoenc, epoch)

def save(autoenc: MultiAutoEncoder, epoch: int, file: str):
    torch.save({
        "model": autoenc.state_dict(),
        "epoch": epoch
    }, file)

if __name__ == "__main__":
    import os
    try:
        os.makedirs(SAVE_FOLDER + "/stats")
    except FileExistsError: pass

    from torch.utils.data import Dataset, DataLoader
    import preprocess
    class Multiset(Dataset):
        def __init__(self, split: str):
            self.data = preprocess.load("saves/preprocessed.pt")
            filter = (lambda x: x % 11 != 0) if split == "train" else (lambda x: x % 11 == 0)
            self.data = [(a, b, c, piece[i:i+16]) for (a, b, c, piece) in self.data for i in range(len(piece)-15)]

        def __len__(self) -> int:
            return len(self.data)
        
        def __getitem__(self, idx: int) -> torch.Tensor:
            return self.data[idx][3]

    import argparse
    parser = argparse.ArgumentParser(description='Run the AutoEncoder')
    parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
    parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
    parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
    parser.add_argument("--learning-rate", type=float, help="Learning rate of Adam optimizer", default=1e-4)
    args = parser.parse_args()

    if args.in_label is not None:
        model, epoch_init = load(SAVE_FOLDER + "/" + args.in_label + ".pt")
        model = model.to("cuda")
        print("Loading: " + args.in_label)
        print("Epoch: " + str(epoch_init))
    else:
        model, epoch_init = MultiAutoEncoder(), 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_ds = Multiset("train")
    test_ds = Multiset("test")

    losses: List[Tuple[int, float, float]] = []
    for n_epoch in range(args.epochs):
        train_dl = DataLoader(train_ds, batch_size=64)
        test_dl = DataLoader(test_ds, batch_size=64)

        train_losses:  List[float] = []
        for batch_num, measure_sets in enumerate(train_dl):
            measure_sets = measure_sets.to("cuda")
#            print(measure_sets.shape)
            pred = model(measure_sets)
#            print(pred.shape)
            loss = F.mse_loss(pred, measure_sets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            print("[E%d][B%d]" % (n_epoch + epoch_init + 1, batch_num) + " Train: %f" % loss.item(), end='\r')

        test_losses: List[float] = []
        for measure_sets in test_dl:
            measure_sets = measure_sets.to("cuda")
#            print(measure_sets.shape)
            pred = model(measure_sets)
#            print(pred.shape)
            loss = F.mse_loss(pred, measure_sets)
            test_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)
        print("[E%d] Train: %f | Test: %f" % (n_epoch + epoch_init +1, avg_train_loss, avg_test_loss))
        losses.append((n_epoch + epoch_init + 1, avg_train_loss, avg_test_loss))

    save(model,  epoch_init + args.epochs, SAVE_FOLDER + "/" + args.out_label + ".pt")

    import pandas
    epochs = list(map(lambda t: t[0], losses))
    train_losses = list(map(lambda t: t[1], losses))
    test_losses = list(map(lambda t: t[2], losses))
    df = pandas.DataFrame({"epoch": epochs, "loss": train_losses, "test_loss": test_losses})
    df.set_index("epoch")
    multi_csv_file = "/stats/epochs-" + str(epoch_init + 1) + "-to-" + str(epoch_init + args.epochs) + ".csv"
    df.to_csv(SAVE_FOLDER + multi_csv_file)
