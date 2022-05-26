from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class MeasurePredictor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.layer1 = nn.Linear(120, 120)
        self.layer2 = nn.Linear(120, 120)
        self.layer3 = nn.Linear(120, 120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    @staticmethod
    def load(f: str, device: str = "cuda") -> Tuple['MeasurePredictor', int, int]:
        save = torch.load(f, map_location=torch.device(device))
        model = MeasurePredictor()
        model.load_state_dict(save['model'])
        return model, save['epoch'], save['autoenc_version']
    
    def save(self, f: str, epoch: int, autoenc_version: int):
        torch.save({
            'model': self.state_dict(),
            "epoch": epoch,
            'autoenc_version': autoenc_version
        }, f)

class MeasurePredictorDataset(Dataset):
    def __init__(self, pieces: List[torch.Tensor]):
        self.measures = [(piece[i], piece[i+1]) for piece in pieces for i in range(len(piece)-1)]

    def __len__(self) -> int:
        return len(self.measures)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.measures[idx][0], self.measures[idx][1]

SAVE_FOLDER = "saves/predictor/trial-2"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the AutoEncoder')
    parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
    parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
    parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
    parser.add_argument("--learning-rate", type=float, help="Learning rate of Adam optimizer", default=1e-3)
    args = parser.parse_args()

    import rnn_preprocess
    pieces, autoenc_version = rnn_preprocess.load("saves/preprocessed-rnn.pt")
    train_ds = MeasurePredictorDataset([piece for i, piece in enumerate(pieces) if i % 11 != 1])
    test_ds = MeasurePredictorDataset([piece for i, piece in enumerate(pieces) if i % 11 == 1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.in_label:
        model, initial_epoch, autoenc_version_ = MeasurePredictor.load(SAVE_FOLDER + "/" + args.in_label + ".pt")
        assert autoenc_version_ == autoenc_version
    else:
        model = MeasurePredictor()
        model.to(device)
        initial_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for n_epoch in range(args.epochs):
        train_dl: DataLoader = DataLoader(train_ds, shuffle=True)
        test_dl: DataLoader = DataLoader(test_ds, shuffle=True)

        x: torch.Tensor
        y: torch.Tensor

        model.train()
        train_losses: List[float] = []
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        test_losses: List[float] = []
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            test_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)
        print("[E%d] Train: %f | Test: %f" % (n_epoch + initial_epoch +1, avg_train_loss, avg_test_loss))

    print("Trained " + str(args.epochs))
    model.save(SAVE_FOLDER + "/" + args.out_label + ".pt", initial_epoch + args.epochs, autoenc_version)
    print("Saved to " + args.out_label)
