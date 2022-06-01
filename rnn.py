from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import rnn_preprocess

class MeasurePredictor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dense1 = nn.Linear(120, 192)
        self.lstm = nn.LSTM(192, hidden_size=192, batch_first=True, num_layers=1)
        self.hidden2next = nn.Linear(192, 120)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x, _ = self.lstm(x)
        x = self.hidden2next(x)
        return x
    
    def predict(self, x: torch.Tensor, hidden: Union[Tuple[torch.Tensor, torch.Tensor], None]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.dense1(x)
        x, hidden2 = self.lstm(x.unsqueeze(0), hidden)
        x = self.hidden2next(x)
        x = x.detach()
        x.apply_(lambda x: max(min(x, 1), 0))
        return x, hidden2

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
        self.pieces = pieces

    def __len__(self) -> int:
        return len(self.pieces)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        piece = self.pieces[idx]
        return piece[:len(piece)-1], piece[1:]

SAVE_FOLDER = "saves/rnn/trial-4"

if __name__ == "__main__":
    import os
    try:
        os.makedirs(SAVE_FOLDER + "/stats")
    except FileExistsError: pass

    import argparse
    parser = argparse.ArgumentParser(description='Run the AutoEncoder')
    parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
    parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
    parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
    parser.add_argument("--learning-rate", type=float, help="Learning rate of Adam optimizer", default=1e-3)
    args = parser.parse_args()


    pieces, autoenc_version = rnn_preprocess.load("saves/preprocessed-rnn.pt")
    train_ds = MeasurePredictorDataset([piece for i, piece in enumerate(pieces) if i % 11 != 1])
    test_ds = MeasurePredictorDataset([piece for i, piece in enumerate(pieces) if i % 11 == 1])

    if args.in_label:
        model, initial_epoch, autoenc_version_ = MeasurePredictor.load(SAVE_FOLDER + "/" + args.in_label + ".pt")
        model = model.to("cuda")
        assert autoenc_version_ == autoenc_version
    else:
        model = MeasurePredictor()
        model.to("cuda")
        initial_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    losses: List[Tuple[int, float, float]] = []
    for n_epoch in range(args.epochs):
        train_dl: DataLoader = DataLoader(train_ds, shuffle=True)
        test_dl: DataLoader = DataLoader(test_ds, shuffle=True)

        x: torch.Tensor
        y: torch.Tensor

        model.train()
        train_losses: List[float] = []
        for x, y in train_dl:
            x = x.to("cuda")
            y = y.to("cuda")
            pred = model(x)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        test_losses: List[float] = []
        for x, y in test_dl:
            x = x.to("cuda")
            y = y.to("cuda")
            pred = model(x)
            loss = F.mse_loss(pred, y)
            test_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)
        print("[E%d] Train: %f | Test: %f" % (n_epoch + initial_epoch +1, avg_train_loss, avg_test_loss))
        losses.append((n_epoch + initial_epoch + 1, avg_train_loss, avg_test_loss))

    print("Trained " + str(args.epochs))
    model.save(SAVE_FOLDER + "/" + args.out_label + ".pt", initial_epoch + args.epochs, autoenc_version)
    print("Saved to " + args.out_label)

    import pandas
    epochs = list(map(lambda t: t[0], losses))
    train_losses = list(map(lambda t: t[1], losses))
    test_losses = list(map(lambda t: t[2], losses))
    df = pandas.DataFrame({"epoch": epochs, "loss": train_losses, "test_loss": test_losses})
    df.set_index("epoch")
    multi_csv_file = "/stats/epochs-" + str(initial_epoch + 1) + "-to-" + str(initial_epoch + args.epochs) + ".csv"
    df.to_csv(SAVE_FOLDER + multi_csv_file)