from re import M
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils_train as utilst

class MeasurePredictor(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ELU(),
            nn.Conv2d(32, 32, (4, 4), stride=(2,2)),
            nn.ELU(),
            nn.Conv2d(32, 32, (4, 4), stride=(2,2)),
            nn.ELU(),
        )
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(32*47*20, 32*21*40)
        self.unflatten = nn.Unflatten(1, (32, 21, 40))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, (4, 4),  stride=(2,2), output_padding=(0, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, (4, 4)),
            nn.ELU(),
            nn.ConvTranspose2d(32, 1, (3, 3))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = F.elu(self.dense(x))
        x = self.unflatten(x)
        x = self.deconv(x)
        return x


SAVE_FOLDER = "saves/convpredictor"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the Predictor')
    parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
    parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
    parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
    parser.add_argument("--learning-rate", type=float, help="Learning rate of Adam optimizer", default=1e-4)
    args = parser.parse_args()

    from torch.utils.data import Dataset, DataLoader
    class MeasuresDataset(Dataset):
        def __init__(self, split: str):
            import preprocess
            pieces = preprocess.load("saves/preprocessed.pt")
            f = (lambda x: x % 11 != 0) if split == "train" else (lambda x: x % 11 == 0)
            self.measures = [(torch.cat((piece[i], piece[i+1], piece[i+2], piece[i+3])), piece[i+4]) for a, b, c, piece in pieces for i in range(len(piece)-5) if f(i)]

        def __len__(self) -> int:
            return len(self.measures)
        
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.measures[idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.in_label is not None:
        model, initial_epoch = utilst.load_model(SAVE_FOLDER, args.in_label, MeasurePredictor(), device=device)
    else:
        model = MeasurePredictor().to(device)
        initial_epoch = 0

    train_ds = MeasuresDataset("train")
    test_ds = MeasuresDataset("test")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses: List[utilst.Loss] = []
    for n_epochs in range(args.epochs):
        train_dl = DataLoader(train_ds, batch_size=64)
        test_dl = DataLoader(test_ds, batch_size=64)

        train_losses: List[float] = []
        for x, y in train_dl:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            pred = model(x)
            print(pred.shape)
            assert pred.shape == y.shape
            loss = F.mse_loss(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        test_losses: List[float] = []
        for x, y in test_dl:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            test_losses.append(float(loss.item()))

        lossobj = utilst.Loss(epoch=initial_epoch+n_epochs+1, train_loss=sum(train_losses) / len(train_losses), test_loss=sum(test_losses)/len(test_losses))
        losses.append(lossobj)
        utilst.print_loss(lossobj)
    
    utilst.save_model(SAVE_FOLDER, args.out_label, model, initial_epoch+args.epochs)
    utilst.save_losses(SAVE_FOLDER + "/stats", losses)