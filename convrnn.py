"""
convrnn - Predicts measure data directly instead of
	codes.
"""

from typing import List, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F
import torch

from notes.note import Measure

class MeasurePredictor(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.conv1 = nn.Conv2d(1, 8, (3, 3))
		self.conv2 = nn.Conv2d(8, 4, (4, 4))
		self.flatten = nn.Flatten()
		self.lstm = nn.LSTM(input_size=4 * 44 * 83, hidden_size=4 * 44 * 83, batch_first=True)
		self.deflatten = nn.Unflatten(1, (4, 44, 83))
		self.deconv1 = nn.ConvTranspose2d(4, 8, (4, 4))
		self.deconv2 = nn.ConvTranspose2d(8, 1, (3, 3))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.flatten(x)
		x, _ = self.lstm(x)
		x = self.deflatten(x)
		x = self.deconv1(x)
		x = self.deconv2(x)
		return x
	
	def predict(self, x: torch.Tensor, hidden: Union[Tuple[torch.Tensor, torch.Tensor], None]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.flatten(x)
		x, hidden_ = self.lstm(x, hidden)
		x = self.deflatten(x)
		x = self.deconv1(x)
		x = self.deconv2(x)
		return x, hidden_
	
def load(f: str, device:str='cuda') -> Tuple[MeasurePredictor, int]:
	s = torch.load(f, to=torch.device(device))
	model = MeasurePredictor().to(device)
	model.load_state_dict(s['model'])
	return model, s['epoch']

def save(f: str, model: MeasurePredictor, epochs: int):
	torch.save({
		"model": model.state_dict(),
		"epoch": epochs
	}, f)

SAVE_FOLDER = "saves/convrnn/trial-1"

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run the AutoEncoder')
	parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
	parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
	parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
	args = parser.parse_args()

	import preprocess
	pieces  = preprocess.load("saves/preprocessed.pt")

	from torch.utils.data import Dataset, DataLoader
	class PieceDataset(Dataset):
		def __init__(self, pieces: List[Tuple[str, str, int, torch.Tensor]]):
			self.pieces = [t[3] for t in pieces]
		
		def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
			return self.pieces[idx][:len(self.pieces[idx])-1].to("cuda"), self.pieces[idx][1:].to("cuda")
		
		def __len__(self) -> int:
			return len(self.pieces)

	train_ds = PieceDataset([piece for i, piece in enumerate(pieces) if i % 11 != 0])
	test_ds = PieceDataset([piece for i, piece in enumerate(pieces) if i % 11 == 0])

	if args.in_label is not None:
		model, epochs = load(SAVE_FOLDER + "/" + args.in_label + ".pt")
	else:
		model = MeasurePredictor()
		epochs = 0

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	
	losses: List[Tuple[int, float, float]] = []
	for n_epochs in range(args.epochs):
		train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
		test_dl = DataLoader(test_ds, batch_size=64, shuffle=True)

		train_losses = []
		for x, y in enumerate(train_dl):
			pred = model(x)
			loss = F.mse_loss(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_losses.append(loss.item())
		
		test_losses = []
		for x, y in enumerate(test_dl):
			pred = model(x)
			loss = F.mse_loss(pred, y)
			test_losses.append(loss.item())

		print("[E%d] train %f | test %f" % (epochs + n_epochs + 1, sum(train_losses) / len(train_losses), sum(test_losses) / len(test_losses)))
		losses.append((epochs + n_epochs + 1, sum(train_losses) / len(train_losses), sum(test_losses) / len(test_losses)))

	import pandas
	df = pandas.DataFrame({
		"epoch": [loss[0] for loss in losses],
		"loss": [loss[1] for loss in losses],
		"test_loss": [loss[2] for loss in losses],
	})
	df.to_csv(SAVE_FOLDER + "/stats/epochs-%d-to-%d.csv" % (epochs + 1, epochs + args.epochs + 1))