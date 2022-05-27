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
		self.flatten = nn.Flatten()
		self.dense = nn.Linear(49*88, 512)
		self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
		self.dedense = nn.Linear(512, 49*88)

	# input: tensor [-1, 49, 88]
	# output: tensor [-1, 49, 88]
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.flatten(x)
		x = F.leaky_relu(self.dense(x))
		x, _ = self.lstm(x)
		x = F.leaky_relu(x)
		x = F.leaky_relu(self.dedense(x))
		x = torch.reshape(x, (-1, 49, 88))
		return x
	
	# predict next measure given one measure
	# input: tensor [49, 88]
	# output: tensor [49, 88]
	def predict(self, x: torch.Tensor, hidden: Union[Tuple[torch.Tensor, torch.Tensor], None]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
		x = self.flatten(x)
		x = F.leaky_relu(self.dense(x))
		x, hidden_ = self.lstm(x, hidden)
		x = F.leaky_relu(x)
		x = F.leaky_relu(self.dedense(x))
		x = torch.reshape(x, (49, 88))
		return x, hidden_
	
def load(f: str, device:str='cuda') -> Tuple[MeasurePredictor, int]:
	s = torch.load(f, map_location=torch.device(device))
	model = MeasurePredictor()
	model.load_state_dict(s['model'])
	model = model.to(device)
	return model, s['epoch']

def save(f: str, model: MeasurePredictor, epochs: int):
	torch.save({
		"model": model.state_dict(),
		"epoch": epochs
	}, f)

SAVE_FOLDER = "saves/convrnn/trial-3"

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Run the AutoEncoder')
	parser.add_argument("--epochs", type=int, help="Number of epochs to train (default: 1)", default=1)
	parser.add_argument("--in-label", type=str, help="Model label to resume from", default=None)
	parser.add_argument("--out-label", type=str, help="Model label to write to", required=True)
	args = parser.parse_args()

	import preprocess
	pieces  = preprocess.load("saves/preprocessed.pt")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	from torch.utils.data import Dataset, DataLoader
	class PieceDataset(Dataset):
		def __init__(self, pieces: List[Tuple[str, str, int, torch.Tensor]]):
			self.pieces = [t[3] for t in pieces]
		
		def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
			return self.pieces[idx][:len(self.pieces[idx])-1].to(device), self.pieces[idx][1:].to(device)
		
		def __len__(self) -> int:
			return len(self.pieces)

	train_ds = PieceDataset([piece for i, piece in enumerate(pieces) if i % 11 != 0])
	test_ds = PieceDataset([piece for i, piece in enumerate(pieces) if i % 11 == 0])

	if args.in_label is not None:
		model, epochs = load(SAVE_FOLDER + "/" + args.in_label + ".pt", device.type)
	else:
		model = MeasurePredictor().to(device)
		epochs = 0

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
	
	losses: List[Tuple[int, float, float]] = []
	for n_epochs in range(args.epochs):
		train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
		test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

		model.train()
		train_losses = []
		for x, y in train_dl:
			x = torch.squeeze(x).to(device)
			y = torch.squeeze(y).to(device)
			pred = model(x)
			loss = F.mse_loss(pred, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_losses.append(loss.item())
		
		model.eval()
		test_losses = []
		for x, y in test_dl:
			x = torch.squeeze(x).to(device)
			y = torch.squeeze(y).to(device)
			pred = model(x)
			loss = F.mse_loss(pred, y)
			test_losses.append(loss.item())

		print("[E%d] train %f | test %f" % (epochs + n_epochs + 1, sum(train_losses) / len(train_losses), sum(test_losses) / len(test_losses)))
		losses.append((epochs + n_epochs + 1, sum(train_losses) / len(train_losses), sum(test_losses) / len(test_losses)))

	save(SAVE_FOLDER + "/" + args.out_label + ".pt", model, epochs + args.epochs)

	import pandas
	df = pandas.DataFrame({
		"epoch": [loss[0] for loss in losses],
		"loss": [loss[1] for loss in losses],
		"test_loss": [loss[2] for loss in losses],
	})
	df.to_csv(SAVE_FOLDER + "/stats/epochs-%d-to-%d.csv" % (epochs + 1, epochs + args.epochs))