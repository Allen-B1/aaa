from typing import List, NamedTuple, Sequence, TypeVar, Tuple
import torch
import torch.nn as nn
import os
import pandas

def save_model(save_folder: str, label: str, model: nn.Module, epoch: int):
    try:
        os.makedirs(save_folder)
    except FileExistsError: pass

    filename = save_folder + "/" + label
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch
    }, filename)

M = TypeVar('M', bound=nn.Module)
def load_model(save_folder: str, label: str, model: M, device: str='cuda') -> Tuple[M, int]:
    filename = save_folder + "/" + label
    dict = torch.load(filename, map_location=torch.device(device))
    model.load_state_dict(dict["model"])
    model.to(device=torch.device(device))
    return model, dict["epoch"]

class Loss(NamedTuple):
    epoch: int
    train_loss: float
    test_loss: float
Losses = Sequence[Loss]

def print_loss(loss: Loss):
    print("[E%d] Train: %f | Test: %f" % (loss.epoch, loss.train_loss, loss.test_loss))

def save_losses(stats_folder: str, losses: Losses):
    try:
        os.makedirs(stats_folder)
    except FileExistsError: pass


    losses = list(losses)
    losses.sort(key=lambda t: t.epoch)
    df = pandas.DataFrame({
        "epoch": [t.epoch for t in losses],
        "loss": [t.train_loss for t in losses],
        "test_loss": [t.test_loss for t in losses],
    })
    df.to_csv(stats_folder + "/epochs-%d-to-%d.csv" % (losses[0].epoch, losses[len(losses)-1].epoch))