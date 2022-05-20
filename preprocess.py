from typing import Dict, Literal, Sequence, Tuple, Union
import torch
from torch.utils.data.dataset import Dataset
import notes.tensor
import notes.mxl as mxl
import os.path
import glob

def preprocess(out_file: str):
    filenames = glob.glob("datasets/midide/*/*.musicxml")
    composers_list = list(set(map(lambda x: os.path.basename(os.path.dirname(x)), glob.glob("datasets/midide/*/*.musicxml"))))
    composers_dict: Dict[str, int] = dict(map(lambda t: (t[1], t[0]), enumerate(composers_list)))

    items = []
    for idx, file in enumerate(filenames):
        piece = mxl.parse_file(file)
        data = torch.stack([notes.tensor.to_tensor(measure) for measure in piece.measures])
        composer = os.path.basename(os.path.dirname(file))
        composer_idx = composers_dict[composer]
        name = os.path.splitext(os.path.basename(file))[0]
        items.append((name, composer, composer_idx, data))
        print("processed %d/%d" % (idx, len(filenames)), end='\r')

    print('done processing! saving to "%s"' % out_file)
    
    torch.save(items, out_file)

if __name__ == "__main__":
    preprocess("saves/preprocessed.pt")

def load(file: str) -> Sequence[Tuple[str, str, int, torch.Tensor]]:
    return torch.load(file)
