from typing import Dict, Literal, Sequence, Tuple, Union
import torch
from torch.utils.data.dataset import Dataset
import notes.tensor
from notes.note import Part
import notes.mxl as mxl
import os.path
import glob

def preprocess(out_file: str, n_measures: Union[int, Literal["all"]]):
    filenames = glob.glob("datasets/midide/*/*.musicxml")
    composers_list = list(set(map(lambda x: os.path.basename(os.path.dirname(x)), glob.glob("datasets/midide/*/*.musicxml"))))
    composers_dict: Dict[str, int] = dict(map(lambda t: (t[1], t[0]), enumerate(composers_list)))

    items = []
    for idx, file in enumerate(filenames):
        parts = mxl.parse_file(file)
        data = torch.stack([notes.tensor.to_tensor(measure) for measure in parts[0].measures] if n_measures == "all" else [notes.tensor.to_tensor(parts[0].measures[i]) for i in range(n_measures)])
        composer = os.path.basename(os.path.dirname(file))
        composer_idx = composers_dict[composer]
        name = os.path.splitext(os.path.basename(file))[0]
        items.append((name, composer, composer_idx, data))
        print("processed %d/%d" % (idx, len(filenames)), end='\r')

    print('done processing! saving to "%s"' % out_file)
    
    torch.save(items, out_file)

if __name__ == "__main__":
    preprocess("saves/preprocessed-all.pt", "all")

def load(file: str) -> Sequence[Tuple[str, str, int, torch.Tensor]]:
    return torch.load(file)
