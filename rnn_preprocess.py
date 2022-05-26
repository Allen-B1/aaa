import torch
from typing import List, Tuple
import autoenc
import preprocess

def load(f: str) -> Tuple[List[torch.Tensor], int]:
    save = torch.load(f)
    return save["measures"], save["autoenc_version"]

if __name__ == "__main__":
    MODEL = autoenc.SAVE_FOLDER + "/model-2000.pt"

    print("Loading model...")
    autoenc_model, epochs = autoenc.load(MODEL, "cpu")
    autoenc_model.eval()
    with torch.no_grad():
        pieces = preprocess.load("saves/preprocessed.pt")
        processed: List[torch.Tensor] = []
        for n, (a, b, c, tensor) in enumerate(pieces):
            print("Encoding %d/%d items..." % (n, len(pieces)), end='\r')
            x = autoenc_model.encode(tensor)
            assert x.shape[1] == 120
            processed.append(x)

        print("Done processing!")

        torch.save({
            "measures": processed,
            "autoenc_version": autoenc_model.version()
        }, "saves/preprocessed-rnn.pt")