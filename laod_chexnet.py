import re
import torch
import torch.nn as nn
from torchvision import models

print("Script started")

ckpt_path = "/home/shakib/Desktop/S5/projet/chexnet/models/m-25012018-123527.pth.tar"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def build_chexnet(num_classes=14):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m

def load_chexnet_checkpoint(model, ckpt_path):
    print("Loading checkpoint:", ckpt_path)

    # IMPORTANT: charge d'abord sur CPU pour éviter blocages CUDA
    ckpt = torch.load(ckpt_path, map_location="cpu")

    print("Checkpoint type:", type(ckpt))
    print("Top keys:", list(ckpt.keys()))

    sd = ckpt["state_dict"]
    print("Original state_dict #params:", len(sd))

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module.densenet121."):
            k = k.replace("module.densenet121.", "", 1)
        if k.startswith("module."):
            k = k.replace("module.", "", 1)

        k = re.sub(r"\.norm\.(\d)\.", r".norm\1.", k)
        k = re.sub(r"\.conv\.(\d)\.", r".conv\1.", k)
        if k.startswith("classifier.0."):
            k = k.replace("classifier.0.", "classifier.", 1)
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if missing:
        print("Sample missing:", missing[:10])
    if unexpected:
        print("Sample unexpected:", unexpected[:10])

    return model

if __name__ == "__main__":
    model = build_chexnet(14)
    model = load_chexnet_checkpoint(model, ckpt_path)
    model = model.to(device).eval()
    print("Model loaded OK and moved to", device)
    import torch
    x = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
      y = model(x)
    print(y.shape)  