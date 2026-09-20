#!/usr/bin/env python
"""experiment_3 label smoke: zero-shot CLIP top-1 on the iNat21 val split
(100k images, 10 per species) for every label naming x template set, with
VILA's CLIP (open_clip ViT-B-16, laion400m_e32 weights from ./ckpt, the
inc_net.py load) and open_clip's own image preprocess.  Image features are
computed once (cached to <out>/val_clip_feats.pt); text features per cell
follow vila.py _feat_from_temp (mean over templates, L2-normalised).
  python clip_label_smoke.py --root data/inat21 --out data/feature_cache/inat21_label_smoke
"""
import argparse, json, os, time
import torch, open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/inat21")
ap.add_argument("--out", default="data/feature_cache/inat21_label_smoke")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--workers", type=int, default=12)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)
sd = torch.load("./ckpt/timm/vit_base_patch16_clip_224.laion400m_e32/open_clip_pytorch_model.bin")
print(model.load_state_dict(sd))
model = model.to(dev).eval()
tok = open_clip.get_tokenizer("ViT-B-16")

val = json.load(open(os.path.join(args.root, "val.json")))
img2file = {im["id"]: im["file_name"] for im in val["images"]}
items = [(os.path.join(args.root, "2021_valid", img2file[a["image_id"]][len("val/"):]),
          int(a["category_id"])) for a in val["annotations"]]
assert len(items) == 100000

class DS(Dataset):
    def __len__(self): return len(items)
    def __getitem__(self, i):
        p, y = items[i]
        return preprocess(Image.open(p).convert("RGB")), y

fp = os.path.join(args.out, "val_clip_feats.pt")
if os.path.exists(fp):
    F_img, Y = torch.load(fp)
else:
    t0 = time.time(); feats, ys = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(DataLoader(DS(), batch_size=args.bs, num_workers=args.workers)):
            f = model.encode_image(x.to(dev)); feats.append((f / f.norm(dim=-1, keepdim=True)).cpu()); ys.append(y)
            if i % 50 == 0: print(f"  img batch {i} {time.time()-t0:.0f}s", flush=True)
    F_img, Y = torch.cat(feats), torch.cat(ys)
    torch.save((F_img, Y), fp)
F_img, Y = F_img.to(dev), Y.to(dev)

variants = json.load(open("utils/inat21_label_variants.json"))
TEMPLATES = {"generic2": ["a photo of a {}.", "a photo of the {}."],
             "imagenet7": ["itap of a {}.", "a bad photo of the {}.", "a origami {}.",
                           "a photo of the large {}.", "a {} in a video game.",
                           "art of the {}.", "a photo of the small {}."]}
rows = []
for vname, labels in variants.items():
    for tname, temps in TEMPLATES.items():
        tf = []
        with torch.no_grad():
            for i in range(0, 10000, 500):
                texts = [t.format(l) for l in labels[i:i + 500] for t in temps]
                e = model.encode_text(tok(texts).to(dev)).view(-1, len(temps), 512).mean(1)
                tf.append(e / e.norm(dim=-1, keepdim=True))
        T = torch.cat(tf)
        pred = torch.cat([(F_img[i:i + 4096] @ T.T).argmax(1) for i in range(0, F_img.shape[0], 4096)])
        top1 = (pred == Y).float().mean().item() * 100
        rows.append((vname, tname, top1)); print(f"{vname:11s} {tname:10s} top1 {top1:.2f}", flush=True)
with open(os.path.join(args.out, "label_smoke.md"), "w") as f:
    f.write("| labels | templates | val zero-shot top-1 |\n|---|---|---|\n")
    for v, t, a in rows: f.write(f"| {v} | {t} | {a:.2f} |\n")
print(open(os.path.join(args.out, "label_smoke.md")).read())
