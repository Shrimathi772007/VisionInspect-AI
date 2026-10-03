import os, sys, json, collections
from PIL import Image
root = "dataset"
cats = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
out = {}
for c in cats:
    r = {}
    p = os.path.join(root, c)
    r["top"] = sorted(os.listdir(p))
    def files(d): return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    tr = os.path.join(p, "train"); te = os.path.join(p, "test"); gt = os.path.join(p, "ground_truth")
    r["train_subdirs"] = sorted(os.listdir(tr))
    r["train_good"] = len(files(os.path.join(tr, "good")))
    r["test_types"] = {}
    r["gt_types"] = {}
    exts = collections.Counter(); dims = collections.Counter(); modes = collections.Counter(); mdims = collections.Counter()
    for t in sorted(os.listdir(te)):
        fs = files(os.path.join(te, t)); r["test_types"][t] = len(fs)
        for f in fs: exts[os.path.splitext(f)[1].lower()] += 1
    for f in files(os.path.join(tr, "good")): exts[os.path.splitext(f)[1].lower()] += 1
    r["gt_types"] = {}
    align = True; mismatch = []
    for t in sorted(os.listdir(gt)):
        fs = files(os.path.join(gt, t)); r["gt_types"][t] = len(fs)
        # alignment: mask name = <img stem>_mask.png
        imgs = files(os.path.join(te, t))
        stems = {os.path.splitext(i)[0] for i in imgs}
        mstems = {os.path.splitext(m)[0].removesuffix("_mask") for m in fs}
        if stems != mstems: align = False; mismatch.append(t)
        # dimension match
        for i in imgs:
            with Image.open(os.path.join(te, t, i)) as im, Image.open(os.path.join(gt, t, os.path.splitext(i)[0] + "_mask.png")) as mk:
                if im.size != mk.size: align = False; mismatch.append((t, i, im.size, mk.size))
    r["align"] = align; r["mismatch"] = mismatch[:5]
    for split, d in [("train", os.path.join(tr, "good"))] + [("test/" + t, os.path.join(te, t)) for t in os.listdir(te)]:
        for f in files(d):
            with Image.open(os.path.join(d, f)) as im:
                dims[im.size] += 1; modes[im.mode] += 1
    r["exts"] = dict(exts); r["dims"] = {f"{k[0]}x{k[1]}": v for k, v in dims.items()}; r["modes"] = dict(modes)
    # mask modes
    mm = collections.Counter()
    for t in os.listdir(gt):
        for f in files(os.path.join(gt, t))[:3]:
            with Image.open(os.path.join(gt, t, f)) as m: mm[m.mode] += 1
    r["mask_modes"] = dict(mm)
    out[c] = r
json.dump(out, open(sys.argv[1], "w"), indent=1)
for c, r in out.items():
    tg = r["test_types"].get("good", 0)
    dt = {k: v for k, v in r["test_types"].items() if k != "good"}
    print(c, "| top", r["top"], "| trainsub", r["train_subdirs"], "| train_good", r["train_good"], "| test_good", tg, "| defective", sum(dt.values()), "| ntypes", len(dt))
    print("   types", dt)
    print("   masks", r["gt_types"], "total", sum(r["gt_types"].values()), "align", r["align"], r["mismatch"])
    print("   exts", r["exts"], "dims", r["dims"], "modes", r["modes"], "maskmodes", r["mask_modes"])
