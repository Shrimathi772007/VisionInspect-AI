import json, os, numpy as np
from PIL import Image
a = json.load(open(os.environ["SP"] + "/audit.json"))
tot_tr = tot_te = 0
print("cat | train | tr | val | test_good | test_def | test_total | def_frac | val FP granularity")
for c, r in a.items():
    n = r["train_good"]; k = min(max(round(n * 0.8), 1), n - 1)
    tg = r["test_types"]["good"]; td = sum(v for t, v in r["test_types"].items() if t != "good")
    print(c, n, k, n - k, tg, td, tg + td, f"{td/(tg+td):.1%}", f"1/{n-k}={1/(n-k):.1%}")
    if c != "bottle": tot_tr += n; tot_te += tg + td
print("14-cat train total", tot_tr, "test total", tot_te)
# mask sanity
bad = []; empty = 0; nonbin = 0; n = 0
for c in a:
    g = f"dataset/{c}/ground_truth"
    for t in os.listdir(g):
        for f in os.listdir(f"{g}/{t}"):
            m = np.array(Image.open(f"{g}/{t}/{f}")); n += 1
            if m.max() == 0: empty += 1
            if set(np.unique(m)) - {0, 255}: nonbin += 1
print("masks", n, "empty", empty, "non-binary", nonbin)
