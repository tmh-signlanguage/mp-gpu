"""Per-video comparison of GPU extraction vs reference .pose files, with anomaly flags.

Usage: python compare_dataset.py REF_DIR TEST_DIR [--ref-suffix .pose] [--csv out.csv]
Compare mp-gpu output with existing legacy extractions. TEST_DIR holds <video>.npz (or .pose); the reference is REF_DIR/<video><ref-suffix>.

Per video: frame counts, fps, size, presence agreement, body error over the first frames where both
have a pose ("early": before tracking drift, should be ~0 for a faithful pipeline) and overall error
(% of image diagonal). Flags: MISSING_REF, LEN, FPS, SIZE, PRESENCE, EARLY, ERR.
"""
import argparse, csv, glob, os, sys
import numpy as np

from mp_gpu.pose_io import load

COMP = {"body": (0, 33), "face": (33, 501), "lh": (501, 522), "rh": (522, 543)}


def compare(ref, test, early_n=10):
    r, rc, t, tc = ref["data"], ref["conf"], test["data"], test["conf"]
    T = min(len(r), len(t))
    diag = (ref["width"] ** 2 + ref["height"] ** 2) ** 0.5
    out = dict(T_ref=len(r), T_test=len(t), fps_ref=ref["fps"], fps_test=test["fps"],
               size_ref=f'{ref["width"]}x{ref["height"]}', size_test=f'{test["width"]}x{test["height"]}')
    for k, (a, b) in COMP.items():
        rp = rc[:T, a:b].sum(1) > 0
        tp = tc[:T, a:b].sum(1) > 0
        out[f"agree_{k}"] = float((rp == tp).mean()) if T else np.nan
        out[f"rate_{k}"] = float(rp.mean()) if T else np.nan
    a, b = COMP["body"]
    both = (rc[:T, a:b].sum(1) > 0) & (tc[:T, a:b].sum(1) > 0)
    e = np.linalg.norm(r[:T, a:b, :2] - t[:T, a:b, :2], axis=-1).mean(1)[both]   # px, mean over joints
    out["early_px"] = float(np.median(e[:early_n])) if len(e) else np.nan
    out["med_px"] = float(np.median(e)) if len(e) else np.nan
    out["med_pct"] = out["med_px"] / diag * 100 if len(e) else np.nan
    out["p95_pct"] = float(np.percentile(e, 95)) / diag * 100 if len(e) else np.nan
    return out


def flags(o):
    f = []
    if o["T_ref"] != o["T_test"]:
        f.append("LEN")
    if abs(o["fps_ref"] - o["fps_test"]) > 0.01:
        f.append("FPS")
    if o["size_ref"] != o["size_test"]:
        f.append("SIZE")
    if min(o["agree_body"], o["agree_face"]) < 0.97 or min(o["agree_lh"], o["agree_rh"]) < 0.95:
        f.append("PRESENCE")
    if not np.isnan(o["early_px"]) and o["early_px"] > 0.5:
        f.append("EARLY")
    if not np.isnan(o["med_pct"]) and o["med_pct"] > 0.3:
        f.append("ERR")
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir"); ap.add_argument("test_dir")
    ap.add_argument("--ref-suffix", default=".pose")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = []
    tests = sorted(glob.glob(os.path.join(a.test_dir, "*.npz"))) or sorted(glob.glob(os.path.join(a.test_dir, "*.pose")))
    for tp in tests:
        name = os.path.basename(tp)[:-4] if tp.endswith(".npz") else os.path.basename(tp)[:-5]
        rp = os.path.join(a.ref_dir, name + a.ref_suffix)
        if not os.path.exists(rp):
            rows.append(dict(video=name, flags="MISSING_REF")); continue
        try:
            o = compare(load(rp), load(tp))
        except Exception as e:  # noqa
            rows.append(dict(video=name, flags=f"LOAD_ERROR {e}")); continue
        o["video"] = name; o["flags"] = " ".join(flags(o))
        rows.append(o)
    ok = [r for r in rows if "T_ref" in r]
    keys = ["video", "T_ref", "T_test", "fps_ref", "fps_test", "size_ref", "size_test", "agree_body", "agree_face",
            "agree_lh", "agree_rh", "rate_body", "rate_lh", "rate_rh", "early_px", "med_px", "med_pct", "p95_pct", "flags"]
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} videos, {len(ok)} compared")
    if ok:
        def m(k):
            v = np.array([r[k] for r in ok], float); v = v[~np.isnan(v)]
            return f"{np.median(v):.4f}" if len(v) else "nan"
        print(f"median over videos: presence agree body={m('agree_body')} face={m('agree_face')} lh={m('agree_lh')} "
              f"rh={m('agree_rh')} | early body err {m('early_px')} px | body err median {m('med_px')} px "
              f"({m('med_pct')} %diag) p95 {m('p95_pct')} %diag")
    from collections import Counter
    c = Counter(fl for r in rows for fl in (r["flags"].split() if r["flags"] else ["OK"]))
    print("flags:", dict(c))
    for r in rows:
        if r["flags"]:
            if "T_ref" in r:
                print(f"  {r['video'][:40]:40s} {r['flags']:22s} T {r['T_ref']}/{r['T_test']} fps {r['fps_ref']:.3f}/{r['fps_test']:.3f} "
                      f"{r['size_ref']}/{r['size_test']} agree b{r['agree_body']:.3f} f{r['agree_face']:.3f} "
                      f"l{r['agree_lh']:.3f} r{r['agree_rh']:.3f} early {r['early_px']:.2f}px med {r['med_pct']:.3f}%")
            else:
                print(f"  {r['video'][:40]:40s} {r['flags']}")
