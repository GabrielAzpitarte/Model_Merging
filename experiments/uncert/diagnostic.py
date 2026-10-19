"""Uncertainty diagnostic for calibration-derived merge corrections (protocol 02 sec. 9), CLIP ViT-B/32 full FT.

Per linear map p (73), with expert updates D_i, anchor R = mean_i D_i, expert-specific input second moments G_i:
    G = sum_i G_i,   B = sum_i (D_i - R) G_i,   F = B G^-1   (RegMean = R + F; budget C task-separated,
                                                             budget B pooled inputs through every expert)
Question: does resampling-based uncertainty of the WHOLE correction F (bootstrap over whole images, B and G
recomputed jointly) predict F's actual error better than conditioning statistics of G?
Data (experiments/uncert/build_cache.py, train images disjoint from everything else):
  uref (1024/task) -> F_ref, the large-calibration reference (diagnosis only)
  upool (640/task) -> disjoint small subsets of n images per task (C: n = 16 and 64; B: n = 16), 8 subsets each
  uind (256/task)  -> G_ind, independent activations on which errors are measured
For each subset: F_hat, and Bboot = 32 bootstrap replicates (images resampled with replacement within each task,
the same image weights for every expert in budget B).
Targets:     r_fun = tr(dF G_ind dF^T) / tr(F_ref G_ind F_ref^T)      (dF = F_hat - F_ref; PRIMARY)
             r_F   = ||dF||_F^2 / ||F_ref||_F^2
Predictors (subset only): u_rel = mean_b ||F_b - mean_b F_b||_F^2 / ||F_hat||_F^2 (PRIMARY), u_rel_mse (vs F_hat),
             u_rel_G (Ghat-weighted), kappa(Ghat), (tr Ghat / d) / lambda_min(Ghat), tr(Ghat^-1) tr(Ghat) / d^2,
             ||F_hat||_F^2 / ||R||_F^2.
Pre-registered success (protocol 9): median over subsets of Spearman_maps(u_rel, r_fun) >= 0.5 AND >= 0.1 above
every competitor, in both budgets.  Only then: stage "gate" (scalar per-map gate) is run (gate.py).
Writes results/uncert/diagnostic_{records.jsonl, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from condmerge.paths import BASE_MODEL, TASKS8, offline_env  # noqa: E402

offline_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32 Grams (TF32 would act as a noise floor)
torch.backends.cudnn.allow_tf32 = False

from condmerge.clip import load_base_state, load_experts  # noqa: E402
from condmerge.data import CLIP_MEAN, CLIP_STD, load_cache  # noqa: E402
from condmerge.grams import input_name  # noqa: E402
from condmerge.spectra import as_matrix, linear_map_names  # noqa: E402

OUT = ROOT / "results" / "uncert"
N_SUB = 8
B_BOOT = 32


class Acts:
    """Token activations at every distinct linear-map input for a batch of images through one model."""

    def __init__(self, model, names, device):
        self.model, self.device = model, device
        self.keys = sorted({input_name(p) for p in names})
        self.mods = {}
        mods = dict(model.named_modules())
        for p in names:
            k = input_name(p)
            if k == "patches" or k in self.mods:
                continue
            self.mods[k] = mods[p[: -len(".weight")]]
        self.mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x_u8: np.ndarray, batch: int = 128) -> dict[str, torch.Tensor]:
        """{input: (n_img, n_tok, d) float32} for uint8 images (fp32 forward)."""
        out = {k: [] for k in self.keys}
        store = {}
        hooks = [m.register_forward_pre_hook(lambda _m, inp, k=k: store.__setitem__(k, inp[0].detach().float()))
                 for k, m in self.mods.items()]
        ps = self.model.config.patch_size
        try:
            for i in range(0, len(x_u8), batch):
                px = (torch.from_numpy(np.ascontiguousarray(x_u8[i:i + batch])).to(self.device).float() / 255 - self.mean) / self.std
                pt = px.unfold(2, ps, ps).unfold(3, ps, ps).permute(0, 2, 3, 1, 4, 5).reshape(px.shape[0], -1, 3 * ps * ps)
                self.model(pixel_values=px)
                out["patches"].append(pt.float())
                for k in self.mods:
                    out[k].append(store[k])
        finally:
            for h in hooks:
                h.remove()
        return {k: torch.cat(v) for k, v in out.items()}


def gram(x: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over tokens of x x^T; x (n_img, n_tok, d); w (n_img,) image weights (bootstrap counts)."""
    n_img, n_tok, d = x.shape
    xf = x.reshape(-1, d)
    if w is None:
        return (xf.T @ xf).double() / (n_img * n_tok)
    wt = w.repeat_interleave(n_tok).to(x.dtype)
    return ((xf * wt[:, None]).T @ xf).double() / float(wt.sum())


@torch.no_grad()
def stream_grams(acts: Acts, x_u8: np.ndarray, batch: int = 128) -> dict[str, torch.Tensor]:
    acc, n = {}, {}
    for i in range(0, len(x_u8), batch):
        a = acts(x_u8[i:i + batch], batch)
        for k, v in a.items():
            g = (v.reshape(-1, v.shape[-1]).T @ v.reshape(-1, v.shape[-1])).double()
            acc[k] = acc.get(k, 0) + g
            n[k] = n.get(k, 0) + v.shape[0] * v.shape[1]
    return {k: acc[k] / n[k] for k in acc}


def correction(D: torch.Tensor, Gs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """D (K, out, in) float64, Gs (K, in, in) float64 -> (F, G) with F = sum_i (D_i - R) G_i G^-1."""
    R = D.mean(0)
    G = Gs.sum(0)
    B = torch.einsum("kod,kde->oe", D - R, Gs)
    Gsym = 0.5 * (G + G.T)
    try:
        L = torch.linalg.cholesky(Gsym)
        F = torch.cholesky_solve(B.T, L).T
    except RuntimeError:
        F = torch.linalg.lstsq(Gsym, B.T).solution.T
    return F, Gsym


def main() -> None:
    from transformers import CLIPVisionConfig, CLIPVisionModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="C:16,C:64,B:16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quick", action="store_true", help="plumbing test: 1 subset, 2 replicates, separate output")
    args = ap.parse_args()
    global OUT, N_SUB, B_BOOT
    if args.quick:
        OUT, N_SUB, B_BOOT = OUT / "quick_test", 1, 2
    dev = args.device
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = load_base_state()
    experts = load_experts(TASKS8)
    names = linear_map_names(base)
    K = len(TASKS8)
    D32 = {p: torch.stack([as_matrix(p, experts[e][p]).double() - as_matrix(p, base[p]).double() for e in TASKS8]).float().to(dev)
           for p in names}

    class _D(dict):  # float64 view on demand (keeps GPU memory at float32 size)
        def __getitem__(self, p):
            return D32[p].double()
    D = _D({p: None for p in names})
    model = CLIPVisionModel(CLIPVisionConfig.from_pretrained(BASE_MODEL)).eval().to(dev)
    acts = Acts(model, names, dev)
    pix = {s: {t: load_cache(t, s, mmap=True)[0] for t in TASKS8} for s in ("uref", "upool", "uind")}
    if args.quick:  # plumbing only: small reference / independent sets
        pix["uref"] = {t: x[:64] for t, x in pix["uref"].items()}
        pix["uind"] = {t: x[:32] for t, x in pix["uind"].items()}

    def load_expert(e):
        model.load_state_dict({k: v.to(dev) for k, v in experts[e].items()}, strict=True)

    # reference correction and independent Grams, per budget: C (task i through expert i), B (pooled through every
    # expert). Accumulated expert by expert (B_p += (D_e - R) G_e, G_k += G_e) so only one expert's Grams are held.
    budgets = sorted({c.split(":")[0] for c in args.configs.split(",")})
    keys = acts.keys
    F_ref, G_ind = {}, {}
    for budget in budgets:
        Bacc = {p: torch.zeros(D32[p].shape[1:], dtype=torch.float64, device=dev) for p in names}
        Gacc, Iacc = {}, {}
        for i, e in enumerate(TASKS8):
            load_expert(e)
            if budget == "C":
                gr, gi = stream_grams(acts, pix["uref"][e]), stream_grams(acts, pix["uind"][e])
            else:
                gr = stream_grams(acts, np.concatenate([pix["uref"][t] for t in TASKS8]))
                gi = stream_grams(acts, np.concatenate([pix["uind"][t] for t in TASKS8]))
            for p in names:
                Bacc[p] += (D32[p][i].double() - D32[p].double().mean(0)) @ gr[input_name(p)]
            for k in keys:
                Gacc[k] = Gacc.get(k, 0) + gr[k]
                Iacc[k] = Iacc.get(k, 0) + gi[k]
            del gr, gi
            torch.cuda.empty_cache()
            print(f"[{budget}] reference/independent Grams for {e} ({time.time() - t0:.0f}s)", flush=True)
        F_ref[budget] = {}
        for p in names:
            G = Gacc[input_name(p)]
            Gs = 0.5 * (G + G.T)
            try:
                F_ref[budget][p] = torch.cholesky_solve(Bacc[p].T, torch.linalg.cholesky(Gs)).T
            except RuntimeError:
                F_ref[budget][p] = torch.linalg.lstsq(Gs, Bacc[p].T).solution.T
        G_ind[budget] = Iacc  # by input name
        del Bacc, Gacc
        torch.cuda.empty_cache()
    rec_path = OUT / "diagnostic_records.jsonl"
    done = set()
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["budget"], r["n"], r["subset"]))
    rng_master = np.random.default_rng(12345)
    for cfg in args.configs.split(","):
        budget, n = cfg.split(":")[0], int(cfg.split(":")[1])
        for s in range(N_SUB):
            seeds = rng_master.integers(0, 2 ** 31, size=B_BOOT)  # drawn in a fixed order -> reproducible per subset
            if (budget, n, s) in done:
                continue
            ts = time.time()
            idx = np.arange(s * n, (s + 1) * n)
            # activations of the subset through each expert
            A = {}
            for e in TASKS8:
                load_expert(e)
                imgs = pix["upool"][e][idx] if budget == "C" else np.concatenate([pix["upool"][t][idx] for t in TASKS8])
                A[e] = acts(imgs)
            n_img = n if budget == "C" else n * K

            by_input = {}
            for p in names:
                by_input.setdefault(input_name(p), []).append(p)

            def gram_k(k, weights):
                """(K, d, d) Grams of one input for per-image weights (dict expert -> (n_img,) tensor or None)."""
                return torch.stack([gram(A[e][k], None if weights is None else weights[e]) for e in TASKS8])

            Fh = {}
            for k, ps in by_input.items():
                gk = gram_k(k, None)
                for p in ps:
                    Fh[p] = correction(D[p], gk)
                del gk
            # running moments of the replicates (no replicate is stored): sum F_b, sum ||F_b||^2,
            # sum tr(F_b Ghat F_b^T), sum ||F_b - F_hat||^2
            sums = {p: torch.zeros_like(Fh[p][0]) for p in names}
            sq = {p: 0.0 for p in names}
            sqG = {p: 0.0 for p in names}
            mse = {p: 0.0 for p in names}
            for b in range(B_BOOT):
                rng = np.random.default_rng(int(seeds[b]))
                if budget == "C":
                    w = {e: torch.as_tensor(rng.multinomial(n_img, np.full(n_img, 1.0 / n_img)), device=dev).float()
                         for e in TASKS8}
                else:  # pooled: resample images within each task's block, the same weights for every expert
                    ww = np.concatenate([rng.multinomial(n, np.full(n, 1.0 / n)) for _ in TASKS8])
                    wt = torch.as_tensor(ww, device=dev).float()
                    w = {e: wt for e in TASKS8}
                for k, ps in by_input.items():
                    gk = gram_k(k, w)
                    for p in ps:
                        Fb, _ = correction(D[p], gk)
                        Ghat = Fh[p][1]
                        sums[p] += Fb
                        sq[p] += float(Fb.pow(2).sum())
                        sqG[p] += float(torch.einsum("od,de,oe->", Fb, Ghat, Fb))
                        mse[p] += float((Fb - Fh[p][0]).pow(2).sum())
                    del gk
            rows = []
            for p in names:
                F_hat, Ghat = Fh[p]
                mu = sums[p] / B_BOOT
                var_F = max(sq[p] / B_BOOT - float(mu.pow(2).sum()), 0.0)
                var_G = max(sqG[p] / B_BOOT - float(torch.einsum("od,de,oe->", mu, Ghat, mu)), 0.0) / float(torch.trace(Ghat))
                fh2 = float(F_hat.pow(2).sum())
                dF = F_hat - F_ref[budget][p]
                Gi = G_ind[budget][input_name(p)]
                Fr = F_ref[budget][p]
                r_fun = float(torch.einsum("od,de,oe->", dF, Gi, dF) / torch.einsum("od,de,oe->", Fr, Gi, Fr))
                r_F = float(dF.pow(2).sum() / Fr.pow(2).sum())
                ev = torch.linalg.eigvalsh(Ghat)
                d_ = Ghat.shape[0]
                lmin = max(float(ev[0]), 1e-300)
                tr = float(torch.trace(Ghat))
                rows.append({"budget": budget, "n": n, "subset": s, "map": p, "r_fun": r_fun, "r_F": r_F,
                             "u_rel": var_F / max(fh2, 1e-300), "u_rel_mse": mse[p] / B_BOOT / max(fh2, 1e-300),
                             "u_rel_G": var_G / max(float(torch.einsum("od,de,oe->", F_hat, Ghat, F_hat) / tr), 1e-300),
                             "kappa_G": float(ev[-1]) / lmin, "inv_lmin": (tr / d_) / lmin,
                             "trinv": float(torch.sum(1.0 / torch.clamp(ev, min=1e-300))) * tr / d_ ** 2,
                             "fnorm_rel": fh2 / float(D[p].mean(0).pow(2).sum()), "d_in": d_})
            with rec_path.open("a") as f:
                for r in rows:
                    f.write(json.dumps({**r, "time": time.strftime("%H:%M:%S")}) + "\n")
            print(f"[{budget} n={n} subset {s}] done ({time.time() - ts:.0f}s; total {time.time() - t0:.0f}s)", flush=True)
            del A, sums, Fh
            torch.cuda.empty_cache()
    summarize()


def summarize() -> dict:
    from scipy.stats import spearmanr

    recs = [json.loads(l) for l in (OUT / "diagnostic_records.jsonl").read_text().splitlines() if l.strip()]
    preds = ["u_rel", "u_rel_mse", "u_rel_G", "kappa_G", "inv_lmin", "trinv", "fnorm_rel"]
    out = {"criterion": "median over subsets of Spearman_maps(u_rel, r_fun) >= 0.5 and >= 0.1 above every competitor, "
                        "in both budgets", "configs": {}}
    for key in sorted({(r["budget"], r["n"]) for r in recs}):
        sub = [r for r in recs if (r["budget"], r["n"]) == key]
        res = {}
        for target in ("r_fun", "r_F"):
            res[target] = {}
            for pr in preds:
                rh = []
                for s in sorted({r["subset"] for r in sub}):
                    rr = [r for r in sub if r["subset"] == s]
                    rh.append(spearmanr([r[pr] for r in rr], [r[target] for r in rr]).statistic)
                pooled = spearmanr([r[pr] for r in sub], [r[target] for r in sub]).statistic
                res[target][pr] = {"median_over_subsets": float(np.median(rh)), "per_subset": [float(x) for x in rh],
                                   "pooled": float(pooled)}
        # within-map across subsets: does u_rel track which subsets are worse for the same map?
        within = []
        for p in sorted({r["map"] for r in sub}):
            rr = [r for r in sub if r["map"] == p]
            if len(rr) >= 4:
                within.append(spearmanr([r["u_rel"] for r in rr], [r["r_fun"] for r in rr]).statistic)
        res["within_map_spearman_u_rel_vs_r_fun_median"] = float(np.nanmedian(within)) if within else None
        res["calibration_median_r_fun_over_u_rel"] = float(np.median([r["r_fun"] / max(r["u_rel"], 1e-300) for r in sub]))
        prim = res["r_fun"]["u_rel"]["median_over_subsets"]
        comp = max(res["r_fun"][p]["median_over_subsets"] for p in ("kappa_G", "inv_lmin", "trinv", "fnorm_rel"))
        res["passes"] = bool(prim >= 0.5 and prim >= comp + 0.1)
        res["primary"], res["best_competitor"] = prim, comp
        out["configs"][f"{key[0]}:n{key[1]}"] = res
    budgets = {k.split(":")[0] for k in out["configs"]}
    out["diagnostic_passed"] = bool(budgets >= {"B", "C"} and all(v["passes"] for v in out["configs"].values()))
    (OUT / "diagnostic_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {"primary": v["primary"], "best_competitor": v["best_competitor"], "passes": v["passes"]}
                      for k, v in out["configs"].items()}, indent=1), "\nDIAGNOSTIC PASSED:", out["diagnostic_passed"],
          flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize()
    else:
        main()
