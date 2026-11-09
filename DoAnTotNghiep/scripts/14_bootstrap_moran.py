"""
BƯỚC 14 — KIỂM ĐỊNH THỐNG KÊ (BOOTSTRAP THEO NGÀY) + MORAN'S I TRÊN PHẦN DƯ
=========================================================================
Đầu vào: predictions_compare.csv + compare_meta.json do 11 tạo (4 baseline + variant tốt nhất theo Val của mỗi họ mô hình).

  A. CI CHO RMSE/MAE/skill — block bootstrap theo NGÀY.
     Các điểm lưới CÙNG NGÀY chia sẻ thời tiết nên sai số của chúng tương quan rất mạnh; bootstrap theo dòng sẽ cho CI
     hẹp một cách giả tạo. Ở đây đơn vị lấy mẫu lại là CẢ NGÀY (--cluster date_t | date_target); --block L lấy mẫu
     L ngày liên tiếp một lần (circular) để giữ tương quan theo thời gian giữa các ngày kề nhau (L=1: bootstrap theo ngày thuần).
     Thống kê RMSE được tính từ tổng bình phương theo ngày nên bootstrap rất nhanh (không dự báo lại).

  B. SO SÁNH TỪNG CẶP (A − B trên RMSE, CÙNG tập mẫu bootstrap nên là kiểm định cặp đôi):
     CI của hiệu, p-value hai phía từ phân phối bootstrap, hiệu chỉnh Holm cho toàn bộ cặp trong cùng kịch bản.
     'XGB hơn RF 0.05 °C' chỉ đáng tin khi CI của hiệu không chứa 0 sau hiệu chỉnh.

  C. MORAN'S I TRÊN PHẦN DƯ (e = dự báo − thực đo) cho các mô hình --moran_models (mặc định: mô hình ML tốt nhất theo Val):
       - theo từng ngày: I của phần dư các điểm hợp lệ ngày đó (KNN k láng giềng, chuẩn hoá hàng; z theo xấp xỉ chuẩn);
         báo cáo I trung bình, tỉ lệ ngày có p<0.05 (một phía dương: tự tương quan không gian DƯƠNG);
       - bản đồ phần dư trung bình theo điểm (xu hướng sai số ổn định theo vị trí): I + p hoán vị (esda).
     I dương rõ rệt = phần dư còn cấu trúc không gian = mô hình còn thiếu thông tin không gian (độ cao, khoảng cách tới biển/nước,
     đất xây dựng...). Đây là chẩn đoán, không phải kiểm định nhân quả.

Đầu ra (../data/ và ../data/figures/):
    bootstrap_ci.csv           RMSE/MAE/skill + CI theo kịch bản × (baseline & mô hình)
    paired_tests.csv           hiệu RMSE từng cặp + CI + p bootstrap + p Holm
    moran_residuals.csv        Moran's I phần dư (theo ngày và bản đồ phần dư trung bình)
    residual_map_mean.csv      phần dư trung bình theo điểm × (mô hình, kịch bản) của --moran_models — vẽ lên WebGIS nếu cần
    figures/rmse_ci.png, moran_by_date.png

Cách chạy:
    python 14_bootstrap_moran.py
    python 14_bootstrap_moran.py --n_boot 5000 --block 3 --moran_models "XGBoost,Random Forest"
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_utils as eu  # noqa: E402

SPLIT_NAMES = {"val": "Validation", "test_temporal": "Thời gian", "test_spatial": "Không gian"}


# ---------------------------------------------------------------------
# BOOTSTRAP THEO NGÀY
# ---------------------------------------------------------------------

def draw_counts(K: int, B: int, block: int, rng) -> np.ndarray:
    """Ma trận (B, K): số lần mỗi ngày được chọn trong mỗi lần bootstrap (circular block bootstrap, block=1 -> thuần)."""
    if block <= 1:
        idx = rng.integers(0, K, size=(B, K))
    else:
        nb = int(np.ceil(K / block))
        starts = rng.integers(0, K, size=(B, nb))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % K).reshape(B, -1)[:, :K]
    flat = (idx + np.arange(B)[:, None] * K).ravel()
    return np.bincount(flat, minlength=B * K).reshape(B, K).astype(float)


def holm(p: np.ndarray) -> np.ndarray:
    """Hiệu chỉnh Holm-Bonferroni (step-down), trả p đã hiệu chỉnh theo thứ tự gốc."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_p(d: np.ndarray, B: int) -> float:
    """p hai phía từ phân phối bootstrap của hiệu d (H0: hiệu = 0), sàn 1/(B+1)."""
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return float(max(p, 1.0 / (B + 1)))


# ---------------------------------------------------------------------
# MORAN'S I (KNN, chuẩn hoá hàng) — thực hiện trực tiếp để chạy nhanh từng ngày
# ---------------------------------------------------------------------

def _xy_km(lon, lat) -> np.ndarray:
    lat0 = float(np.mean(lat))
    return np.column_stack([np.asarray(lon, float) * 111.32 * np.cos(np.radians(lat0)), np.asarray(lat, float) * 110.57])


def knn_weights(xy: np.ndarray, k: int):
    """W thưa chuẩn hoá hàng (KNN k láng giềng gần nhất, không tính chính nó)."""
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    n = len(xy)
    k = min(k, n - 1)
    _, nbr = cKDTree(xy).query(xy, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = nbr[:, 1:].ravel()
    return csr_matrix((np.full(len(rows), 1.0 / k), (rows, cols)), shape=(n, n))


def moran_fast(z: np.ndarray, W):
    """(I, E[I], z-score chuẩn tắc, p một phía dương) theo xấp xỉ chuẩn (cùng công thức z_norm của esda)."""
    from scipy.stats import norm

    n = len(z)
    zc = z - z.mean()
    den = float(zc @ zc)
    if n < 10 or den <= 0:
        return np.nan, np.nan, np.nan, np.nan
    s0 = float(W.sum())
    I = (n / s0) * float(zc @ (W @ zc)) / den
    Wt = W + W.T
    s1 = 0.5 * float(Wt.multiply(Wt).sum())
    s2 = float(((np.asarray(W.sum(axis=1)).ravel() + np.asarray(W.sum(axis=0)).ravel()) ** 2).sum())
    EI = -1.0 / (n - 1)
    var = (n * n * s1 - n * s2 + 3 * s0 * s0) / ((n * n - 1) * s0 * s0) - EI * EI
    zs = (I - EI) / np.sqrt(var) if var > 0 else np.nan
    return float(I), float(EI), float(zs), float(1 - norm.cdf(zs)) if np.isfinite(zs) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=3, help="Độ dài khối ngày liên tiếp (1 = bootstrap theo ngày thuần)")
    ap.add_argument("--cluster", default="date_t", choices=["date_t", "date_target"], help="Cột ngày làm đơn vị lấy mẫu")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--moran_models", default=None, help="Tên mô hình (phân tách bởi dấu phẩy); mặc định: ML tốt nhất theo Validation")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--max_dates", type=int, default=80, help="Số ngày tối đa lấy mẫu mỗi kịch bản cho Moran theo ngày")
    ap.add_argument("--n_perm", type=int, default=499, help="Số hoán vị esda cho bản đồ phần dư trung bình")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    path = os.path.join(args.data_dir, "predictions_compare.csv")
    meta = eu.load_json(os.path.join(args.data_dir, "compare_meta.json"))
    if not os.path.exists(path) or meta is None:
        raise SystemExit("❌ Thiếu predictions_compare.csv / compare_meta.json — chạy 11_improved_models.py trước.")
    P = pd.read_csv(path)
    cols = [(c["name"], c["column"], c["kind"], c["variant"]) for c in meta["columns"] if c["column"] in P.columns]
    names = [c[0] for c in cols]
    kind_of = {c[0]: c[2] for c in cols}
    rng = np.random.default_rng(args.seed)
    print(f"{len(P):,} dòng dự báo | mô hình/baseline: {names}")

    ci_rows, pair_rows = [], []
    for split, scen in SPLIT_NAMES.items():
        D = P[P["split"] == split]
        if D.empty:
            continue
        y = D[eu.TARGET].to_numpy(float)
        codes, uniq = pd.factorize(pd.to_datetime(D[args.cluster]), sort=True)
        K = len(uniq)
        nk = np.bincount(codes, minlength=K).astype(float)
        sse = {nm: np.bincount(codes, weights=(D[col].to_numpy(float) - y) ** 2, minlength=K) for nm, col, _, _ in cols}
        sae = {nm: np.bincount(codes, weights=np.abs(D[col].to_numpy(float) - y), minlength=K) for nm, col, _, _ in cols}
        C = draw_counts(K, args.n_boot, args.block, rng)
        den = C @ nk
        rm = {nm: np.sqrt((C @ sse[nm]) / den) for nm in names}
        ma = {nm: (C @ sae[nm]) / den for nm in names}
        lo_q, hi_q = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
        for nm in names:
            pt = eu.metrics(y, D[next(c[1] for c in cols if c[0] == nm)].to_numpy(float))
            sk = 1 - rm[nm] / rm["Persistence"] if "Persistence" in rm else np.full(args.n_boot, np.nan)
            ci_rows.append({
                "scenario": scen, "model": nm, "kind": kind_of[nm], "n_rows": pt["n"], "n_days": K,
                "rmse": pt["rmse"], "rmse_lo": float(np.percentile(rm[nm], lo_q)), "rmse_hi": float(np.percentile(rm[nm], hi_q)),
                "mae": pt["mae"], "mae_lo": float(np.percentile(ma[nm], lo_q)), "mae_hi": float(np.percentile(ma[nm], hi_q)),
                "skill_vs_persistence": 1 - pt["rmse"] / eu.metrics(y, D["pred_Persistence"].to_numpy(float))["rmse"] if "pred_Persistence" in D.columns else np.nan,
                "skill_lo": float(np.percentile(sk, lo_q)), "skill_hi": float(np.percentile(sk, hi_q)),
            })
        rows_s = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                d = rm[a] - rm[b]
                pt_d = next(r["rmse"] for r in ci_rows if r["scenario"] == scen and r["model"] == a) - next(r["rmse"] for r in ci_rows if r["scenario"] == scen and r["model"] == b)
                rows_s.append({"scenario": scen, "A": a, "B": b, "rmse_diff_A_minus_B": pt_d, "ci_lo": float(np.percentile(d, lo_q)),
                               "ci_hi": float(np.percentile(d, hi_q)), "p_boot": boot_p(d, args.n_boot)})
        ph = holm(np.array([r["p_boot"] for r in rows_s]))
        for r, p_adj in zip(rows_s, ph):
            r["p_holm"] = float(p_adj)
            r["significant"] = bool(p_adj < args.alpha)
            r["better"] = (r["A"] if r["rmse_diff_A_minus_B"] < 0 else r["B"]) if p_adj < args.alpha else "không khác biệt"
        pair_rows += rows_s

    ci = pd.DataFrame(ci_rows)
    pairs = pd.DataFrame(pair_rows)
    ci.to_csv(os.path.join(args.data_dir, "bootstrap_ci.csv"), index=False)
    pairs.to_csv(os.path.join(args.data_dir, "paired_tests.csv"), index=False)

    ml_names = [n for n in names if kind_of[n] == "ml"]
    val_rm = {n: eu.metrics(P[P["split"] == "val"][eu.TARGET], P[P["split"] == "val"][next(c[1] for c in cols if c[0] == n)])["rmse"] for n in ml_names} if (P["split"] == "val").any() else {}
    selected = min(val_rm, key=val_rm.get) if val_rm else (ml_names[0] if ml_names else None)

    print("\n" + "=" * 100)
    print(f" RMSE (°C) với CI {1 - args.alpha:.0%} — block bootstrap theo ngày ({args.n_boot} lần, block={args.block}, đơn vị = {args.cluster})")
    print("=" * 100)
    for scen in SPLIT_NAMES.values():
        sub = ci[ci["scenario"] == scen].sort_values("rmse")
        if sub.empty:
            continue
        print(f"\n [{scen}] {int(sub['n_days'].iloc[0])} ngày, {int(sub['n_rows'].iloc[0]):,} dòng")
        for _, r in sub.iterrows():
            tag = " ★" if r["model"] == selected else ""
            print(f"   {r['model']:<20} {r['rmse']:.3f}  [{r['rmse_lo']:.3f}; {r['rmse_hi']:.3f}]   skill {r['skill_vs_persistence']:+.2f} [{r['skill_lo']:+.2f}; {r['skill_hi']:+.2f}]{tag}")
        if selected:
            pp = pairs[(pairs["scenario"] == scen) & ((pairs["A"] == selected) | (pairs["B"] == selected))]
            print(f"   — {selected} (★ chọn theo Val) so với từng mô hình (hiệu RMSE; âm = {selected} tốt hơn; p đã Holm):")
            for _, r in pp.iterrows():
                other = r["B"] if r["A"] == selected else r["A"]
                dif = r["rmse_diff_A_minus_B"] * (1 if r["A"] == selected else -1)
                lo, hi = (r["ci_lo"], r["ci_hi"]) if r["A"] == selected else (-r["ci_hi"], -r["ci_lo"])
                print(f"       vs {other:<20} {dif:+.3f} [{lo:+.3f}; {hi:+.3f}]  p_holm={r['p_holm']:.3f} {'*' if r['significant'] else ''}")

    # =================== C. MORAN ===================
    mm = [m.strip() for m in args.moran_models.split(",")] if args.moran_models else ([selected] if selected else [])
    mm = [m for m in mm if m in names]
    mrows, daily_store, map_saved = [], {}, False
    for mname in mm:
        col = next(c[1] for c in cols if c[0] == mname)
        for split, scen in SPLIT_NAMES.items():
            D = P[P["split"] == split]
            if D.empty or not {"lon", "lat"}.issubset(D.columns):
                continue
            D = D.assign(e=D[col].to_numpy(float) - D[eu.TARGET].to_numpy(float))
            dates = np.array(sorted(D["date_target"].unique()))
            if len(dates) > args.max_dates:
                dates = rng.choice(dates, args.max_dates, replace=False)
            recs = []
            for dt in sorted(dates):
                g = D[D["date_target"] == dt].drop_duplicates("point_id")
                if len(g) < 30:
                    continue
                W = knn_weights(_xy_km(g["lon"], g["lat"]), args.knn)
                I, EI, zs, p = moran_fast(g["e"].to_numpy(float), W)
                recs.append({"date": dt, "n": len(g), "I": I, "z": zs, "p": p})
            daily = pd.DataFrame(recs)
            daily_store[(mname, scen)] = daily
            if len(daily):
                mrows.append({"model": mname, "scenario": scen, "level": "theo ngày", "n_units": len(daily), "I": float(daily["I"].mean()),
                              "z_mean": float(daily["z"].mean()), "p_perm": np.nan, "share_p_lt_005": float((daily["p"] < 0.05).mean()), "mean_points": float(daily["n"].mean())})
            mp = D.groupby("point_id").agg(lon=("lon", "first"), lat=("lat", "first"), mean_resid=("e", "mean"), n=("e", "size")).reset_index()
            mp = mp[mp["n"] >= 3]
            if len(mp) >= 30:
                try:
                    import warnings

                    from esda.moran import Moran
                    from libpysal.weights import KNN

                    warnings.filterwarnings("ignore", message=".*not fully connected.*")
                    warnings.filterwarnings("ignore", category=UserWarning, module="libpysal")
                    w = KNN.from_array(_xy_km(mp["lon"], mp["lat"]), k=min(args.knn, len(mp) - 1))
                    w.transform = "r"
                    mo = Moran(mp["mean_resid"].to_numpy(float), w, permutations=args.n_perm)
                    mrows.append({"model": mname, "scenario": scen, "level": "bản đồ phần dư TB", "n_units": len(mp), "I": float(mo.I),
                                  "z_mean": np.nan, "p_perm": float(mo.p_sim), "share_p_lt_005": np.nan, "mean_points": float(len(mp))})
                except ImportError:
                    print("⚠️  Thiếu esda/libpysal — bỏ qua Moran trên bản đồ phần dư trung bình.")
                mp.assign(model=mname, scenario=scen).to_csv(os.path.join(args.data_dir, "residual_map_mean.csv"), mode="a" if map_saved else "w",
                                                             header=not map_saved, index=False)
                map_saved = True
    mor = pd.DataFrame(mrows)
    if len(mor):
        eu.db_columns(mor).to_csv(os.path.join(args.data_dir, "moran_residuals.csv"), index=False)
        print("\n" + "=" * 100)
        print(" MORAN'S I TRÊN PHẦN DƯ (I > 0: phần dư tự tương quan không gian dương → còn thiếu thông tin không gian)")
        print("=" * 100)
        for _, r in mor.iterrows():
            if r["level"] == "theo ngày":
                print(f"   {r['model']:<16} {r['scenario']:<11} theo ngày: I TB = {r['I']:+.3f}, z TB = {r['z_mean']:+.1f}, "
                      f"{r['share_p_lt_005']:.0%} số ngày có p<0.05 ({int(r['n_units'])} ngày, ~{r['mean_points']:.0f} điểm/ngày)")
            else:
                print(f"   {r['model']:<16} {r['scenario']:<11} bản đồ TB:  I = {r['I']:+.3f}, p hoán vị = {r['p_perm']:.3f} ({int(r['n_units'])} điểm)")

    # =================== VẼ ẢNH ===================
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.join(args.data_dir, "figures"), exist_ok=True)
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": eu.AXIS, "figure.facecolor": eu.SURFACE, "axes.facecolor": eu.SURFACE})
        scens = [s for s in SPLIT_NAMES.values() if (ci["scenario"] == s).any()]
        fig, axes = plt.subplots(1, len(scens), figsize=(4.6 * len(scens), 0.42 * len(names) + 1.8), sharey=True, squeeze=False)
        order = ci[ci["scenario"] == scens[0]].sort_values("rmse")["model"].tolist()
        for ax, scen in zip(axes[0], scens):
            sub = ci[ci["scenario"] == scen].set_index("model").reindex(order)
            ypos = np.arange(len(order))[::-1]
            for yy, (nm, r) in zip(ypos, sub.iterrows()):
                col = eu.PALETTE[0] if kind_of[nm] == "ml" else eu.MUTED
                ax.plot([r["rmse_lo"], r["rmse_hi"]], [yy, yy], color=col, lw=2.2, solid_capstyle="round")
                ax.plot(r["rmse"], yy, "o", color=col, ms=7, mec=eu.SURFACE, mew=1.5)
            ax.set_yticks(ypos)
            ax.set_yticklabels(order, color=eu.INK2)
            ax.set_xlabel("RMSE (°C)", color=eu.INK2)
            ax.set_title(scen, color=eu.INK, fontsize=11, loc="left")
            ax.grid(axis="x", color=eu.GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(eu.AXIS)
            ax.tick_params(colors=eu.INK2, left=False)
        fig.suptitle(f"RMSE và CI {1 - args.alpha:.0%} (block bootstrap theo ngày) — xanh: mô hình ML, xám: baseline", color=eu.INK, fontsize=11.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(args.data_dir, "figures", "rmse_ci.png"), dpi=160)
        plt.close(fig)

        have = [(k, v) for k, v in daily_store.items() if len(v)]
        if have:
            fig, axes = plt.subplots(1, len(have), figsize=(4.8 * len(have), 3.6), sharey=True, squeeze=False)
            for ax, ((mname, scen), dd) in zip(axes[0], have):
                dd = dd.sort_values("date")
                ax.axhline(0, color=eu.AXIS, lw=1)
                ax.plot(pd.to_datetime(dd["date"]), dd["I"], color=eu.PALETTE[0], lw=1.4, marker="o", ms=4, mec=eu.SURFACE, mew=1)
                ax.set_title(f"{mname} — {scen}", color=eu.INK, fontsize=10.5, loc="left")
                ax.tick_params(axis="x", rotation=30)
                eu.style_axes(ax)
            axes[0][0].set_ylabel("Moran's I của phần dư", color=eu.INK2)
            fig.tight_layout()
            fig.savefig(os.path.join(args.data_dir, "figures", "moran_by_date.png"), dpi=160)
            plt.close(fig)
        print(f"\n🖼  Ảnh: {os.path.join(args.data_dir, 'figures')}")

    print(f"\n✅ Đã lưu bootstrap_ci.csv, paired_tests.csv, moran_residuals.csv, residual_map_mean.csv trong {args.data_dir}")


if __name__ == "__main__":
    main()
