"""
BƯỚC 15 — SPATIAL CV: LEAVE-DISTRICT-OUT (mỗi quận lần lượt là tập kiểm tra, mô hình KHÔNG thấy quận đó)
==========================================================================================================
Vì sao cần: Test không gian của 04 chỉ có vài quận nông thôn (mặc định Cần Giờ, Củ Chi, Dầu Tiếng) nên không nói được mô hình
chuyển sang quận ĐÔ THỊ tốt đến đâu. CV này cho RMSE của TỪNG quận khi quận đó bị giữ ra ngoài, kể cả quận đô thị (Gò Vấp...).

Thiết kế (mô hình = đúng họ + variant + số cây của best_model_v2.pkl; siêu tham số đã dò của 12 được dùng nếu có; KHÔNG early stopping
vì tập giữ lại không được tham gia chọn):
  * Mỗi fold: nhóm quận giữ ra (--folds 0 = leave-ONE-district-out nếu ≤ 30 quận, ngược lại 10 fold cân bằng số dòng).
    Climatology và damped-anomaly baseline CŨNG được fit lại không có quận giữ ra (quận đó dùng đường lùi nhóm NDVI×tháng).
  * Huấn luyện chỉ trên các quận còn lại VÀ chỉ các dòng có date_target < --cutoff (mặc định 2025-01-01).
  * Mỗi quận giữ ra được chấm trên HAI cửa sổ thời gian với CÙNG một mô hình:
        'cùng giai đoạn'  : dòng nằm hoàn toàn trước cutoff — trùng NGÀY với dữ liệu huấn luyện của các quận khác
        'sau giai đoạn'   : dòng có date_t >= cutoff — các ngày mô hình chưa từng thấy ở BẤT KỲ quận nào (chặt chẽ)
    ⚠️ Lý do: ERA5 gần như giống nhau mọi điểm trong cùng ngày nên cây có thể dùng chúng như 'mã ngày' để nhớ dị thường thời tiết
    toàn vùng từ quận khác cùng ngày. Nếu skill (so với Persistence) ở 'cùng giai đoạn' cao hơn rõ so với 'sau giai đoạn' thì kiểu
    Test không gian 'quận giữ ra, mọi ngày' đang lạc quan. Số liệu 'sau giai đoạn' là con số nên báo cáo cho khả năng chuyển sang quận mới.
    (So theo skill chứ không theo RMSE thô, vì hai cửa sổ thuộc năm khác nhau nên Persistence cũng khó/dễ khác nhau.)

Đầu ra (../data/ và ../data/figures/):
    spatial_cv_by_district.csv   RMSE/MAE/bias từng quận × cửa sổ × (mô hình + 4 baseline), kèm NDBI/NDVI trung bình để biết quận đô thị
    spatial_cv_summary.csv       gộp theo cửa sổ: RMSE gộp, trung bình/trung vị/tệ nhất theo quận, số quận thắng Persistence & Damped anomaly
    spatial_cv_predictions.csv   (chỉ với --save_preds) dự báo từng dòng
    figures/spatial_cv.png

Cách chạy:
    python 15_spatial_cv.py
    python 15_spatial_cv.py --model_file best_model_v2_tuned.pkl --cutoff 2025-01-01 --max_train_rows 300000
    python 15_spatial_cv.py --folds 8                 # nhanh hơn khi có quá nhiều quận
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_utils as eu  # noqa: E402

WINDOWS = [("overlap", "cùng giai đoạn"), ("future", "sau giai đoạn")]


def assign_folds(counts: pd.Series, n_folds: int) -> list:
    """Chia quận vào n_folds nhóm cân bằng số dòng (tham lam: quận lớn trước, vào nhóm đang nhỏ nhất)."""
    folds = [[] for _ in range(n_folds)]
    load = np.zeros(n_folds)
    for d, c in counts.sort_values(ascending=False).items():
        i = int(np.argmin(load))
        folds[i].append(d)
        load[i] += c
    return [f for f in folds if f]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--model_file", default="best_model_v2.pkl")
    ap.add_argument("--folds", type=int, default=0, help="0 = tự động (leave-one-district-out nếu <= 30 quận, ngược lại 10 fold)")
    ap.add_argument("--cutoff", default="2025-01-01", help="Ranh giới huấn luyện / cửa sổ 'sau giai đoạn'")
    ap.add_argument("--min_clim_n", type=int, default=20)
    ap.add_argument("--min_leaf", type=int, default=50)
    ap.add_argument("--rf_trees", type=int, default=100)
    ap.add_argument("--max_train_rows", type=int, default=None, help="Lấy mẫu ngẫu nhiên dòng huấn luyện mỗi fold (thử nhanh)")
    ap.add_argument("--min_eval_rows", type=int, default=100, help="Quận có ít hơn số dòng này ở một cửa sổ sẽ bị bỏ khỏi cửa sổ đó")
    ap.add_argument("--save_preds", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    sp = eu.load_splits(args.data_dir)
    pool = pd.concat([sp[k] for k in ("train", "val", "test_temporal", "test_spatial") if len(sp[k])], ignore_index=True)
    pool = pool.drop_duplicates(["point_id", "date_t", "date_target"]).reset_index(drop=True)
    pool["district"] = pool["ADM1_NAME"].astype(str) + " / " + pool["ADM2_NAME"].astype(str)
    spatial_now = set(sp["test_spatial"]["ADM1_NAME"].astype(str) + " / " + sp["test_spatial"]["ADM2_NAME"].astype(str)) if len(sp["test_spatial"]) else set()

    cutoff = pd.Timestamp(args.cutoff)
    d_t, d_n = pd.to_datetime(pool["date_t"]), pd.to_datetime(pool["date_target"])
    train_ok = (d_n < cutoff).to_numpy()
    win_mask = {"overlap": (d_n < cutoff).to_numpy(), "future": (d_t >= cutoff).to_numpy()}
    print(f"Pool: {len(pool):,} dòng, {pool['district'].nunique()} quận | cutoff {args.cutoff}: "
          f"huấn luyện {int(train_ok.sum()):,} | 'cùng giai đoạn' {int(win_mask['overlap'].sum()):,} | 'sau giai đoạn' {int(win_mask['future'].sum()):,}")
    if win_mask["future"].sum() == 0:
        print("⚠️  Không có dòng nào sau cutoff — chỉ chấm cửa sổ 'cùng giai đoạn'.")

    bundle = eu.load_bundle(args.data_dir, args.model_file)
    mkey = bundle["model_key"]
    v0 = eu.Variant.from_dict(bundle["variant"])
    n_trees = bundle.get("n_trees")
    tuned = bundle.get("tuned_params") or {}
    print(f"Mô hình: {bundle['model_name']} / {v0.name}, {n_trees if n_trees else args.rf_trees} cây" + (" (siêu tham số đã dò)" if tuned else ""))

    counts = pool.groupby("district").size()
    n_dist = len(counts)
    n_folds = n_dist if (args.folds == 0 and n_dist <= 30) else (args.folds or 10)
    n_folds = min(n_folds, n_dist)
    folds = assign_folds(counts, n_folds)
    print(f"{len(folds)} fold ({'leave-one-district-out' if len(folds) == n_dist else 'nhóm quận cân bằng'})")

    rng = np.random.default_rng(args.seed)
    parts = []
    for fi, held in enumerate(folds, 1):
        t0 = time.time()
        in_held = pool["district"].isin(held).to_numpy()
        tr = pool[(~in_held) & train_ok]
        if len(tr) < 1000:
            print(f"   fold {fi}: huấn luyện chỉ {len(tr)} dòng — bỏ qua.")
            continue
        if args.max_train_rows and len(tr) > args.max_train_rows:
            tr = tr.sample(args.max_train_rows, random_state=args.seed)
        clim = eu.Climatology(args.min_clim_n).fit(tr)
        damped = eu.DampedAnomaly(clim).fit(tr)
        v = eu.Variant(v0.mode, v0.fset, clim if v0.needs_clim else None)
        trp = v.prepare(tr)
        model, _, _ = eu.fit_model(mkey, trp[v.features], v.y(trp), None, None, seed=args.seed, n_jobs=args.n_jobs, params=tuned,
                                   rf_trees=args.rf_trees, min_leaf=args.min_leaf, fixed_trees=n_trees)
        for wkey, wlabel in WINDOWS:
            ev = pool[in_held & win_mask[wkey]]
            if len(ev) == 0:
                continue
            pred = v.predict(model, v.prepare(ev))
            base = eu.baseline_predictions(ev, clim, damped)
            part = ev[["district", "point_id", "date_t", "date_target", "delta_t", "LST_t", "LST_target", "NDVI", "NDBI"]].copy()
            part["window"] = wkey
            part["fold"] = fi
            part["pred_model"] = pred
            for nm, arr in base.items():
                part["pred_" + nm.replace(" ", "_")] = arr
            parts.append(part)
        print(f"   fold {fi}/{len(folds)}: {len(held)} quận giữ ra ({', '.join(held[:3])}{'...' if len(held) > 3 else ''}), "
              f"huấn luyện {len(tr):,} dòng, {time.time() - t0:.0f}s")
    if not parts:
        raise SystemExit("❌ Không có kết quả nào — kiểm tra --cutoff và dữ liệu.")
    R = pd.concat(parts, ignore_index=True)
    if args.save_preds:
        R.to_csv(os.path.join(args.data_dir, "spatial_cv_predictions.csv"), index=False)

    pred_cols = {"Mô hình": "pred_model", "Persistence": "pred_Persistence", "Climatology": "pred_Climatology",
                 "Anomaly persistence": "pred_Anomaly_persistence", "Damped anomaly": "pred_Damped_anomaly"}
    rows = []
    for (wkey, dist), g in R.groupby(["window", "district"]):
        if len(g) < args.min_eval_rows:
            continue
        y = g["LST_target"].to_numpy(float)
        rec = {"window": wkey, "district": dist, "n": len(g), "n_days": int(g["date_t"].nunique()),
               "mean_NDBI": float(g["NDBI"].mean()), "mean_NDVI": float(g["NDVI"].mean()),
               "in_current_test_spatial": dist in spatial_now}
        for nm, col in pred_cols.items():
            mt = eu.metrics(y, g[col].to_numpy(float))
            rec[f"rmse_{nm}"], rec[f"mae_{nm}"], rec[f"bias_{nm}"] = mt["rmse"], mt["mae"], mt["bias"]
        rec["skill_vs_persistence"] = 1 - rec["rmse_Mô hình"] / rec["rmse_Persistence"]
        rows.append(rec)
    by = pd.DataFrame(rows).sort_values(["window", "rmse_Mô hình"])
    eu.db_columns(by).to_csv(os.path.join(args.data_dir, "spatial_cv_by_district.csv"), index=False)

    srows = []
    for wkey, wlabel in WINDOWS:
        g = R[R["window"] == wkey]
        b = by[by["window"] == wkey]
        if g.empty or b.empty:
            continue
        y = g["LST_target"].to_numpy(float)
        rec = {"window": wkey, "label": wlabel, "n_districts": len(b), "n_rows": len(g)}
        for nm, col in pred_cols.items():
            rec[f"pooled_rmse_{nm}"] = eu.metrics(y, g[col].to_numpy(float))["rmse"]
        rec["district_rmse_mean"] = float(b["rmse_Mô hình"].mean())
        rec["district_rmse_median"] = float(b["rmse_Mô hình"].median())
        rec["district_rmse_worst"] = float(b["rmse_Mô hình"].max())
        rec["worst_district"] = b.loc[b["rmse_Mô hình"].idxmax(), "district"]
        rec["districts_beating_persistence"] = int((b["rmse_Mô hình"] < b["rmse_Persistence"]).sum())
        rec["districts_beating_damped"] = int((b["rmse_Mô hình"] < b["rmse_Damped anomaly"]).sum())
        srows.append(rec)
    summ = pd.DataFrame(srows)
    eu.db_columns(summ).to_csv(os.path.join(args.data_dir, "spatial_cv_summary.csv"), index=False)

    print("\n" + "=" * 100)
    print(" SPATIAL CV — RMSE (°C) khi quận bị giữ ra (mô hình không thấy quận đó lúc huấn luyện)")
    print("=" * 100)
    for _, r in summ.iterrows():
        print(f"\n [{r['label']}] {int(r['n_districts'])} quận, {int(r['n_rows']):,} dòng")
        print(f"   RMSE gộp: " + " | ".join(f"{nm} {r['pooled_rmse_' + nm]:.3f}" for nm in pred_cols))
        print(f"   Theo quận (mô hình): TB {r['district_rmse_mean']:.3f} | trung vị {r['district_rmse_median']:.3f} | tệ nhất {r['district_rmse_worst']:.3f} ({r['worst_district']})")
        print(f"   Số quận mô hình thắng Persistence: {int(r['districts_beating_persistence'])}/{int(r['n_districts'])} | thắng Damped anomaly: "
              f"{int(r['districts_beating_damped'])}/{int(r['n_districts'])}")
    if len(summ) == 2:
        a, b2 = summ.iloc[0], summ.iloc[1]
        sk_a = 1 - a["pooled_rmse_Mô hình"] / a["pooled_rmse_Persistence"]
        sk_b = 1 - b2["pooled_rmse_Mô hình"] / b2["pooled_rmse_Persistence"]
        gap = sk_a - sk_b
        print(f"\n Skill so với Persistence: 'cùng giai đoạn' {sk_a:+.2f} | 'sau giai đoạn' {sk_b:+.2f} (chênh {gap:+.2f}; "
              f"so theo skill vì Persistence khó/dễ khác nhau giữa hai cửa sổ năm khác nhau)")
        print(" → " + ("cửa sổ 'cùng giai đoạn' LẠC QUAN hơn rõ rệt (nghi rò rỉ qua ngày chung giữa các quận); báo cáo số 'sau giai đoạn' cho khả năng chuyển sang quận mới."
                       if gap > 0.05 else "không thấy dấu hiệu rò rỉ qua ngày chung rõ rệt (hai cửa sổ vẫn khác năm/mùa nên chỉ mang tính chỉ báo)."))

    wref = "future" if (by["window"] == "future").any() else "overlap"
    b = by[by["window"] == wref].sort_values("mean_NDBI", ascending=False)
    print(f"\n QUẬN THEO MỨC ĐÔ THỊ HOÁ (NDBI TB giảm dần), cửa sổ '{dict(WINDOWS)[wref]}':")
    print(f"   {'Quận':<34} {'NDBI':>6} {'NDVI':>6} {'RMSE':>6} {'Persist.':>8} {'Skill':>6}  Ghi chú")
    for _, r in b.iterrows():
        note = "đang ở Test không gian" if r["in_current_test_spatial"] else ""
        print(f"   {r['district']:<34} {r['mean_NDBI']:>6.2f} {r['mean_NDVI']:>6.2f} {r['rmse_Mô hình']:>6.2f} {r['rmse_Persistence']:>8.2f} {r['skill_vs_persistence']:>+6.2f}  {note}")
    cand = b[~b["in_current_test_spatial"]].head(3)["district"].tolist()
    if cand:
        print("\n   → Ứng viên đô thị/ven đô thị để thêm vào Test không gian (chạy lại 04 với --spatial_districts, vd \"Can Gio,Cu Chi,Dau Tieng,Go Vap\"): " + "; ".join(c.split(' / ')[-1] for c in cand))

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.join(args.data_dir, "figures"), exist_ok=True)
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": eu.AXIS, "figure.facecolor": eu.SURFACE, "axes.facecolor": eu.SURFACE})
        bb = by[by["window"] == wref].sort_values("rmse_Mô hình")
        both = by.pivot(index="district", columns="window", values="rmse_Mô hình").dropna() if by["window"].nunique() == 2 else None
        ncols = 2 if both is not None and len(both) else 1
        fig, axes = plt.subplots(1, ncols, figsize=(6.2 + (4.6 if ncols == 2 else 0), 0.3 * len(bb) + 2.0), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.5, 1] if ncols == 2 else [1]})
        ax = axes[0][0]
        ypos = np.arange(len(bb))[::-1]
        for yy, (_, r) in zip(ypos, bb.iterrows()):
            ax.plot([r["rmse_Mô hình"], r["rmse_Persistence"]], [yy, yy], color=eu.GRID, lw=2.5, zorder=1)
            ax.plot(r["rmse_Persistence"], yy, "o", color=eu.MUTED, ms=6.5, mec=eu.SURFACE, mew=1.2, zorder=2)
            ax.plot(r["rmse_Mô hình"], yy, "o", color=eu.PALETTE[0], ms=7.5, mec=eu.SURFACE, mew=1.2, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels([d.split(" / ")[-1] + (" ✱" if s else "") for d, s in zip(bb["district"], bb["in_current_test_spatial"])], color=eu.INK2)
        ax.set_xlabel("RMSE (°C)", color=eu.INK2)
        ax.set_title(f"Leave-district-out — cửa sổ '{dict(WINDOWS)[wref]}'", color=eu.INK, fontsize=10.5, loc="left")
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], marker="o", ls="", color=eu.PALETTE[0], mec=eu.SURFACE, ms=8, label="Mô hình"),
                   Line2D([], [], marker="o", ls="", color=eu.MUTED, mec=eu.SURFACE, ms=7, label="Persistence"),
                   Line2D([], [], marker="$✱$", ls="", color=eu.INK2, ms=8, label="đang ở Test không gian")]
        ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=eu.INK2, loc="upper right")
        ax.grid(axis="x", color=eu.GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(eu.AXIS)
        ax.tick_params(colors=eu.INK2, left=False)
        if ncols == 2:
            a2 = axes[0][1]
            lim = [min(both.min()) - 0.1, max(both.max()) + 0.1]
            a2.plot(lim, lim, color=eu.AXIS, lw=1, ls="--")
            a2.scatter(both["overlap"], both["future"], s=46, color=eu.PALETTE[1], edgecolor=eu.SURFACE, linewidth=1.2, zorder=3)
            a2.set_xlim(lim)
            a2.set_ylim(lim)
            a2.set_xlabel("RMSE 'cùng giai đoạn' (°C)", color=eu.INK2)
            a2.set_ylabel("RMSE 'sau giai đoạn' (°C)", color=eu.INK2)
            a2.set_title("Mỗi chấm là một quận\n(trên đường chéo = 'cùng giai đoạn' lạc quan hơn)", color=eu.INK, fontsize=9.5, loc="left")
            eu.style_axes(a2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.data_dir, "figures", "spatial_cv.png"), dpi=160)
        plt.close(fig)
        print(f"\n🖼  Ảnh: {os.path.join(args.data_dir, 'figures', 'spatial_cv.png')}")

    print(f"\n✅ Đã lưu spatial_cv_by_district.csv, spatial_cv_summary.csv trong {args.data_dir}")


if __name__ == "__main__":
    main()
