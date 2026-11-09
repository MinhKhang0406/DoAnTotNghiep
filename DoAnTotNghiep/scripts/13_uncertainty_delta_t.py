"""
BƯỚC 13 — KHOẢNG TIN CẬY CHO DỰ BÁO + SAI SỐ THEO delta_t + KIỂM TRA delta_t CÓ LÀM PROXY CHO MÙA
==================================================================================================
Nạp mô hình đã chọn (best_model_v2.pkl của 11, hoặc best_model_v2_tuned.pkl của 12 qua --model_file) — KHÔNG train lại
mô hình điểm. Gồm 3 phần:

  A. KHOẢNG TIN CẬY (mặc định mức 90%):
       conformal : ŷ ± q(delta_t)           q = phân vị conformal của |y − ŷ| trên Validation, RIÊNG từng nhóm delta_t
       cqr       : [lo − q, hi + q]         lo/hi = XGBoost phân vị 5%/95% (huấn luyện trên Train), q hiệu chuẩn trên Validation
     Hiệu chuẩn trên Validation (2024). Độ phủ được KIỂM TRA trên Test thời gian và Test không gian (không dùng để chỉnh).
     Lưu ý trung thực: bảo đảm độ phủ của conformal đòi hỏi hiệu chuẩn và dự báo cùng phân phối (hoán đổi được); các Test
     là năm 2025-2026 và quận chưa thấy nên độ phủ thực tế có thể thấp hơn mức đặt — đó là kết quả cần báo cáo, không phải lỗi.
     Early stopping đã dùng Validation nên phần dư Val hơi lạc quan (ảnh hưởng nhỏ vì chỉ chọn 1 tham số số-cây).
     delta_t ngoài khoảng Train (nhỏ hơn min hoặc lớn hơn phân vị 99%) được gắn cờ delta_t_out_of_range.

  B. SAI SỐ THEO NHÓM delta_t (1-3 | 4-8 | 9-16 | >16 ngày): RMSE/MAE của mô hình và 4 baseline trên Validation/Test.

  C. delta_t CÓ ĐANG LÀM PROXY CHO MÙA? delta_t do mây và mùa quyết định (mùa mưa ít ảnh hơn -> delta_t dài hơn) nên mô hình
     có thể dùng nó thay cho mùa thay vì cho 'khoảng cách thời gian' thật. Ba phép kiểm tra trên Validation:
       1) η² của tháng với delta_t (bao nhiêu % phương sai delta_t được giải thích bởi tháng);
       2) hoán vị delta_t TOÀN CỤC vs hoán vị delta_t TRONG CÙNG THÁNG: nếu hoán vị trong tháng gần như không làm hỏng mô
          hình nhưng hoán vị toàn cục làm hỏng nhiều -> mô hình dùng delta_t chủ yếu như nhãn mùa;
       3) phụ thuộc riêng phần (partial dependence) của dự báo theo delta_t.
     (Chỉ đổi cột delta_t; với bộ feature 'aug', clim_target vẫn giữ nguyên nên PD phản ánh đúng phần delta_t đóng góp.)

Đầu ra (../data/ và ../data/figures/):
    interval_calibration.json     phân vị hiệu chuẩn theo nhóm (conformal & cqr) + khoảng delta_t Train — 10b/16 nạp lại
    quantile_model_v2.pkl         XGBoost phân vị cho CQR
    interval_eval.csv             độ phủ + độ rộng trung bình theo kịch bản × phương pháp × nhóm delta_t
    interval_predictions.csv      dự báo điểm + khoảng (conformal & cqr) cho Validation/Test
    error_by_delta_t.csv          RMSE/MAE theo kịch bản × nhóm delta_t × (mô hình + 4 baseline)
    delta_t_diagnostic.json       kết quả 3 phép kiểm tra phần C
    partial_dependence_delta_t.csv
    figures/error_by_delta_t.png, interval_coverage.png, delta_t_diagnostic.png

Cách chạy:
    python 13_uncertainty_delta_t.py
    python 13_uncertainty_delta_t.py --model_file best_model_v2_tuned.pkl --coverage 0.8
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_utils as eu  # noqa: E402


def interval_stats(y, out: pd.DataFrame, lo_col: str, hi_col: str) -> dict:
    y = np.asarray(y, dtype=float)
    lo, hi = out[lo_col].to_numpy(float), out[hi_col].to_numpy(float)
    return {"n": int(len(y)), "coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float(np.mean(hi - lo))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--model_file", default="best_model_v2.pkl")
    ap.add_argument("--coverage", type=float, default=0.9, help="Mức phủ danh nghĩa (0.9 = khoảng 90%%)")
    ap.add_argument("--min_cal", type=int, default=200, help="Số mẫu hiệu chuẩn tối thiểu mỗi nhóm delta_t")
    ap.add_argument("--min_leaf", type=int, default=50)
    ap.add_argument("--n_perm", type=int, default=5, help="Số lần hoán vị cho phần C")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()
    if not 0.5 <= args.coverage < 1:
        raise SystemExit("❌ --coverage phải trong [0.5, 1).")

    sp = eu.load_splits(args.data_dir)
    train, val = sp["train"], sp["val"]
    sets = {"val": val, **{k: sp[k] for k in ("test_temporal", "test_spatial") if len(sp[k])}}
    bundle = eu.load_bundle(args.data_dir, args.model_file)
    v = eu.Variant.from_dict(bundle["variant"])
    print(f"Mô hình: {bundle['model_name']} / {v.name} (từ {args.model_file})")

    # =================== A. KHOẢNG TIN CẬY ===================
    pred_val = eu.predict_bundle(bundle, val)
    yv = val[eu.TARGET].to_numpy(float)
    conf = eu.fit_binned_conformal(np.abs(yv - pred_val), val["delta_t"], args.coverage, args.min_cal)

    print("Huấn luyện XGBoost phân vị cho CQR...")
    trp = v.prepare(train)
    qm = eu.fit_quantile_xgb(trp[v.features], v.y(trp), args.coverage, args.seed, args.n_jobs, params={"min_child_weight": args.min_leaf})
    qbundle = {"model": qm, "variant": bundle["variant"], "coverage": args.coverage, "created": eu.now_iso()}
    lo_r, hi_r = eu.predict_quantiles(qbundle, val)
    cqr = eu.fit_binned_conformal(np.maximum(lo_r - yv, yv - hi_r), val["delta_t"], args.coverage, args.min_cal)

    dts = train["delta_t"].to_numpy(float)
    calib = {
        "created": eu.now_iso(), "coverage": args.coverage, "model_file": args.model_file, "model_name": bundle["model_name"],
        "variant": v.name, "calibration_split": "val", "conformal": conf, "cqr": cqr,
        "train_delta_t": {"min": float(dts.min()), "max": float(np.quantile(dts, 0.99)), "max_observed": float(dts.max())},
        "delta_bins": eu.DELTA_LABELS,
    }
    eu.save_json(os.path.join(args.data_dir, "interval_calibration.json"), calib)
    eu.save_pickle(os.path.join(args.data_dir, "quantile_model_v2.pkl"), qbundle)
    print("Nửa-độ-rộng conformal theo nhóm delta_t (°C): " + ", ".join(f"{k}={x:.2f}" for k, x in conf["bins"].items()))
    print("Hiệu chỉnh CQR theo nhóm delta_t (°C):        " + ", ".join(f"{k}={x:+.2f}" for k, x in cqr["bins"].items()))

    rows, pred_parts = [], []
    for scen_name, key in eu.SCENARIOS:
        df = sets.get(key)
        if df is None or len(df) == 0:
            continue
        y = df[eu.TARGET].to_numpy(float)
        o_c = eu.predict_with_interval(bundle, calib, df, "conformal")
        o_q = eu.predict_with_interval(bundle, calib, df, "cqr", qbundle)
        bins = o_c["delta_t_bin"].to_numpy()
        for method, o in (("conformal", o_c), ("cqr", o_q)):
            rows.append({"scenario": scen_name, "method": method, "delta_bin": "tất cả", **interval_stats(y, o, "LST_lo", "LST_hi")})
            for lab in eu.DELTA_LABELS:
                m = bins == lab
                if m.sum():
                    rows.append({"scenario": scen_name, "method": method, "delta_bin": lab, **interval_stats(y[m], o[m], "LST_lo", "LST_hi")})
        keep = [c for c in ("point_id", "lon", "lat", "date_t", "date_target", "delta_t", "ADM1_NAME", "ADM2_NAME") if c in df.columns]
        part = df[keep].copy()
        part.insert(0, "split", key)
        part["LST_target"] = y
        part["LST_pred"] = o_c["LST_pred"].to_numpy()
        part["lo_conformal"], part["hi_conformal"] = o_c["LST_lo"].to_numpy(), o_c["LST_hi"].to_numpy()
        part["lo_cqr"], part["hi_cqr"] = o_q["LST_lo"].to_numpy(), o_q["LST_hi"].to_numpy()
        part["delta_t_bin"] = bins
        part["delta_t_out_of_range"] = o_c["delta_t_out_of_range"].to_numpy()
        pred_parts.append(part)
    ev = pd.DataFrame(rows)
    ev["nominal"] = args.coverage
    ev.to_csv(os.path.join(args.data_dir, "interval_eval.csv"), index=False)
    pd.concat(pred_parts, ignore_index=True).to_csv(os.path.join(args.data_dir, "interval_predictions.csv"), index=False)

    print("\n" + "=" * 78)
    print(f" ĐỘ PHỦ THỰC TẾ (mức đặt {args.coverage:.0%}) — Validation là tập hiệu chuẩn nên phủ ≈ mức đặt; hai Test mới là kiểm tra thật")
    print("=" * 78)
    print(f"| {'Kịch bản':<11} | {'Phương pháp':<10} | {'Nhóm':<7} | {'Phủ':>6} | {'Rộng TB (°C)':>12} | {'n':>8} |")
    for _, r in ev[ev["delta_bin"] == "tất cả"].iterrows():
        print(f"| {r['scenario']:<11} | {r['method']:<10} | {r['delta_bin']:<7} | {r['coverage']:>6.1%} | {r['mean_width']:>12.2f} | {int(r['n']):>8,} |")
    n_out = int(sum(p["delta_t_out_of_range"].sum() for p in pred_parts))
    print(f"(Số dòng có delta_t ngoài khoảng đã học [{calib['train_delta_t']['min']:.0f}, {calib['train_delta_t']['max']:.0f}] ngày: {n_out:,})")

    # =================== B. SAI SỐ THEO delta_t ===================
    clim = eu.Climatology(20).fit(train)
    damped = eu.DampedAnomaly(clim).fit(train)
    erows = []
    for scen_name, key in eu.SCENARIOS:
        df = sets.get(key)
        if df is None or len(df) == 0:
            continue
        y = df[eu.TARGET].to_numpy(float)
        series = {bundle["model_name"] + " (đã chọn)": eu.predict_bundle(bundle, df), **eu.baseline_predictions(df, clim, damped)}
        bins = eu.delta_bin(df["delta_t"].to_numpy(float))
        for name, p in series.items():
            for lab in eu.DELTA_LABELS:
                m = bins == lab
                if m.sum() >= 2:
                    mt = eu.metrics(y[m], p[m])
                    erows.append({"scenario": scen_name, "delta_bin": lab, "series": name, "rmse": mt["rmse"], "mae": mt["mae"], "bias": mt["bias"], "n": mt["n"]})
    err = pd.DataFrame(erows)
    err.to_csv(os.path.join(args.data_dir, "error_by_delta_t.csv"), index=False)
    print("\n RMSE (°C) theo nhóm delta_t — mô hình đã chọn vs Persistence (n = số dòng):")
    mname = bundle["model_name"] + " (đã chọn)"
    for scen_name, _ in eu.SCENARIOS:
        sub = err[err["scenario"] == scen_name]
        if sub.empty:
            continue
        cells = []
        for lab in eu.DELTA_LABELS:
            a = sub[(sub["delta_bin"] == lab) & (sub["series"] == mname)]
            b = sub[(sub["delta_bin"] == lab) & (sub["series"] == "Persistence")]
            if len(a) and len(b):
                cells.append(f"{lab}: {a['rmse'].iloc[0]:.2f} vs {b['rmse'].iloc[0]:.2f} (n={int(a['n'].iloc[0]):,})")
        print(f"   {scen_name:<11} " + " | ".join(cells))

    # =================== C. delta_t CÓ PHẢI PROXY CHO MÙA? ===================
    month_t = pd.to_datetime(train["date_t"]).dt.month.to_numpy()
    overall = float(np.mean(train["delta_t"]))
    ss_tot = float(np.sum((train["delta_t"].to_numpy(float) - overall) ** 2))
    ss_between = float(sum((month_t == m).sum() * (train["delta_t"].to_numpy(float)[month_t == m].mean() - overall) ** 2 for m in np.unique(month_t)))
    eta2 = ss_between / ss_tot if ss_tot > 0 else float("nan")
    by_month = {int(m): float(train["delta_t"].to_numpy(float)[month_t == m].mean()) for m in np.unique(month_t)}

    vp = v.prepare(val)
    base_rmse = eu.metrics(vp[eu.TARGET], v.predict(bundle["model"], vp))["rmse"]
    rng = np.random.default_rng(args.seed)
    m_val = pd.to_datetime(val["date_t"]).dt.month.to_numpy()

    def perm_rmse(within_month: bool) -> float:
        out = []
        for _ in range(args.n_perm):
            d = vp["delta_t"].to_numpy(float).copy()
            if within_month:
                for m in np.unique(m_val):
                    idx = np.where(m_val == m)[0]
                    d[idx] = rng.permutation(d[idx])
            else:
                d = rng.permutation(d)
            tmp = vp.copy()
            tmp["delta_t"] = d
            out.append(eu.metrics(tmp[eu.TARGET], v.predict(bundle["model"], tmp))["rmse"])
        return float(np.mean(out))

    r_glob, r_in = perm_rmse(False), perm_rmse(True)
    inc_glob, inc_in = r_glob - base_rmse, r_in - base_rmse

    grid = [1, 2, 4, 6, 8, 12, 16, 24, 32]
    pd_rows, ref = [], None
    for g in grid:
        tmp = vp.copy()
        tmp["delta_t"] = float(g)
        mean_pred = float(np.mean(v.predict(bundle["model"], tmp)))
        pd_rows.append({"delta_t": g, "mean_pred": mean_pred})
    pdf = pd.DataFrame(pd_rows)
    pdf["pd_centered"] = pdf["mean_pred"] - pdf["mean_pred"].iloc[0]
    pdf.to_csv(os.path.join(args.data_dir, "partial_dependence_delta_t.csv"), index=False)

    if inc_glob <= 0.005:
        verdict = "delta_t gần như KHÔNG được mô hình dùng (hoán vị toàn cục hầu như không đổi RMSE)."
    elif inc_in < 0.35 * inc_glob:
        verdict = ("delta_t có vẻ chủ yếu làm PROXY CHO MÙA: hoán vị trong cùng tháng ảnh hưởng nhỏ hơn nhiều so với hoán vị toàn cục "
                   "— nên ghi vào mục hạn chế và thử bỏ delta_t hoặc thay bằng cặp (tháng, delta_t).")
    else:
        verdict = "delta_t mang thông tin RIÊNG ngoài mùa (hoán vị trong tháng vẫn làm hỏng đáng kể) — phù hợp với vai trò đặc trưng khoảng thời gian."
    diag = {"created": eu.now_iso(), "eta2_month_on_delta_t": eta2, "mean_delta_t_by_month_train": by_month,
            "val_rmse_base": base_rmse, "val_rmse_perm_global": r_glob, "val_rmse_perm_within_month": r_in,
            "rmse_increase_global": inc_glob, "rmse_increase_within_month": inc_in, "n_perm": args.n_perm, "verdict": verdict,
            "pd_range_celsius": float(pdf["pd_centered"].max() - pdf["pd_centered"].min())}
    eu.save_json(os.path.join(args.data_dir, "delta_t_diagnostic.json"), diag)
    print("\n delta_t có phải proxy cho mùa? (Validation)")
    print(f"   η² (tháng -> delta_t) = {eta2:.2f} | RMSE gốc {base_rmse:.3f} | hoán vị toàn cục +{inc_glob:.3f} | hoán vị trong tháng +{inc_in:.3f} °C")
    print(f"   PD(delta_t) biến thiên {diag['pd_range_celsius']:.2f} °C trên lưới 1-32 ngày")
    print(f"   → {verdict}")

    # =================== VẼ ẢNH ===================
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.join(args.data_dir, "figures"), exist_ok=True)
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": eu.AXIS, "figure.facecolor": eu.SURFACE, "axes.facecolor": eu.SURFACE})
        scens = [s for s, k in eu.SCENARIOS if (k in sets)]
        names = [mname, "Persistence", "Damped anomaly", "Climatology"]

        fig, axes = plt.subplots(1, len(scens), figsize=(4.4 * len(scens), 3.9), sharey=True, squeeze=False)
        for ax, scen in zip(axes[0], scens):
            sub = err[err["scenario"] == scen]
            for i, nm in enumerate(names):
                s = sub[sub["series"] == nm].set_index("delta_bin").reindex(eu.DELTA_LABELS)
                ax.plot(eu.DELTA_LABELS, s["rmse"], color=eu.PALETTE[i], lw=2, marker="o", ms=6, mec=eu.SURFACE, mew=1.5, label=nm)
            ax.set_title(scen, color=eu.INK, fontsize=11, loc="left")
            ax.set_xlabel("delta_t (ngày)", color=eu.INK2)
            eu.style_axes(ax)
        axes[0][0].set_ylabel("RMSE (°C)", color=eu.INK2)
        axes[0][0].legend(frameon=False, fontsize=8.5, labelcolor=eu.INK2, loc="upper left")
        fig.suptitle("Sai số theo khoảng cách tới ảnh nhãn", color=eu.INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(args.data_dir, "figures", "error_by_delta_t.png"), dpi=160)
        plt.close(fig)

        tscens = [s for s, k in eu.SCENARIOS if k != "val" and k in sets]
        if tscens:
            fig, axes = plt.subplots(1, len(tscens), figsize=(4.8 * len(tscens), 3.9), sharey=True, squeeze=False)
            for ax, scen in zip(axes[0], tscens):
                sub = ev[(ev["scenario"] == scen) & (ev["delta_bin"] != "tất cả")]
                x = np.arange(len(eu.DELTA_LABELS))
                for j, (meth, col) in enumerate((("conformal", eu.PALETTE[0]), ("cqr", eu.PALETTE[1]))):
                    s = sub[sub["method"] == meth].set_index("delta_bin").reindex(eu.DELTA_LABELS)
                    ax.bar(x + (j - 0.5) * 0.36, s["coverage"].fillna(0), width=0.34, color=col, label=meth.upper() if meth == "cqr" else "Conformal",
                           linewidth=0)
                ax.axhline(args.coverage, color=eu.INK2, lw=1.2, ls="--", label=f"Mức đặt {args.coverage:.0%}")
                ax.set_xticks(x)
                ax.set_xticklabels(eu.DELTA_LABELS)
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("delta_t (ngày)", color=eu.INK2)
                ax.set_title(scen, color=eu.INK, fontsize=11, loc="left")
                eu.style_axes(ax)
            axes[0][0].set_ylabel("Độ phủ thực tế", color=eu.INK2)
            h, l = axes[0][0].get_legend_handles_labels()
            fig.suptitle("Độ phủ của khoảng tin cậy trên Test", color=eu.INK, fontsize=12, x=0.01, ha="left")
            fig.legend(h, l, frameon=False, fontsize=8.5, labelcolor=eu.INK2, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 0.99))
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(os.path.join(args.data_dir, "figures", "interval_coverage.png"), dpi=160)
            plt.close(fig)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
        months = sorted(by_month)
        a1.bar([str(m) for m in months], [by_month[m] for m in months], color=eu.PALETTE[0], width=0.62, linewidth=0)
        a1.set_xlabel("Tháng của ngày t", color=eu.INK2)
        a1.set_ylabel("delta_t trung bình (ngày)", color=eu.INK2)
        a1.set_title(f"delta_t theo tháng (η² = {eta2:.2f})", color=eu.INK, fontsize=11, loc="left")
        eu.style_axes(a1)
        a2.plot(pdf["delta_t"], pdf["pd_centered"], color=eu.PALETTE[1], lw=2, marker="o", ms=6, mec=eu.SURFACE, mew=1.5)
        a2.axhline(0, color=eu.GRID, lw=1)
        a2.set_xlabel("delta_t (ngày)", color=eu.INK2)
        a2.set_ylabel("Thay đổi dự báo TB (°C)", color=eu.INK2)
        a2.set_title("Phụ thuộc riêng phần vào delta_t", color=eu.INK, fontsize=11, loc="left")
        eu.style_axes(a2)
        fig.tight_layout()
        fig.savefig(os.path.join(args.data_dir, "figures", "delta_t_diagnostic.png"), dpi=160)
        plt.close(fig)
        print(f"\n🖼  Ảnh: {os.path.join(args.data_dir, 'figures')}")

    print(f"\n✅ Đã lưu interval_calibration.json, quantile_model_v2.pkl, interval_eval.csv, interval_predictions.csv, "
          f"error_by_delta_t.csv, delta_t_diagnostic.json, partial_dependence_delta_t.csv trong {args.data_dir}")


if __name__ == "__main__":
    main()
