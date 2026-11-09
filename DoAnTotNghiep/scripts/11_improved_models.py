"""
BƯỚC 11 — BASELINE MẠNH + DỰ BÁO PHẦN CHÊNH (ANOMALY/DELTA) + LightGBM/CatBoost
=================================================================================
Nâng cấp khoa học nhóm 1 (chạy SAU 04, độc lập với 05/06 — không sửa/ghi đè chúng):

  (1) BASELINE MẠNH HƠN Persistence (đều chỉ học từ Train, không rò rỉ):
        Persistence            ŷ = LST_t
        Climatology            ŷ = TB LST theo (đơn vị × tháng của ngày nhãn); đơn vị chưa thấy (Test không gian)
                               lùi về (nhóm NDVI × tháng) -> (tháng) -> toàn cục
        Anomaly persistence    ŷ = clim(nhãn) + [LST_t − clim(t)]        (dị thường giữ nguyên)
        Damped anomaly         ŷ = clim(nhãn) + α(delta_t)·[LST_t − clim(t)]   (α khớp trên Train theo nhóm delta_t)
      Mô hình ML chỉ thuyết phục khi đánh bại CẢ các baseline này, không chỉ Persistence.

  (2) DỰ BÁO PHẦN CHÊNH thay vì giá trị tuyệt đối (--variants, mỗi variant = kiểu target:bộ feature):
        absolute:base   y = LST_target                 (như 06)
        delta:base      y = LST_target − LST_t
        anomaly:aug     y = LST_target − clim(nhãn); feature thêm clim_t, anom_t, clim_target
      (mặc định chạy 3 variant trên; thêm vd delta:aug, anomaly:base bằng --variants)

  (3) THÊM LightGBM, CatBoost cạnh XGBoost, Random Forest (--models xgb,lgbm,cat,rf).

Quy ước: Validation = early stopping + CHỌN mô hình (theo RMSE Val); các tập Test chỉ BÁO CÁO, không dùng để chọn.
Mọi sai số tính trên LST gốc (°C) — dự báo phần chênh được cộng ngược lại phần nền trước khi chấm.

Đầu ra (../data/):
    metrics_improved.csv        baseline + mọi (họ mô hình × variant) × 3 kịch bản: rmse/mae/r2/bias/skill so với Persistence
    predictions_compare.csv     dự báo của 4 baseline + variant TỐT NHẤT (theo Val) của mỗi họ mô hình — đầu vào của 13/14
    compare_meta.json           tên cột -> mô hình/variant, hệ số damped, cấu hình chạy
    best_model_v2.pkl           mô hình tốt nhất theo Val (kèm variant + climatology) — 12/13/10/16 nạp lại, KHÔNG train lại
    climatology_district_month.csv   bảng climatology (để vẽ/đẩy CSDL)
    models_v2/*.pkl            (chỉ với --save_all) mọi mô hình đã huấn luyện

Cách chạy:
    python 11_improved_models.py
    python 11_improved_models.py --models xgb,lgbm --variants absolute:base,delta:base,anomaly:aug
    python 11_improved_models.py --max_train_rows 150000 --rf_trees 100      # thử nhanh khi dữ liệu lớn
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_utils as eu  # noqa: E402


def slug(name: str) -> str:
    return "pred_" + name.replace(" ", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--models", default="xgb,lgbm,cat,rf", help="Các họ mô hình: xgb,lgbm,cat,rf")
    ap.add_argument("--variants", default="absolute:base,delta:base,anomaly:aug", help="kiểu_target:bộ_feature, phân tách bởi dấu phẩy")
    ap.add_argument("--min_clim_n", type=int, default=20, help="Số quan sát tối thiểu mỗi ô climatology")
    ap.add_argument("--max_train_rows", type=int, default=None, help="Lấy mẫu ngẫu nhiên Train khi fit (climatology vẫn dùng toàn bộ Train)")
    ap.add_argument("--rf_trees", type=int, default=200)
    ap.add_argument("--min_leaf", type=int, default=50,
                    help="Số dòng tối thiểu mỗi lá (các dòng cùng ngày tương quan mạnh -> lá nhỏ ghi nhớ từng ngày; 12 sẽ dò giá trị này)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--save_all", action="store_true", help="Lưu mọi mô hình vào models_v2/ (RF có thể rất nặng)")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in eu.MODEL_LABELS]
    if bad:
        raise SystemExit(f"❌ Họ mô hình không hợp lệ: {bad}. Hợp lệ: {list(eu.MODEL_LABELS)}")

    print("Đang nạp dữ liệu...")
    sp = eu.load_splits(args.data_dir)
    train, val = sp["train"], sp["val"]
    eval_sets = {k: sp[k] for _, k in eu.SCENARIOS if len(sp[k])}

    clim = eu.Climatology(args.min_clim_n).fit(train)
    damped = eu.DampedAnomaly(clim).fit(train)
    print(f"Climatology: {len(clim.t_district)} ô (đơn vị×tháng) | lùi: {len(clim.t_surface)} ô (nhóm NDVI×tháng), {len(clim.t_month)} tháng")
    print("Hệ số damped α theo nhóm delta_t (Train): " + ", ".join(f"{k}={v:.2f}" for k, v in damped.alpha.items()))
    variants = eu.parse_variants(args.variants, clim)

    rows, preds = [], {}   # preds[(tên, variant)] = {split: mảng}

    def add_row(kind, model, variant, scen_name, y, p, n_trees=None, secs=None):
        m = eu.metrics(y, p)
        rows.append({"kind": kind, "model": model, "variant": variant, "scenario": scen_name, **m, "n_trees": n_trees, "fit_seconds": secs})
        return m

    # ---------- baseline
    base_preds = {split: eu.baseline_predictions(df, clim, damped) for split, df in {"val": val, **eval_sets}.items()}
    for name in ["Persistence", "Climatology", "Anomaly persistence", "Damped anomaly"]:
        preds[(name, "—")] = {}
        for scen_name, key in eu.SCENARIOS:
            df = val if key == "val" else eval_sets.get(key)
            if df is None or len(df) == 0:
                continue
            p = base_preds[key][name]
            preds[(name, "—")][key] = p
            add_row("baseline", name, "—", scen_name, df[eu.TARGET], p)

    # ---------- mô hình ML
    tr_fit = train
    if args.max_train_rows and len(train) > args.max_train_rows:
        tr_fit = train.sample(args.max_train_rows, random_state=args.seed).sort_values(["date_t", "point_id"]).reset_index(drop=True)
        print(f"   (--max_train_rows: fit trên {len(tr_fit):,}/{len(train):,} dòng Train)")

    fitted = {}
    for v in variants:
        trp, vap = v.prepare(tr_fit), v.prepare(val)
        ev_p = {k: v.prepare(df) for k, df in eval_sets.items()}
        for mk in models:
            label = eu.MODEL_LABELS[mk]
            print(f"\n[{label} | {v.name}] đang huấn luyện...")
            model, n_trees, secs = eu.fit_model(mk, trp[v.features], v.y(trp), vap[v.features], v.y(vap), seed=args.seed,
                                                n_jobs=args.n_jobs, rf_trees=args.rf_trees, min_leaf=args.min_leaf)
            preds[(label, v.name)] = {}
            for scen_name, key in eu.SCENARIOS:
                dfp = vap if key == "val" else ev_p.get(key)
                if dfp is None or len(dfp) == 0:
                    continue
                p = v.predict(model, dfp)
                preds[(label, v.name)][key] = p
                add_row("ml", label, v.name, scen_name, dfp[eu.TARGET], p, n_trees, secs)
            val_rmse = next(r["rmse"] for r in rows if r["model"] == label and r["variant"] == v.name and r["scenario"] == "Validation")
            print(f"   xong {secs:.0f}s, {n_trees if n_trees else args.rf_trees} cây | Val RMSE = {val_rmse:.3f} °C")
            fitted[(mk, v.name)] = (model, v, n_trees, val_rmse)
            if args.save_all:
                os.makedirs(os.path.join(args.data_dir, "models_v2"), exist_ok=True)
                eu.save_pickle(os.path.join(args.data_dir, "models_v2", f"{mk}_{v.name.replace(':', '_')}.pkl"),
                               eu.make_bundle(model, v, mk, {"n_trees": n_trees, "val_rmse": val_rmse}))

    # ---------- kết quả
    res = pd.DataFrame(rows)
    pers = res[res["model"] == "Persistence"].set_index("scenario")["rmse"]
    res["skill_vs_persistence"] = 1 - res["rmse"] / res["scenario"].map(pers)
    ml = res[res["kind"] == "ml"]
    best_idx = None
    if len(ml):
        best_idx = ml[ml["scenario"] == "Validation"]["rmse"].idxmin()
        res["selected_by_val"] = res.index.map(lambda i: bool(
            res.loc[i, "kind"] == "ml" and res.loc[i, "model"] == res.loc[best_idx, "model"] and res.loc[i, "variant"] == res.loc[best_idx, "variant"]))
    else:
        res["selected_by_val"] = False

    print("\n" + "=" * 112)
    print(" KẾT QUẢ (RMSE/MAE tính bằng °C trên LST gốc; skill = 1 − RMSE/RMSE_Persistence, càng cao càng tốt)")
    print("=" * 112)
    print(f"| {'Mô hình':<21} | {'Variant':<14} | {'Kịch bản':<10} | {'RMSE':>6} | {'MAE':>6} | {'R²':>6} | {'Bias':>6} | {'Skill':>6} | {'n':>8} |")
    print("-" * 112)
    order = {"baseline": 0, "ml": 1}
    for scen_name, _ in eu.SCENARIOS:
        sub = res[res["scenario"] == scen_name].sort_values(["kind", "rmse"], key=lambda s: s.map(order) if s.name == "kind" else s)
        for _, r in sub.iterrows():
            star = " ★" if r["selected_by_val"] else ""
            print(f"| {r['model']:<21} | {r['variant']:<14} | {r['scenario']:<10} | {r['rmse']:>6.3f} | {r['mae']:>6.3f} | {r['r2']:>6.3f} | "
                  f"{r['bias']:>+6.2f} | {r['skill_vs_persistence']:>+6.2f} | {int(r['n']):>8,} |{star}")
        print("-" * 112)
    print("(★ = mô hình tốt nhất theo RMSE trên Validation. Không chọn theo Test.)")

    os.makedirs(args.data_dir, exist_ok=True)
    res.to_csv(os.path.join(args.data_dir, "metrics_improved.csv"), index=False)
    clim.t_district.to_csv(os.path.join(args.data_dir, "climatology_district_month.csv"), index=False)

    # ---------- predictions_compare.csv: baseline + variant tốt nhất (theo Val) của mỗi họ
    chosen = [(n, "—", "baseline") for n in ["Persistence", "Climatology", "Anomaly persistence", "Damped anomaly"]]
    for mk in models:
        cands = [(val_rmse, vn) for (k, vn), (_, _, _, val_rmse) in fitted.items() if k == mk]
        if cands:
            chosen.append((eu.MODEL_LABELS[mk], min(cands)[1], "ml"))
    meta_cols, out_parts = [], []
    keep = ["point_id", "lon", "lat", "date_t", "date_target", "delta_t", "ADM1_NAME", "ADM2_NAME", "LST_t", "LST_target"]
    for _, key in eu.SCENARIOS:
        df = val if key == "val" else eval_sets.get(key)
        if df is None or len(df) == 0:
            continue
        part = df[[c for c in keep if c in df.columns]].copy()
        part.insert(0, "split", key)
        for name, vn, kind in chosen:
            part[slug(name)] = preds[(name, vn)][key]
        out_parts.append(part)
    pd.concat(out_parts, ignore_index=True).to_csv(os.path.join(args.data_dir, "predictions_compare.csv"), index=False)
    for name, vn, kind in chosen:
        meta_cols.append({"column": slug(name), "name": name, "variant": vn, "kind": kind})

    eu.save_json(os.path.join(args.data_dir, "compare_meta.json"), {
        "created": eu.now_iso(), "columns": meta_cols, "damped_alpha": damped.alpha, "min_clim_n": args.min_clim_n,
        "variants_run": [v.name for v in variants], "models_run": models, "max_train_rows": args.max_train_rows, "seed": args.seed, "min_leaf": args.min_leaf,
    })

    if best_idx is not None:
        mk = next(k for k, lab in eu.MODEL_LABELS.items() if lab == res.loc[best_idx, "model"])
        model, v, n_trees, val_rmse = fitted[(mk, res.loc[best_idx, "variant"])]
        eu.save_pickle(os.path.join(args.data_dir, "best_model_v2.pkl"),
                       eu.make_bundle(model, v, mk, {"n_trees": n_trees, "val_rmse": val_rmse, "damped_alpha": damped.alpha}))
        print(f"\n★ Tốt nhất theo Validation: {res.loc[best_idx, 'model']} / {res.loc[best_idx, 'variant']} (Val RMSE {val_rmse:.3f} °C) -> best_model_v2.pkl")

    print(f"✅ Đã lưu metrics_improved.csv, predictions_compare.csv, compare_meta.json, climatology_district_month.csv trong {args.data_dir}")
    print("   Tiếp theo: 12 (dò siêu tham số XGBoost) -> 13 (khoảng tin cậy + sai số theo delta_t) -> 14 (CI/kiểm định/Moran) -> 15 (CV theo quận).")


if __name__ == "__main__":
    main()
