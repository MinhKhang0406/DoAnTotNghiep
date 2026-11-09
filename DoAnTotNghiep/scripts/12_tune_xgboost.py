"""
BƯỚC 12 — DÒ SIÊU THAM SỐ XGBoost BẰNG OPTUNA TRÊN WALK-FORWARD CÓ PURGE
=========================================================================
Thay cho cấu hình cố định + early stopping của 06/11. Quy trình KHÔNG rò rỉ:

  * Dò chỉ trên TRAIN, bằng walk-forward theo ngày có purge (cùng logic 06): mỗi fold huấn luyện trên các khối ngày
    trước và xác thực trên khối kế tiếp; loại khỏi train mọi dòng có date_target >= ngày đầu khối xác thực.
  * Variant cần climatology (anomaly / bộ feature aug): climatology được fit LẠI trên phần train của TỪNG fold
    (không dùng ngày thuộc khối xác thực).
  * Mục tiêu Optuna = RMSE trung bình qua các fold (°C, trên LST gốc). Early stopping trong fold dùng chính khối
    xác thực của fold đó để chọn số cây (giống nhau cho mọi trial nên so sánh công bằng; số cây cuối cùng lấy trung vị
    best_iteration các fold của trial tốt nhất rồi KHÔNG dùng lại — mô hình cuối cùng early-stop trên Validation 2024).
  * Sau khi dò: fit lại bằng tham số tốt nhất trên toàn bộ Train, early stopping trên Validation (2024),
    rồi BÁO CÁO Val/Test so với cấu hình mặc định của 11. Test không bao giờ tham gia chọn tham số.

Đầu ra (../data/):
    best_params_xgb.json       tham số tốt nhất + RMSE walk-forward + cấu hình chạy
    optuna_trials.csv          mọi trial (tham số, RMSE từng fold)
    metrics_tuned.csv          mặc định vs đã dò × (Validation, Thời gian, Không gian)
    best_model_v2_tuned.pkl    XGBoost đã dò (cùng format bundle với best_model_v2.pkl; 13/10/16 nạp được bằng --model_file)

Cách chạy:
    python 12_tune_xgboost.py                                  # variant lấy từ XGBoost tốt nhất trong compare_meta.json
    python 12_tune_xgboost.py --variant delta:base --n_trials 60 --folds 3
    python 12_tune_xgboost.py --max_train_rows 150000          # thử nhanh khi dữ liệu lớn
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_utils as eu  # noqa: E402


def pick_default_variant(data_dir: str) -> str:
    meta = eu.load_json(os.path.join(data_dir, "compare_meta.json"))
    if meta:
        for c in meta.get("columns", []):
            if c.get("name") == "XGBoost" and c.get("variant") not in (None, "—"):
                return c["variant"]
    return "delta:base"


def suggest_params(trial, min_leaf_hi: int) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, min_leaf_hi, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--variant", default=None, help="kiểu_target:bộ_feature (mặc định: variant XGBoost tốt nhất theo 11)")
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--folds", type=int, default=3, help="Số fold walk-forward")
    ap.add_argument("--min_clim_n", type=int, default=20)
    ap.add_argument("--max_train_rows", type=int, default=None)
    ap.add_argument("--min_leaf_hi", type=int, default=2000, help="Cận trên của min_child_weight khi dò")
    ap.add_argument("--timeout", type=int, default=None, help="Giới hạn thời gian dò (giây)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sp = eu.load_splits(args.data_dir)
    train, val = sp["train"], sp["val"]
    eval_sets = {k: sp[k] for _, k in eu.SCENARIOS if len(sp[k])}
    vname = args.variant or pick_default_variant(args.data_dir)
    print(f"Variant dò: {vname}")

    tr_all = train
    if args.max_train_rows and len(train) > args.max_train_rows:
        tr_all = train.sample(args.max_train_rows, random_state=args.seed).sort_values(["date_t", "point_id"]).reset_index(drop=True)
        print(f"   (--max_train_rows: dò trên {len(tr_all):,}/{len(train):,} dòng Train)")

    folds = eu.purged_walk_forward_splits(tr_all, args.folds)
    print(f"{len(folds)} fold walk-forward: " + " | ".join(f"train={len(a):,}/val={len(b):,}" for a, b in folds))

    # Chuẩn bị dữ liệu từng fold MỘT LẦN (climatology fit trên train của fold)
    prepared = []
    for tr_idx, va_idx in folds:
        tr_df, va_df = tr_all.iloc[tr_idx], tr_all.iloc[va_idx]
        clim_f = eu.Climatology(args.min_clim_n).fit(tr_df) if eu.Variant.parse(vname).needs_clim else None
        v = eu.Variant.parse(vname, clim_f)
        trp, vap = v.prepare(tr_df), v.prepare(va_df)
        prepared.append((v, trp[v.features], v.y(trp), vap[v.features], v.y(vap)))

    trial_rows = []

    def objective(trial):
        params = suggest_params(trial, args.min_leaf_hi)
        rmses = []
        for v, Xtr, ytr, Xva, yva in prepared:
            m, n_trees, _ = eu.fit_model("xgb", Xtr, ytr, Xva, yva, seed=args.seed, n_jobs=args.n_jobs, params=params)
            rmses.append(float(np.sqrt(np.mean((np.asarray(m.predict(Xva)) - yva) ** 2))))
        trial.set_user_attr("fold_rmse", rmses)
        trial_rows.append({"trial": trial.number, **params, **{f"rmse_fold{i + 1}": r for i, r in enumerate(rmses)}, "rmse_mean": float(np.mean(rmses))})
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    # đưa cấu hình mặc định của 11 vào làm trial đầu tiên để luôn có mốc so sánh
    study.enqueue_trial({"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8,
                         "reg_lambda": 1.0, "reg_alpha": 0.001, "gamma": 0.0})
    print(f"Đang dò {args.n_trials} trial...")
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)
    best = study.best_trial
    base_rmse = trial_rows[0]["rmse_mean"] if trial_rows else float("nan")
    print(f"★ Tốt nhất: trial {best.number} | RMSE walk-forward {best.value:.4f} °C (mặc định {base_rmse:.4f} °C)")
    print("   " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in best.params.items()))

    pd.DataFrame(trial_rows).sort_values("rmse_mean").to_csv(os.path.join(args.data_dir, "optuna_trials.csv"), index=False)

    # ---------- fit cuối trên toàn bộ Train, early stopping trên Validation
    clim = eu.Climatology(args.min_clim_n).fit(train)
    v = eu.Variant.parse(vname, clim if eu.Variant.parse(vname).needs_clim else None)
    tr_fit = tr_all
    trp, vap = v.prepare(tr_fit), v.prepare(val)
    ev_p = {k: v.prepare(df) for k, df in eval_sets.items()}

    rows = []
    default_params = {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8}
    finals = {}
    for tag, params in (("Mặc định (11)", default_params), ("Đã dò (12)", best.params)):
        m, n_trees, secs = eu.fit_model("xgb", trp[v.features], v.y(trp), vap[v.features], v.y(vap), seed=args.seed, n_jobs=args.n_jobs, params=params)
        finals[tag] = (m, n_trees)
        for scen_name, key in eu.SCENARIOS:
            dfp = vap if key == "val" else ev_p.get(key)
            if dfp is None or len(dfp) == 0:
                continue
            rows.append({"config": tag, "variant": v.name, "scenario": scen_name, "n_trees": n_trees, **eu.metrics(dfp[eu.TARGET], v.predict(m, dfp))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.data_dir, "metrics_tuned.csv"), index=False)

    print("\n" + "=" * 86)
    print(f" XGBoost {v.name}: MẶC ĐỊNH vs ĐÃ DÒ (Test chỉ báo cáo, không dùng để chọn tham số)")
    print("=" * 86)
    print(f"| {'Cấu hình':<14} | {'Kịch bản':<10} | {'Cây':>5} | {'RMSE':>6} | {'MAE':>6} | {'R²':>6} | {'Bias':>6} | {'n':>8} |")
    for _, r in res.iterrows():
        print(f"| {r['config']:<14} | {r['scenario']:<10} | {int(r['n_trees']):>5} | {r['rmse']:>6.3f} | {r['mae']:>6.3f} | {r['r2']:>6.3f} | {r['bias']:>+6.2f} | {int(r['n']):>8,} |")

    m_best, n_trees = finals["Đã dò (12)"]
    eu.save_pickle(os.path.join(args.data_dir, "best_model_v2_tuned.pkl"),
                   eu.make_bundle(m_best, v, "xgb", {"n_trees": n_trees, "tuned_params": best.params, "walk_forward_rmse": best.value}))
    eu.save_json(os.path.join(args.data_dir, "best_params_xgb.json"), {
        "created": eu.now_iso(), "variant": v.name, "best_params": best.params, "walk_forward_rmse": best.value,
        "default_walk_forward_rmse": base_rmse, "n_trials": len(study.trials), "folds": len(folds), "seed": args.seed,
        "final_n_trees": n_trees, "max_train_rows": args.max_train_rows,
    })
    print(f"\n✅ Đã lưu best_params_xgb.json, optuna_trials.csv, metrics_tuned.csv, best_model_v2_tuned.pkl trong {args.data_dir}")
    print("   Muốn dùng mô hình đã dò cho 13/10/16: thêm --model_file best_model_v2_tuned.pkl")


if __name__ == "__main__":
    main()
