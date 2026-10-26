"""
BƯỚC 6 & 7 — RANDOM FOREST (walk-forward) VÀ XGBOOST (Early Stopping)
=======================================================================
CẬP NHẬT 2026-10-19 (review rò rỉ dữ liệu khi dò siêu tham số):

  1. TimeSeriesSplit trên DÒNG đã sắp theo date_t có 2 lỗ hổng so với đề cương
     ("các fold huấn luyện luôn có thời điểm sớm hơn fold xác thực"):
       (a) cắt fold giữa chừng 1 ngày -> cùng 1 date_t vừa nằm ở fold train vừa
           ở fold xác thực (các điểm cùng ngày có LST rất giống nhau);
       (b) dòng train có NHÃN (date_target) rơi vào giai đoạn của fold xác thực
           -> mô hình đã "nhìn" LST của giai đoạn cần dự báo.
     Giờ dùng walk-forward theo NGÀY có PURGE: chia danh sách ngày t thành các khối
     liên tiếp; fold k huấn luyện trên các khối trước, xác thực trên khối kế tiếp,
     và LOẠI khỏi train mọi dòng có date_target >= ngày đầu của khối xác thực.
  2. Đề cương: "Tập Validation (2024) dùng cho Early Stopping của XGBoost VÀ để
     chọn siêu tham số cuối cùng". Bản cũ RF không dùng val. Giờ: lấy top-3 bộ
     tham số theo walk-forward CV, huấn luyện lại trên toàn Train, chọn bộ có RMSE
     thấp nhất trên Validation. Test chỉ dùng đúng 1 lần ở cuối.
  3. GridSearchCV(n_jobs=-1) + RF(n_jobs=-1) lồng nhau gây quá tải CPU: giờ RF trong
     lưới dùng n_jobs=1, lưới chạy song song.
  4. LƯU metrics ra ../data/metrics_ml.csv (rmse/mae/r2 trên Val và 2 kịch bản Test)
     để 09 đẩy vào bảng model_metrics cho WebGIS.

Giữ nguyên: LƯU model đã tune ra .pkl (kèm đúng danh sách feature) để 09/10 dùng
ĐÚNG model này, không train lại bản khác.
"""

import importlib.util
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prepare_features", os.path.join(_THIS_DIR, "03_prepare_features.py")
)
prepare_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_features)

FEATURES = prepare_features.FEATURE_COLS
TARGET = "LST_target"
RESULTS = []


def evaluate_model(y_true, y_pred, model_name, scenario):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"| {model_name:<20} | {scenario:<15} | {rmse:<7.3f} | {mae:<7.3f} | {r2:<7.3f} |")
    RESULTS.append({"model": model_name, "scenario": scenario, "rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))})
    return rmse, mae, r2


def purged_walk_forward_splits(df: pd.DataFrame, n_splits: int = 3):
    """Walk-forward theo NGÀY t có purge theo date_target. Trả về list (train_idx, val_idx)
    (chỉ số vị trí) dùng trực tiếp làm cv= của GridSearchCV.

    Ví dụ n_splits=3 -> chia ngày thành 4 khối B0..B3:
        fold1: train B0           | val B1
        fold2: train B0+B1        | val B2
        fold3: train B0+B1+B2     | val B3
    Dòng train có date_target >= ngày đầu của khối val bị loại (purge)."""
    if "date_target" not in df.columns:
        print("   ⚠️  Không có cột date_target -> dùng TimeSeriesSplit thường (có thể rò rỉ nhẹ giữa các fold).")
        return list(TimeSeriesSplit(n_splits=n_splits).split(df))

    d_t = pd.to_datetime(df["date_t"]).to_numpy()
    d_n = pd.to_datetime(df["date_target"]).to_numpy()
    dates = np.unique(d_t)  # đã sắp tăng dần
    blocks = np.array_split(dates, n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_dates = np.concatenate(blocks[: k + 1])
        val_dates = blocks[k + 1]
        val_start = val_dates.min()
        tr_mask = np.isin(d_t, train_dates) & (d_n < val_start)
        va_mask = np.isin(d_t, val_dates)
        tr_idx, va_idx = np.where(tr_mask)[0], np.where(va_mask)[0]
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        splits.append((tr_idx, va_idx))
    if not splits:
        raise SystemExit("❌ Không tạo được fold walk-forward nào — dữ liệu Train quá ít ngày.")
    return splits


def main():
    data_dir = "../data"
    print("Đang nạp dữ liệu...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_temp_df = pd.read_csv(os.path.join(data_dir, "test_temporal.csv"))
    test_spat_df = pd.read_csv(os.path.join(data_dir, "test_spatial.csv"))

    missing = [c for c in FEATURES + [TARGET] if c not in train_df.columns]
    if missing:
        raise SystemExit(f"❌ train.csv thiếu cột {missing} — chạy lại 04_build_dataset.py bản mới (--rebuild_raw).")
    if len(train_df) == 0 or len(val_df) == 0:
        raise SystemExit("❌ Train hoặc Validation RỖNG — xem lại Bước 4 trước khi huấn luyện.")

    train_df = train_df.sort_values(["date_t", "point_id"] if "point_id" in train_df.columns else ["date_t"]).reset_index(drop=True)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    scenarios = [("Thời gian", test_temp_df), ("Không gian", test_spat_df)]

    print("\n" + "=" * 75)
    print(" BÁO CÁO KẾT QUẢ MÔ HÌNH HỌC MÁY PHI TUYẾN (RF & XGBOOST)")
    print("=" * 75)
    print(f"| {'Mô hình':<20} | {'Kịch bản Test':<15} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} |")
    print("-" * 75)

    # ================= RANDOM FOREST =================
    print("Đang huấn luyện Random Forest (GridSearchCV + walk-forward theo ngày có purge)...")
    splits = purged_walk_forward_splits(train_df, n_splits=3)
    print(f"   {len(splits)} fold walk-forward; kích thước (train/val): {[(len(a), len(b)) for a, b in splits]}")
    rf_param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=1),
        param_grid=rf_param_grid, cv=splits, scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    cv_res = pd.DataFrame(rf_grid.cv_results_).sort_values("rank_test_score")
    print("   Xếp hạng theo walk-forward CV (RMSE):")
    for _, r in cv_res.iterrows():
        print(f"     {str(r['params']):<45} CV-RMSE={-r['mean_test_score']:.3f}")

    # Chọn siêu tham số CUỐI CÙNG trên Validation trong top-3 của CV (đúng đề cương)
    best_rf, best_val_rmse, best_params = None, np.inf, None
    for params in cv_res["params"].head(3):
        m = RandomForestRegressor(random_state=42, n_jobs=-1, **params).fit(X_train, y_train)
        v = float(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))
        print(f"   Val RMSE {str(params):<45} = {v:.3f}")
        if v < best_val_rmse:
            best_rf, best_val_rmse, best_params = m, v, params
    print(f"-> Tham số RF chọn theo Validation: {best_params} (Val RMSE={best_val_rmse:.3f})")
    evaluate_model(y_val, best_rf.predict(X_val), "Random Forest", "Validation")
    for sc, d in scenarios:
        if len(d):
            evaluate_model(d[TARGET], best_rf.predict(d[FEATURES]), "Random Forest", sc)
    print("-" * 75)

    # ================= XGBOOST =================
    print("Đang huấn luyện XGBoost (Early Stopping trên tập Val)...")
    xgb_model = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        early_stopping_rounds=50, random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"-> XGBoost dừng sớm ở vòng lặp thứ: {xgb_model.best_iteration}")
    evaluate_model(y_val, xgb_model.predict(X_val), "XGBoost", "Validation")
    for sc, d in scenarios:
        if len(d):
            evaluate_model(d[TARGET], xgb_model.predict(d[FEATURES]), "XGBoost", sc)
    print("=" * 75)

    # ================= LƯU MODEL + METRICS =================
    os.makedirs(data_dir, exist_ok=True)
    rf_path = os.path.join(data_dir, "best_model_rf.pkl")
    xgb_path = os.path.join(data_dir, "best_model_xgb.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump({"model": best_rf, "features": FEATURES}, f)
    with open(xgb_path, "wb") as f:
        pickle.dump({"model": xgb_model, "features": FEATURES}, f)
    metrics_path = os.path.join(data_dir, "metrics_ml.csv")
    pd.DataFrame(RESULTS).to_csv(metrics_path, index=False)
    print(f"\n✅ Đã lưu model (kèm đúng danh sách feature):\n   {rf_path}\n   {xgb_path}\n✅ Đã lưu metrics: {metrics_path}")
    print("   -> 09_export_to_postgis.py / 10_build_inference_grid.py sẽ nạp lại đúng các file này, KHÔNG train lại.")


if __name__ == "__main__":
    main()
