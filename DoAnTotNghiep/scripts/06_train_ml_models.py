"""
BƯỚC 6 & 7 — RANDOM FOREST (TimeSeriesSplit) VÀ XGBOOST (Early Stopping)
==========================================================================
CẬP NHẬT 2026-09-09: `delta_t` giờ NẰM TRONG feature (đúng đề cương mới
nhất — bài toán "dự báo linh hoạt", xem 03_prepare_features.py) thay vì bị
loại như bản trước. Không cần sửa gì ở file này — FEATURES lấy trực tiếp
từ prepare_features.FEATURE_COLS nên tự động cập nhật theo.

Vẫn giữ nguyên từ bản trước (đã đúng):
  - LƯU LẠI model đã tune ra .pkl — để 09_export_to_postgis.py dùng ĐÚNG
    model này khi đẩy dự báo lên WebGIS, thay vì tự train lại 1 bản khác
    với tham số mặc định (tránh model báo cáo kết quả khác model thực tế
    chạy trên WebGIS).
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


def evaluate_model(y_true, y_pred, model_name, scenario):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"| {model_name:<20} | {scenario:<15} | {rmse:<7.3f} | {mae:<7.3f} | {r2:<7.3f} |")
    return rmse, mae, r2


def main():
    data_dir = "../data"
    print("Đang nạp dữ liệu...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_temp_df = pd.read_csv(os.path.join(data_dir, "test_temporal.csv"))
    test_spat_df = pd.read_csv(os.path.join(data_dir, "test_spatial.csv"))

    train_df = train_df.sort_values("date_t").reset_index(drop=True)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    X_test_temp, y_test_temp = test_temp_df[FEATURES], test_temp_df[TARGET]
    X_test_spat, y_test_spat = test_spat_df[FEATURES], test_spat_df[TARGET]

    print("\n" + "=" * 75)
    print(" BÁO CÁO KẾT QUẢ MÔ HÌNH HỌC MÁY PHI TUYẾN (RF & XGBOOST)")
    print("=" * 75)
    print(f"| {'Mô hình':<20} | {'Kịch bản Test':<15} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} |")
    print("-" * 75)

    # ================= RANDOM FOREST =================
    print("Đang huấn luyện Random Forest (GridSearchCV + TimeSeriesSplit)...")
    tscv = TimeSeriesSplit(n_splits=3)
    rf_param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20, None]}
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(
        estimator=rf_base, param_grid=rf_param_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"-> Tham số RF tốt nhất: {rf_grid.best_params_}")

    evaluate_model(y_test_temp, best_rf.predict(X_test_temp), "Random Forest", "Thời gian")
    evaluate_model(y_test_spat, best_rf.predict(X_test_spat), "Random Forest", "Không gian")
    print("-" * 75)

    # ================= XGBOOST =================
    print("Đang huấn luyện XGBoost (Early Stopping trên tập Val)...")
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"-> XGBoost dừng sớm ở vòng lặp thứ: {xgb_model.best_iteration}")

    evaluate_model(y_test_temp, xgb_model.predict(X_test_temp), "XGBoost", "Thời gian")
    evaluate_model(y_test_spat, xgb_model.predict(X_test_spat), "XGBoost", "Không gian")
    print("=" * 75)

    # ================= LƯU MODEL =================
    os.makedirs(data_dir, exist_ok=True)
    rf_path = os.path.join(data_dir, "best_model_rf.pkl")
    xgb_path = os.path.join(data_dir, "best_model_xgb.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump({"model": best_rf, "features": FEATURES}, f)
    with open(xgb_path, "wb") as f:
        pickle.dump({"model": xgb_model, "features": FEATURES}, f)
    print(f"\n✅ Đã lưu model (kèm đúng danh sách feature đã dùng) vào:\n   {rf_path}\n   {xgb_path}")
    print("   -> 09_export_to_postgis.py sẽ nạp lại đúng 2 file này, KHÔNG train lại từ đầu.")


if __name__ == "__main__":
    main()
