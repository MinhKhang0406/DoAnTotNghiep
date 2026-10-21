"""
BƯỚC 5 — MÔ HÌNH CƠ SỞ (BASELINE): Persistence + Linear Regression
====================================================================
CẬP NHẬT 2026-09-09: đề cương bản mới nhất xác định bài toán "dự báo linh
hoạt cho kỳ quan sát khả dụng tiếp theo", với delta_t là 1 FEATURE độc lập
(không phải "t+7 cố định" như hiểu ở bản trước) — xem chi tiết thay đổi ở
đầu 03_prepare_features.py và 04_build_dataset.py. File này KHÔNG cần sửa
logic gì thêm vì FEATURES lấy trực tiếp từ prepare_features.FEATURE_COLS,
tự động có delta_t sau khi 03/04 được cập nhật và chạy lại.

Persistence baseline (dự báo = LST tại t) vẫn giữ nguyên ý nghĩa gốc —
"lấy giá trị quan trắc gần nhất làm giá trị dự báo cho kỳ tiếp theo",
không cần đổi theo horizon cố định hay linh hoạt.

Danh sách feature lấy trực tiếp từ 03_prepare_features.FEATURE_COLS để
không bao giờ lệch với cột thật sự có trong train/val/test*.csv.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    print(f"| {model_name:<19} | {scenario:<15} | {rmse:<7.3f} | {mae:<7.3f} | {r2:<7.3f} |")
    return rmse, mae, r2


def main():
    data_dir = "../data"
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_temp_df = pd.read_csv(os.path.join(data_dir, "test_temporal.csv"))
        test_spat_df = pd.read_csv(os.path.join(data_dir, "test_spatial.csv"))
    except FileNotFoundError:
        print("❌ Không tìm thấy dữ liệu. Hãy chắc chắn bạn đã chạy xong Bước 4.")
        return

    missing = [c for c in FEATURES + [TARGET] if c not in train_df.columns]
    if missing:
        raise SystemExit(
            f"❌ train.csv thiếu cột {missing} — dataset này có thể từ bản 04 CŨ (chưa có đủ "
            f"biến ERA5 + lag). Chạy lại 04_build_dataset.py bản mới với --rebuild_raw."
        )

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test_temp, y_test_temp = test_temp_df[FEATURES], test_temp_df[TARGET]
    X_test_spat, y_test_spat = test_spat_df[FEATURES], test_spat_df[TARGET]

    print("\n" + "=" * 70)
    print(" BÁO CÁO KẾT QUẢ MÔ HÌNH CƠ SỞ (BASELINE MODELS)")
    print("=" * 70)
    print(f"| {'Mô hình':<19} | {'Kịch bản Test':<15} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} |")
    print("-" * 70)

    evaluate_model(y_test_temp, test_temp_df["LST_t"], "Persistence", "Thời gian")
    evaluate_model(y_test_spat, test_spat_df["LST_t"], "Persistence", "Không gian")
    print("-" * 70)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_lr_temp = lr_model.predict(X_test_temp)
    evaluate_model(y_test_temp, y_pred_lr_temp, "Linear Regression", "Thời gian")

    y_pred_lr_spat = lr_model.predict(X_test_spat)
    evaluate_model(y_test_spat, y_pred_lr_spat, "Linear Regression", "Không gian")
    print("=" * 70)

    print(f"\nSố feature đầu vào: {len(FEATURES)} — {FEATURES}")
    print("=> Random Forest / XGBoost ở bước sau phải đánh bại các con số RMSE/MAE này.")


if __name__ == "__main__":
    main()
