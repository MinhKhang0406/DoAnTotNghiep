"""
BƯỚC 5 — MÔ HÌNH CƠ SỞ (BASELINE): Persistence + Linear Regression
====================================================================
Bài toán: "dự báo linh hoạt cho kỳ quan sát khả dụng tiếp theo" — delta_t là 1
FEATURE độc lập (không phải "t+7 cố định"). Persistence baseline (dự báo = LST
tại t) giữ nguyên ý nghĩa: "lấy giá trị quan trắc gần nhất làm giá trị dự báo cho
kỳ tiếp theo", không phụ thuộc horizon. (Đề cương mục 4 / Tuần 6 còn ghi
"LST(t+7) = LST(t)" — nên sửa thành "LST(t+Δt) = LST(t)" cho khớp bài toán mới.)

CẬP NHẬT 2026-10-19:
  - LƯU kết quả ra ../data/metrics_baseline.csv (trước chỉ in ra màn hình) để
    09_export_to_postgis.py đẩy vào bảng model_metrics -> WebGIS Tuần 13 vẽ được
    biểu đồ so sánh RMSE/MAE/R² giữa các mô hình mà không phải gõ tay số liệu.
  - Danh sách feature vẫn lấy trực tiếp từ 03_prepare_features.FEATURE_COLS.
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

RESULTS = []


def evaluate_model(y_true, y_pred, model_name, scenario):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"| {model_name:<19} | {scenario:<15} | {rmse:<7.3f} | {mae:<7.3f} | {r2:<7.3f} |")
    RESULTS.append({"model": model_name, "scenario": scenario, "rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))})
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
            f"❌ train.csv thiếu cột {missing} — dataset này có thể từ bản 04 CŨ. "
            f"Chạy lại 04_build_dataset.py bản mới với --rebuild_raw."
        )
    for name, d in [("test_temporal.csv", test_temp_df), ("test_spatial.csv", test_spat_df)]:
        if len(d) == 0:
            print(f"⚠️  {name} RỖNG — bỏ qua kịch bản tương ứng (xem cảnh báo ở Bước 4).")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]

    print("\n" + "=" * 70)
    print(" BÁO CÁO KẾT QUẢ MÔ HÌNH CƠ SỞ (BASELINE MODELS)")
    print("=" * 70)
    print(f"| {'Mô hình':<19} | {'Kịch bản Test':<15} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} |")
    print("-" * 70)

    scenarios = [("Thời gian", test_temp_df), ("Không gian", test_spat_df)]
    for sc, d in scenarios:
        if len(d):
            evaluate_model(d[TARGET], d["LST_t"], "Persistence", sc)
    print("-" * 70)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    for sc, d in scenarios:
        if len(d):
            evaluate_model(d[TARGET], lr_model.predict(d[FEATURES]), "Linear Regression", sc)
    print("=" * 70)

    out = os.path.join(data_dir, "metrics_baseline.csv")
    pd.DataFrame(RESULTS).to_csv(out, index=False)
    print(f"\nSố feature đầu vào: {len(FEATURES)} — {FEATURES}")
    print(f"✅ Đã lưu {out}")
    print("=> Random Forest / XGBoost ở bước sau phải đánh bại các con số RMSE/MAE này.")


if __name__ == "__main__":
    main()
