"""
BƯỚC 8 — ABLATION STUDY (mở rộng) + SHAP
===========================================
Đánh giá PHÂN TẦNG trên cùng 1 model đã huấn luyện đầy đủ (không train riêng
từng mùa/bề mặt): (1) ablation theo nhóm biến (+ kịch bản "không có delta_t"),
(2) mùa mưa (5-11) / mùa khô (12-4), (3) loại bề mặt (Nước / Xây dựng / Thực vật),
(4) SHAP.

CẬP NHẬT 2026-10-19:
  - LƯU kết quả ra ../data/ablation_results.csv và ../data/shap_importance.csv
    (trước chỉ in màn hình + 1 ảnh PNG) để WebGIS Tuần 13 vẽ biểu đồ Feature
    Importance/so sánh nhóm biến từ CSDL (09 đẩy vào các bảng ablation_results,
    shap_importance).
  - MÙA tính theo ngày NHÃN (date_target, ngày được dự báo) khi có cột đó — đúng ý
    "hiệu năng dự báo mùa mưa vs mùa khô"; bản cũ dùng date_t (ngày làm đầu vào).
  - Bỏ qua gọn gàng khi 1 nhóm trống (vd chưa có mẫu 'Xây dựng') thay vì crash.
  - Giữ nguyên: classify_season/classify_surface dùng chung từ 03; vá tương thích
    xgboost>=3 + shap (base_score dạng chuỗi JSON list).
"""

import importlib.util
import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def _fix_xgboost_base_score_for_shap(model):
    """Vá lỗi tương thích xgboost>=3.0 + shap: base_score lưu dạng chuỗi JSON list
    (vd '[47.74468]') khiến shap.TreeExplainer ném ValueError. Ghi đè save_config()
    của CHÍNH booster này, dò ĐỆ QUY mọi key "base_score" dạng chuỗi JSON-array —
    không phụ thuộc đường dẫn cấu hình cố định (xem shap/shap#4288)."""
    try:
        booster = model.get_booster()
        _original_save_config = booster.save_config

        def _fix_recursive(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "base_score" and isinstance(v, str) and v.strip().startswith("["):
                        obj[k] = str(float(json.loads(v)[0]))
                    else:
                        _fix_recursive(v)
            elif isinstance(obj, list):
                for item in obj:
                    _fix_recursive(item)

        def _patched_save_config():
            config = json.loads(_original_save_config())
            _fix_recursive(config)
            return json.dumps(config)

        booster.save_config = _patched_save_config
    except Exception:
        pass
    return model


def evaluate(y_true, y_pred, label, section):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"| {label:<28} | {rmse:<7.3f} | {mae:<7.3f} | {r2:<7.3f} |")
    RESULTS.append({"section": section, "group": label, "rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))})
    return rmse, mae, r2


def _strat(title, labels_order, strat_series, y_test, model, X_test, section):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)
    print(f"| {'Nhóm':<28} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} |")
    print("-" * 65)
    for s in labels_order:
        mask = np.asarray(strat_series == s)
        n = int(mask.sum())
        if n >= 2:
            evaluate(y_test[mask], model.predict(X_test[mask]), f"{s} (n={n})", section)
        else:
            print(f"| {s:<28} | không đủ mẫu trong tập Test (n={n}) |")
    print("=" * 65)


def main():
    data_dir = "../data"
    print("Đang nạp dữ liệu...")
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test_temporal.csv"))
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file CSV. Vui lòng kiểm tra lại Bước 4.")
        return
    if len(train_df) == 0 or len(test_df) == 0:
        raise SystemExit("❌ Train hoặc test_temporal RỖNG — xem lại Bước 4.")

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    era5_cols = prepare_features.ERA5_VARS  # 5 biến tại t, không lag
    features_no_delta = [f for f in FEATURES if f != "delta_t"]
    ablation_scenarios = {
        "Chỉ dùng Viễn thám": ["LST_t", "NDVI", "NDBI", "NDWI"],
        "Chỉ dùng Khí tượng": ["LST_t"] + era5_cols,
        "Đầy đủ, không có delta_t": features_no_delta,
        "Đầy đủ (Full Model)": FEATURES,
    }

    print("\n" + "=" * 65)
    print(" (1) ABLATION THEO NHÓM BIẾN")
    print("=" * 65)
    print(f"| {'Nhóm biến đầu vào':<28} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} |")
    print("-" * 65)

    full_model, X_test_full = None, None
    for name, feats in ablation_scenarios.items():
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(train_df[feats], y_train)
        evaluate(y_test, model.predict(test_df[feats]), name, "Nhóm biến")
        if name == "Đầy đủ (Full Model)":
            full_model, X_test_full = model, test_df[feats]
    print("=" * 65)

    # (2) Mùa — theo ngày NHÃN (ngày được dự báo) nếu có
    season_date_col = "date_target" if "date_target" in test_df.columns else "date_t"
    season = pd.Series(prepare_features.classify_season(test_df[season_date_col]), index=test_df.index)
    _strat(f"(2) MODEL ĐẦY ĐỦ THEO MÙA (mùa của {season_date_col}, tập Test)",
           ["Mùa mưa", "Mùa khô"], season, y_test, full_model, X_test_full, "Mùa")

    # (3) Loại bề mặt (phân loại tại t từ NDVI/NDBI/NDWI)
    surface = pd.Series(prepare_features.classify_surface(test_df), index=test_df.index)
    _strat("(3) MODEL ĐẦY ĐỦ THEO LOẠI BỀ MẶT (tập Test)",
           ["Mặt nước", "Xây dựng", "Thực vật"], surface, y_test, full_model, X_test_full, "Bề mặt")

    pd.DataFrame(RESULTS).to_csv(os.path.join(data_dir, "ablation_results.csv"), index=False)
    print(f"\n✅ Đã lưu {os.path.join(data_dir, 'ablation_results.csv')}")

    # ============ SHAP ============
    print("\nĐang tính toán giá trị SHAP (Feature Importance)...")
    _fix_xgboost_base_score_for_shap(full_model)
    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(X_test_full)

    imp = pd.DataFrame({
        "feature": X_test_full.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp["share_pct"] = (imp["mean_abs_shap"] / imp["mean_abs_shap"].sum() * 100).round(2)
    imp.to_csv(os.path.join(data_dir, "shap_importance.csv"), index=False)
    print(imp.to_string(index=False))

    plt.ioff()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_full, show=False, max_display=15)
    plt.title("Biểu đồ SHAP Feature Importance (Đồ án Tốt nghiệp)", fontsize=14, fontweight="bold", pad=20)
    shap_out_path = os.path.join(data_dir, "shap_summary_plot.png")
    plt.savefig(shap_out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ SHAP tại: {shap_out_path} và shap_importance.csv")
    print(f"   (với {len(FEATURES)} feature, gồm cả delta_t — giới hạn hiển thị 15 biến quan trọng nhất)")


if __name__ == "__main__":
    main()
