"""
BƯỚC 8 — ABLATION STUDY (mở rộng) + SHAP
===========================================
CẬP NHẬT 2026-09-09 — đồng bộ với 03_prepare_features.py bản mới:
  - `delta_t` GIỜ NẰM TRONG FEATURES (đúng đề cương mới nhất — bài toán
    "dự báo linh hoạt", xem 03_prepare_features.py). Vì delta_t là feature
    trung tâm của thiết kế mới, BỔ SUNG thêm 1 kịch bản ablation để CHỨNG
    MINH BẰNG SỐ nó thực sự có đóng góp: "Đầy đủ, không có delta_t" so với
    "Đầy đủ (Full Model)".
  - ERA5-Land không còn lag t-1/t-3/t-7 (03 đã bỏ) -> era5_cols giờ chỉ còn
    5 biến tại t, không phải 5x4=20 cột như bản trước.
  - classify_season()/classify_surface() chuyển sang dùng chung từ
    03_prepare_features.py (thay vì định nghĩa riêng ở đây) — để
    07_compute_suhii.py dùng lại đúng 1 định nghĩa "bề mặt xây dựng/thực
    vật/mặt nước", tránh 2 nơi lệch nhau.

Giữ nguyên từ bản trước (đã đúng, không cần sửa):
  (1) Ablation theo nhóm biến (viễn thám / khí tượng / đầy đủ)
  (2) So sánh mùa mưa (5-11) / mùa khô (12-4) — đánh giá riêng model "Đầy
      đủ" trên từng nhóm mẫu theo mùa.
  (3) So sánh theo loại bề mặt (Nước / Xây dựng / Thực vật) — đánh giá
      riêng model "Đầy đủ" trên từng nhóm.
  Đây là ĐÁNH GIÁ PHÂN TẦNG (stratified evaluation) trên cùng 1 model đã
  huấn luyện đầy đủ — không train model riêng cho từng mùa/bề mặt (giữ
  đúng tinh thần "1 model duy nhất, phân tích ở đâu nó mạnh/yếu" thường
  dùng trong ablation study).
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


def _fix_xgboost_base_score_for_shap(model):
    """Vá lỗi tương thích xgboost>=3.0 + shap cũ: xgboost mới lưu base_score
    dạng chuỗi JSON list (vd '[47.74468]') thay vì số thực đơn — khiến
    shap.TreeExplainer ném ValueError: could not convert string to float
    khi ép float() trực tiếp (lỗi đã biết, xem shap/shap#4288).

    SỬA LỖI (2026-10-02): bản trước dò theo đường dẫn cố định
    config["learner"]["learner_model_param"]["base_score"] rồi round-trip
    save_config/load_config — ĐàXÁC NHẬN KHÔNG ăn thua trên máy thật (traceback
    vẫn y hệt lỗi gốc "could not convert string to float: '[4.7923004E1]'"),
    nhiều khả năng do bản xgboost cài trên máy lồng cấu trúc config khác đi,
    KeyError bị try/except nuốt mất nên fix không chạy, hoặc load_config()
    không "dính" khi SHAP tự gọi lại save_config() ở nơi khác.

    Cách sửa mới, chắc ăn hơn: KHÔNG sửa trong config rồi hy vọng nó được đọc
    lại đúng chỗ — thay vào đó GHI ĐÈ trực tiếp phương thức save_config() của
    CHÍNH booster này (chỉ instance này, không đụng tới class/booster khác),
    để BẤT KỲ chỗ nào bên trong SHAP gọi booster.save_config() (dù gọi ở đâu,
    qua đường nào) đều nhận lại chuỗi JSON đã sửa sẵn base_score — không phụ
    thuộc phải biết trước SHAP/xgboost lồng key này ở path nào. Dò ĐỆ QUY
    toàn bộ config tìm MỌI key "base_score" dạng chuỗi JSON-array, không chỉ
    1 đường dẫn cố định, nên vẫn ăn được dù cấu trúc đổi khác giữa các bản."""
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


def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"| {label:<28} | {rmse:<7.3f} | {mae:<7.3f} | {r2:<7.3f} |")
    return rmse, mae, r2


def main():
    data_dir = "../data"
    print("Đang nạp dữ liệu...")
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test_temporal.csv"))
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file CSV. Vui lòng kiểm tra lại Bước 4.")
        return

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    era5_cols = prepare_features.ERA5_VARS  # chỉ còn 5 biến tại t, không lag
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
        y_pred = model.predict(test_df[feats])
        evaluate(y_test, y_pred, name)
        if name == "Đầy đủ (Full Model)":
            full_model = model
            X_test_full = test_df[feats]
    print("=" * 65)

    # ============ (2) SO SÁNH MÙA MƯA / MÙA KHÔ ============
    print("\n" + "=" * 65)
    print(" (2) ĐÁNH GIÁ MODEL ĐẦY ĐỦ THEO MÙA (trên tập Test)")
    print("=" * 65)
    print(f"| {'Mùa':<28} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} |")
    print("-" * 65)
    season = prepare_features.classify_season(test_df["date_t"])
    for s in ["Mùa mưa", "Mùa khô"]:
        mask = season == s
        if mask.sum() > 0:
            evaluate(y_test[mask], full_model.predict(X_test_full[mask]), f"{s} (n={mask.sum()})")
        else:
            print(f"| {s:<28} | không có mẫu trong tập Test |")
    print("=" * 65)

    # ============ (3) SO SÁNH THEO LOẠI BỀ MẶT ============
    print("\n" + "=" * 65)
    print(" (3) ĐÁNH GIÁ MODEL ĐẦY ĐỦ THEO LOẠI BỀ MẶT (trên tập Test)")
    print("=" * 65)
    print(f"| {'Loại bề mặt':<28} | {'RMSE':<7} | {'MAE':<7} | {'R²':<7} |")
    print("-" * 65)
    surface = prepare_features.classify_surface(test_df)
    for s in ["Mặt nước", "Xây dựng", "Thực vật"]:
        mask = surface == s
        if mask.sum() > 0:
            evaluate(y_test[mask], full_model.predict(X_test_full[mask]), f"{s} (n={mask.sum()})")
        else:
            print(f"| {s:<28} | không có mẫu trong tập Test |")
    print("=" * 65)

    # ============ SHAP ============
    print("\nĐang tính toán giá trị SHAP (Feature Importance)...")
    _fix_xgboost_base_score_for_shap(full_model)
    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(X_test_full)

    plt.ioff()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_full, show=False, max_display=15)
    plt.title("Biểu đồ SHAP Feature Importance (Đồ án Tốt nghiệp)", fontsize=14, fontweight="bold", pad=20)
    shap_out_path = os.path.join(data_dir, "shap_summary_plot.png")
    plt.savefig(shap_out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"✅ Đã lưu biểu đồ SHAP tại: {shap_out_path}")
    print(f"   (với {len(FEATURES)} feature, gồm cả delta_t — giới hạn hiển thị 15 biến quan trọng nhất)")


if __name__ == "__main__":
    main()
