# LST Thesis Pipeline — TP.HCM

Thứ tự thực hiện (theo đúng đề cương, không nhảy cóc để tránh phải làm lại):

| # | Nội dung | File |
|---|----------|------|
| 1 | Setup GEE + ranh giới hành chính THẬT của TP.HCM | `scripts/01_setup_gee_boundary.py` ✅ đã có |
| 2 | Thu thập Landsat 8/9 + Sentinel-2, lọc mây/bóng mây/lỗi cảm biến | *sắp tới* |
| 3 | Tính NDVI/NDBI/NDWI, đồng bộ ERA5-Land (không gian + thời gian) | *sắp tới* |
| 4 | Ghép mẫu (t, t+7), xuất dataset dạng bảng, chia Train/Val/Test theo năm | *sắp tới* |
| 5 | Baseline: Persistence + Linear Regression | *sắp tới* |
| 6 | Random Forest (walk-forward tuning) | *sắp tới* |
| 7 | XGBoost (early stopping) | *sắp tới* |
| 8 | Đánh giá Test (2 kịch bản) + ablation + SHAP | *sắp tới* |
| 9 | Gắn kết quả thật vào WebGIS đã build | *sắp tới* |

## Cài môi trường (1 lần)

```bash
conda env create -f environment.yml
conda activate lst-thesis
earthengine authenticate
```

## Chạy Bước 1

```bash
cd scripts
python 01_setup_gee_boundary.py --project YOUR_GEE_PROJECT_ID
```

Kết quả: file `data/hcmc_boundary.geojson` chứa ranh giới quận/huyện thật của
TP.HCM — dùng để:
- thay placeholder hình vuông trong WebGIS
- làm vùng clip ảnh vệ tinh ở Bước 2
- chọn 2-3 quận/huyện làm test set không gian ở Bước 4

Chạy xong gửi lại output (đặc biệt là danh sách tên quận/huyện in ra) để
qua Bước 2.
