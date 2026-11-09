# Nhóm 1 — Nâng cấp khoa học cho đồ án WebGIS đảo nhiệt (bước 11 → 16)

Bộ script **chỉ thêm file mới**, không sửa và không ghi đè bất kỳ file/bảng nào của 01–10 (kể cả `best_model_xgb.pkl`, `metrics_ml.csv`, các bảng của 09).
Mô hình mới lưu riêng ở `best_model_v2*.pkl`.

## 1. Gói gồm gì

| File | Làm gì | Đầu ra chính (`../data/`) |
|---|---|---|
| `eval_utils.py` | Module dùng chung (không chạy trực tiếp): climatology, baseline, 3 kiểu target, 4 họ mô hình, conformal/CQR, walk-forward có purge | — |
| `11_improved_models.py` | Baseline mạnh (Climatology, Anomaly persistence, Damped anomaly) + dự báo phần chênh (delta / anomaly) + LightGBM, CatBoost | `metrics_improved.csv`, `predictions_compare.csv`, `compare_meta.json`, `best_model_v2.pkl`, `climatology_district_month.csv` |
| `12_tune_xgboost.py` | Optuna trên walk-forward có purge; so mặc định vs đã dò | `best_params_xgb.json`, `optuna_trials.csv`, `metrics_tuned.csv`, `best_model_v2_tuned.pkl` |
| `13_uncertainty_delta_t.py` | Khoảng tin cậy 90% (conformal + CQR), sai số theo nhóm delta_t, kiểm tra delta_t có là proxy cho mùa | `interval_calibration.json`, `quantile_model_v2.pkl`, `interval_eval.csv`, `interval_predictions.csv`, `error_by_delta_t.csv`, `delta_t_diagnostic.json`, `partial_dependence_delta_t.csv`, `figures/*.png` |
| `14_bootstrap_moran.py` | CI bootstrap theo khối ngày, so sánh từng cặp (Holm), Moran's I phần dư | `bootstrap_ci.csv`, `paired_tests.csv`, `moran_residuals.csv`, `residual_map_mean.csv`, `figures/*.png` |
| `15_spatial_cv.py` | Leave-district-out: RMSE từng quận khi quận đó bị giữ ra, hai cửa sổ thời gian | `spatial_cv_by_district.csv`, `spatial_cv_summary.csv`, `figures/spatial_cv.png` |
| `10b_add_intervals_to_grid.py` | Đọc `forecast_grid_<t>.csv` của 10, thêm dự báo v2 + khoảng tin cậy (không cần GEE) | `forecast_grid_v2_<t>.csv` |
| `16_export_to_postgis_v2.py` | Đẩy kết quả vào PostGIS, dùng lại hàm của 09 | 15 bảng mới (mục 5) |
| `tests/make_fake_data.py`, `tests/make_fake_grid.py` | Sinh dữ liệu GIẢ để thử luồng script mà không cần GEE | `../data_fake/` |

## 2. Cài đặt

1. Chép toàn bộ `*.py` (trừ thư mục `tests/` nếu không cần) vào `D:\DoAnTotNghiep\scripts\`, cùng chỗ với `03_prepare_features.py`.
2. Trong env đang chạy đồ án: `pip install -r requirements_nhom1.txt` (chỉ thêm `lightgbm`, `catboost`, `optuna`).
3. Cần dataset ở **schema mới của 04** (có `point_id`, `date_target`, `ADM1_NAME`…). Nếu `train.csv` thiếu các cột này, script báo rõ và yêu cầu chạy lại 04.

## 3. Thứ tự chạy (từ thư mục `scripts/`)

```bat
python 11_improved_models.py
python 12_tune_xgboost.py
python 13_uncertainty_delta_t.py --model_file best_model_v2.pkl
python 14_bootstrap_moran.py
python 15_spatial_cv.py
python 10b_add_intervals_to_grid.py
python 16_export_to_postgis_v2.py
```

- Dữ liệu lớn/máy chậm: thêm `--max_train_rows 300000` (11, 12, 15) và `--rf_trees 100`; 15 có `--folds 8` thay cho leave-one-district-out.
- Muốn dùng XGBoost đã dò cho khoảng tin cậy/lưới/CV: thêm `--model_file best_model_v2_tuned.pkl` ở **13, 10b, 15** (và chạy lại 13 trước 10b để hiệu chuẩn khớp mô hình).
- 14 đọc `predictions_compare.csv` của 11 (4 baseline + variant tốt nhất theo Val của mỗi họ); nếu muốn đưa cả bản đã dò vào so sánh, chạy 14 sau khi đã quyết định mô hình cuối.
- Thử luồng không cần dữ liệu thật: `python ..\tests\make_fake_data.py --out ..\data_fake`, rồi thêm `--data_dir ..\data_fake` cho từng bước.

## 4. Cách đọc kết quả khi viết báo cáo

- **Chọn mô hình = theo RMSE trên Validation (2024).** Hai Test chỉ để báo cáo; `selected_by_val` trong `metrics_improved.csv` đánh dấu mô hình được chọn.
- **Baseline phải đánh bại là Damped anomaly / Climatology**, không chỉ Persistence (bảng `metrics_improved.csv`, cột `skill_vs_persistence`).
- **Chênh lệch giữa các mô hình chỉ nêu khi CI không chứa 0** (`paired_tests.csv`, cột `p_holm` / `better`). Số dòng trong một ngày tương quan mạnh nên CI theo dòng sẽ hẹp giả; CI ở 14 lấy mẫu lại theo cả ngày.
- **Test không gian kiểu 04 có thể lạc quan.** ERA5 gần như giống nhau mọi điểm trong cùng ngày nên cây có thể nhớ "ngày" từ quận khác. 15 đo điều này (cột `time_window`: `overlap` vs `future`, so theo skill). Báo cáo số `future` cho khả năng chuyển sang quận mới.
- **delta_t có thể là proxy cho mùa** (mây/mùa quyết định delta_t). Xem `delta_t_diagnostic.json` (`verdict`) và ghi vào mục hạn chế.
- **Độ phủ khoảng tin cậy** phải báo cáo theo Test (không phải Validation: đó là tập hiệu chuẩn nên phủ ≈ mức đặt). Test là năm 2025–2026 và quận chưa thấy nên độ phủ có thể thấp hơn 90% — đó là kết quả, không phải lỗi.
- **Moran's I phần dư > 0 rõ** (`moran_residuals.csv`) nghĩa là còn cấu trúc không gian chưa giải thích. Đây là chẩn đoán, không phải kiểm định nhân quả.
- Ứng viên quận đô thị để đưa vào Test không gian: cuối bảng in của 15; sau đó chạy lại 04 với `--spatial_districts`.

## 5. Bảng PostGIS mới (16) và dùng khoảng tin cậy ở frontend

`lst_predictions_v2`*, `lst_forecast_grid_v2`*, `model_metrics_v2`, `model_tuning`, `model_rmse_ci`, `model_paired_tests`, `error_by_delta_t`, `delta_t_partial_dependence`, `interval_eval`, `interval_calibration`, `delta_t_diagnostic`, `spatial_cv_district`, `spatial_cv_summary`, `moran_residuals`, `residual_map_mean`* (\* = bảng điểm, EPSG:4326, chỉ mục GiST + ngày).

Cột cho frontend trong `lst_forecast_grid_v2`: `LST_pred_v2` (dự báo điểm), `LST_lo`, `LST_hi` (khoảng theo `interval_method`, mức `interval_coverage`), `delta_t_out_of_range` (true → khoảng kém tin cậy, nên hiển thị cảnh báo), `has_data`/`reason` (giữ nguyên từ 10). Ví dụ truy vấn:

```sql
SELECT point_id, delta_t, date_forecast, "LST_pred_v2", "LST_lo", "LST_hi", delta_t_out_of_range
FROM lst_forecast_grid_v2 WHERE has_data AND date_forecast = '2026-06-12';
```

(Tên cột có chữ hoa như `LST_pred_v2` cần đặt trong ngoặc kép, như ví dụ.)

## 6. Đã kiểm thử gì — và chưa

Đã chạy đủ 11 → 15, 10b, 16 trên **dữ liệu giả có cấu trúc giống dataset thật** (schema lấy từ `04_build_dataset.py`), với đúng `03_prepare_features.py` trong repo của bạn; Moran's I tự cài đã đối chiếu khớp `esda`; 16 đã ghi thật vào SQLite và một PostgreSQL 16 (các bảng không có hình học).

**Chưa chạy:**
- Trên dataset thật của bạn (CSV trong repo là schema cũ nên không dùng được). Con số trên dữ liệu giả **không có ý nghĩa khoa học**; thời gian chạy trên dữ liệu thật chưa đo.
- Đường `to_postgis` (bảng điểm) trên PostGIS thật: sandbox không có PostGIS. Mã dựng `GeoDataFrame` và chỉ mục theo đúng cách 09 đang làm, nhưng bạn nên chạy thử `python 16_export_to_postgis_v2.py --dry_run` rồi `--tables ci` (bảng không hình học) trước, sau đó mới `--tables predictions,grid`.

## 7. Sự cố thường gặp

- `❌ ... thiếu cột [...]` → dataset từ bản 04 cũ; chạy lại `04_build_dataset.py` (kèm `--rebuild_raw` như thông báo).
- `import ee` bị Windows chặn DLL (Application Control): các script vẫn chạy được — `eval_utils.py` tự đọc `FEATURE_COLS`/`ERA5_VARS` từ mã nguồn 03 và in cảnh báo. 01–04 và 10 vẫn cần `ee`.
- Early stopping dừng ở rất ít cây (vài chục) ở variant `anomaly:aug`/`delta:base`: bình thường khi phần dư gần như là nhiễu thời tiết toàn vùng; xem cột `n_trees` trong `metrics_improved.csv`. Nếu mô hình đó vẫn thắng baseline ở Val thì hợp lệ; nếu không, kết luận trung thực là "ML chưa hơn baseline" — vẫn là kết quả đáng báo cáo.
