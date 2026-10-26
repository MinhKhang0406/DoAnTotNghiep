# Backend LST WebGIS — bản sửa luồng lấy dữ liệu (23/10/2026)

Giải nén để thư mục `backend/` nằm cạnh thư mục `data/` (các script mặc định đọc/ghi `../data/`, chạy từ BÊN TRONG `backend/`):

```
DATN/
├─ backend/    ← 14 file .py trong gói này
└─ data/       ← tự sinh ra; đặt gadm41_VNM_2.json vào đây
```

---

## 0. Trả lời 3 câu bạn hỏi

**Cần thêm file để hiểu backend không?** Không. Thứ còn thiếu không phải file mà là SỐ LIỆU THẬT từ GEE (mức mất pixel theo từng bước mặt nạ, độ phủ ngày, S2 còn bao nhiêu) để chốt vài ngưỡng. Mình đã viết sẵn script đo (mục 3, bước D); bạn chạy rồi dán output lên chat là chốt được.

**Phần lấy dữ liệu đã tốt chưa?** Chưa. Bản cũ có 3 lỗi ảnh hưởng thẳng tới kết quả: (1) quận đô thị gần như 0 mẫu vì bốc điểm TRƯỚC khi mask mây, (2) "ngày hợp lệ" tính cả cảnh chỉ phủ dải mỏng ở rìa nên sinh delta_t = 1 ảo, (3) Sentinel-2 cửa sổ ±3 ngày làm ảnh chụp SAU ngày nhãn lọt vào feature (rò rỉ tương lai). Danh sách đầy đủ ở mục 5.

**Thiếu luồng dữ liệu nào?** Có 6: bước overlay QA mà đề cương §4 Data Fusion yêu cầu (bản cũ không có), khung điểm cố định xuyên suốt 3 tập, kiểm tra độ "tươi" của nguồn (ERA5 trễ), cột `date_target` (ngày được dự báo), lưới dự báo phủ toàn vùng có trạng thái "không có dữ liệu" (Tuần 12), và đường đẩy metrics/SUHII/ablation/SHAP vào CSDL (Tuần 13). Chi tiết ở mục 6.

---

## 1. Cài đặt

```
pip install -r requirements.txt
earthengine authenticate        (1 lần; script tự gọi nếu chưa đăng nhập)
```

## 2. Luồng dữ liệu mới

```
01 ranh giới ──► 04 frame (khung điểm cố định, offline)
                     │
                     ▼
            04 lst  (GEE: LST tại điểm cho MỌI ngày Landsat, cache + resume)
                     │
                     ▼   check_t7_pairing.py  → chọn --min_valid_frac (offline)
                     │   diagnose_mask_components.py → chọn ST_QA / z-score / cửa sổ S2 (GEE)
                     ▼
            04 features (GEE: S2 [t-3, t] + ERA5 tại t, cache + resume)
                     │
                     ▼
            04 pair  (offline: ghép t → ngày hợp lệ liền kề; Moran chỉ báo cáo)
                     │
                     ▼
         ┌── QA cổng 1 (qa_overlay.py: vị trí / thời điểm / hệ tọa độ / giá trị) ── FAIL thì DỪNG
                     ▼
            04 split (Train ≤2023 | Val 2024 | Test ≥2025 | Test không gian; purge theo nhãn)
                     ▼
         └── QA cổng 2 (mốc năm, rò rỉ nhãn, quận spatial tách hẳn) ── FAIL thì DỪNG
                     ▼
   05 baseline → 06 RF/XGB → 07 SUHII → 08 ablation + SHAP → 10 lưới dự báo → 09 PostGIS
```

## 3. Thứ tự chạy (Windows; chạy từ trong `backend/`, thay `ID` bằng GEE Cloud Project ID)

**A. Chuẩn bị (1 lần)**
```
python 01_setup_gee_boundary.py --source gadm_file --gadm_path ../data/gadm41_VNM_2.json
python 04_build_dataset.py --stage frame
```
01 gộp 3 tỉnh cũ (HCM + Bình Dương + BR-VT), sửa polygon lỗi, tính diện tích, IN DANH SÁCH TÊN ĐƠN VỊ và các cảnh báo (tên trùng giữa tỉnh, polygon chồng lấn). Đọc danh sách tên rồi chọn `--spatial_districts` ở bước sau. `04 --stage frame` tạo `sampling_frame.csv` (100 điểm/đơn vị, cách ranh giới ≥45 m, bỏ điểm mơ hồ ở vùng chồng lấn).

**B. Thử nhỏ trước khi chạy thật**
```
python check_filter_stages.py --project ID --quick
python 04_build_dataset.py --project ID --rebuild_raw --limit_dates 3
```
Lệnh đầu cho biết ERA5/Landsat/S2 có dữ liệu tới ngày nào. Lệnh sau chạy trọn luồng với 3 ngày (gồm cổng QA; chia tập sẽ có Val/Test rỗng là bình thường) — để bắt lỗi quyền/project/API sớm và đo thời gian 1 ngày.

**C. Lấy LST (nặng nhất, tự tiếp tục nếu đứt — chạy lại đúng lệnh cũ)**
```
python 04_build_dataset.py --project ID --stage lst --workers 3
```

**D. Chốt ngưỡng bằng số liệu thật**
```
python check_t7_pairing.py
python diagnose_mask_components.py --project ID --s2
python diagnose_dataset_balance.py
```
- `check_t7_pairing.py`: bảng quét `--min_valid_frac` (ngưỡng độ phủ để 1 ngày được tính là hợp lệ), phân phối delta_t, số cặp theo năm/tập → chọn ngưỡng ở đoạn delta_t ổn định (không nhiều delta_t = 1 ảo, không mất quá nhiều ngày). Ghi `delta_t_distribution.csv` để trích vào đề cương mục 1.1.
- `diagnose_mask_components.py`: mỗi bước mặt nạ (mây, bão hòa, ngoài khoảng vật lý, ST_QA 2/3/5 K, z-score) làm mất bao nhiêu pixel, theo quận; thêm `--s2` cho Sentinel-2 (QA60 / s2cloudless / SCL). Dán các bảng "TOÀN VÙNG" và "giữ sau mây" lên chat nếu cần phân tích tiếp. Thêm `--districts "Quan 1,Quan 3"` để soi quận cụ thể.
- Đổi tham số trích (`--max_st_unc_k`, `--zscore`, `--focal_radius_px`, `--s2_back_days`) thì thêm `--reset_cache` (cache sẽ bị xóa và làm lại; không thêm thì 04 DỪNG để không trộn 2 cấu hình).

**E. Lấy S2 + ERA5, ghép cặp, QA, chia tập**
```
python 04_build_dataset.py --project ID --stage features --min_valid_frac 0.10 --workers 3
python 04_build_dataset.py --stage offline --min_valid_frac 0.10 --spatial_districts "Can Gio,Cu Chi,Dau Tieng"
```
(Thay 0.10 bằng ngưỡng đã chọn ở bước D. Đổi ngưỡng sau này: chạy lại `--stage features` — chỉ tính các ngày còn thiếu — rồi `--stage offline`, không gọi lại GEE cho phần đã có.) Muốn chạy trọn một lệnh: `python 04_build_dataset.py --project ID --rebuild_raw --workers 3`.

Sau lệnh này có `raw_dataset.csv`, `train.csv`, `val.csv`, `test_temporal.csv`, `test_spatial.csv`, `split_summary.csv` (tỷ lệ mẫu thực tế đề cương yêu cầu báo cáo) và `qa_report.csv`.

**F. Mô hình, phân tích, dự báo, CSDL**
```
python 05_baseline_models.py
python 06_train_ml_models.py
python 07_compute_suhii.py
python 08_ablation_shap.py
python 10_build_inference_grid.py --project ID --delta_t 1,7,14
set LST_DB_URL=postgresql://USER:MATKHAU@HOST:5432/postgres
python 09_export_to_postgis.py --dry_run
python 09_export_to_postgis.py
```
PowerShell: `$env:LST_DB_URL="postgresql://USER:MATKHAU@HOST:5432/postgres"`. Supabase thêm `?sslmode=require` nếu cần. 09 không còn mật khẩu ghi cứng trong code (nếu mật khẩu cũ từng lên GitHub thì nên đổi). Cần đã bật PostGIS (`CREATE EXTENSION postgis;`) và user có quyền tạo bảng trong schema `public`. 09 tự chọn driver theo gói đang cài (psycopg2 hoặc psycopg v3 — SQLAlchemy 2.1+ mặc định đòi psycopg v3 nên chuỗi `postgresql://` thuần từng lỗi "No module named 'psycopg'"), nhận cả `postgres://`, và ghi cột ngày dạng DATE.

**Khi cần**
```
python qa_overlay.py --plot                      ảnh chồng lớp 2×2 (qa_overlay_<ngày>.png) để nhìn bằng mắt
python qa_overlay.py --gee --project ID          đối chiếu lại GEE: projection nguồn, trích lại ~60 điểm so với cache, thumbnail
python qa_overlay.py --strict                    coi WARN như FAIL
python check_filter_stages.py --project ID --compare_k 2,3,5     số NGÀY Landsat mất theo ngưỡng ST_QA
```

**Đọc `qa_report.csv`:** FAIL = lỗi thật (sai vị trí/thời điểm/hệ tọa độ/giá trị, rò rỉ nhãn, khác cache...) → 04 dừng; WARN = đáng nhìn lại (vd NDVI–LST không âm, vùng xây dựng không nóng hơn thực vật) nhưng không dừng; INFO = số liệu để trích báo cáo; SKIP = mẫu quá ít. Bỏ cổng QA bằng `--skip_qa` chỉ khi chắc FAIL là báo động giả (và ghi lý do vào báo cáo).

---

## 4. Danh sách file

| File | Trạng thái | Làm gì |
|---|---|---|
| `01_setup_gee_boundary.py` | sửa | Ranh giới GADM 3 tỉnh: dissolve, sửa polygon, diện tích, cảnh báo tên trùng/chồng lấn, xuất bản nhẹ `hcmc_boundary_web.geojson` cho web |
| `03_prepare_features.py` | viết lại | Module lõi dùng chung: mặt nạ Landsat, S2, ERA5, `reduce_at_points`, `ee_retry`, `load_districts`… |
| `04_build_dataset.py` | viết lại | Các stage `frame → lst → features → pair → split`, cache + manifest, đa luồng, cổng QA |
| `qa_overlay.py` | MỚI | Overlay QA (đề cương §4) + ảnh chồng lớp + đối chiếu GEE tùy chọn |
| `diagnose_mask_components.py` | MỚI | Đo mất pixel theo từng bước mặt nạ, theo quận (thay `diagnose_district_pixel_count.py`) |
| `check_t7_pairing.py` | viết lại | Chọn ngưỡng độ phủ, phân phối delta_t, số cặp theo năm/tập (đọc cache, không cần GEE) |
| `check_filter_stages.py` | viết lại | Số cảnh/ngày Landsat, so sánh ngưỡng ST_QA, độ tươi của ERA5/Landsat/S2 |
| `diagnose_dataset_balance.py` | viết lại | Cân bằng dữ liệu theo quận/năm/mùa/tập (không còn sjoin) |
| `05_baseline_models.py` | sửa | Persistence + Linear Regression, lưu `metrics_baseline.csv` |
| `06_train_ml_models.py` | sửa | RF + XGBoost, walk-forward theo ngày có purge, dùng Val, lưu `metrics_ml.csv` + model `.pkl` |
| `07_compute_suhii.py` | sửa | SUHII mô tả, trọng số diện tích, đúng ngày quan sát |
| `08_ablation_shap.py` | sửa | Ablation, theo mùa/bề mặt, SHAP; lưu CSV |
| `10_build_inference_grid.py` | MỚI | Lưới dự báo phủ toàn vùng cho ngày t, nhiều delta_t, cờ `has_data`/`reason` |
| `09_export_to_postgis.py` | sửa | Đẩy 8 bảng, tạo chỉ mục, `--dry_run`, kết nối qua `LST_DB_URL`, tự chọn driver, cột ngày kiểu DATE |

**Bỏ đi (đã bị thay thế):** `02_analyze_landsat_gaps.py`, `diagnose_paint_overlap.py`, `diagnose_district_pixel_count.py` — các lỗi chúng dò đã sửa tận gốc. `diagnose_boundary.py` của bạn giữ nguyên dùng được (01 giờ đã in sẵn các cảnh báo tương tự).

**PHẢI xóa trước khi chạy lại:** `raw_dataset.csv`, `train.csv`, `val.csv`, `test_*.csv` và mọi cache cũ trong `data/` — thuộc schema và cách lấy mẫu cũ (04 sẽ báo lỗi "file từ bản 04 CŨ" nếu quên).

---

## 5. Lỗi bản cũ và cách sửa

| # | Lỗi | Hậu quả | Sửa ở |
|---|---|---|---|
| 1 | `stratifiedSample` bốc điểm TRƯỚC khi mask mây | Quận đô thị (nhiều pixel bị loại) gần như 0 mẫu, quận nông thôn chiếm hết | Khung điểm cố định + `reduce_at_points` lấy giá trị TẠI điểm (04/03) |
| 2 | `.paint()` nền 0 trùng `district_idx` = 0 | Quận idx 0 bị lẫn với "nền" | Bỏ paint; `district_idx` gán sẵn trong khung điểm |
| 3 | `sjoin` nhân đôi điểm nằm vùng polygon chồng lấn (vd Quận 1/Quận 3) | Dòng trùng, thống kê quận lệch | 01 dissolve; khung điểm loại điểm mơ hồ; không còn sjoin ở các bước sau |
| 4 | "Ngày hợp lệ" = chỉ cần 1 pixel ở bất kỳ đâu | Cảnh phủ dải mỏng ở rìa ROI cũng tính → delta_t = 1 ảo | Ngày hợp lệ theo ĐỘ PHỦ DIỆN TÍCH (`--min_valid_frac`) |
| 5 | S2 cửa sổ [t−3, t+3] | Ảnh chụp sau ngày nhãn lọt vào feature (rò rỉ tương lai) | Cửa sổ lùi [t−3, t]; nếu bật `--s2_fwd_days` thì tự chặn ≤ delta_t−1 |
| 6 | S2/ERA5 rỗng làm văng exception | Mất NGUYÊN ngày t | Trả ảnh mask hoàn toàn → chỉ loại mẫu thiếu, không nội suy |
| 7 | `resample("bilinear")` đặt trên ảnh composite | Nhiều khả năng không có tác dụng, S2 vẫn nearest | Gắn vào từng ảnh đầu vào trước khi median |
| 8 | Ngưỡng ST_QA 2 K | Nghi loại thiên lệch pixel đô thị (chưa đo được) | Mặc định 3 K; đo bằng `diagnose_mask_components.py`, đổi bằng `--max_st_unc_k` |
| 9 | Chia tập chỉ xét `date_t` | Nhãn của Train chạm sang năm của Val/Test | Chia theo năm cả `date_t` lẫn `date_target`, purge cặp vắt qua mốc |
| 10 | `TimeSeriesSplit` theo dòng (06) | Cùng 1 ngày nằm ở cả train và valid; nhãn train rơi vào giai đoạn valid | Walk-forward theo NGÀY, loại dòng train có nhãn ≥ ngày đầu fold valid |
| 11 | RF không dùng Validation (06) | Trái đề cương | Top-3 theo walk-forward → chọn bộ RMSE thấp nhất trên Val; Test chỉ dùng 1 lần |
| 12 | SUHII: nhóm theo `date_t` nhưng lấy `LST_target` (07) | SUHII gắn nhãn ngày lệch cả tuần/tháng | Dùng `LST_t` + chỉ số tại t |
| 13 | SUHII không trọng số diện tích; nền có mặt nước; chỉ 2025–2026 (07) | Quận 3 (≈4 km²) nặng bằng Cần Giờ (≈700 km²); nước kéo SUHII lên | Trọng số diện tích; nền = thực vật (vẫn xuất `suhii_incl_water`); dùng toàn bộ 2019–2026 |
| 14 | Mùa ở ablation tính theo `date_t` (08) | Sai ý "hiệu năng dự báo mùa mưa vs khô" | Tính theo `date_target` |
| 15 | Không có `date_target` | Frontend không biết dự báo cho ngày nào | Cột `date_target` xuyên suốt 04 → 09 |
| 16 | Cache lẫn cấu hình (từng gặp `raw_dataset.csv` lẫn dữ liệu trước bản sửa) | Dữ liệu 2 cấu hình trộn im lặng | `cache_manifest.json`: tham số đổi mà không `--reset_cache` thì DỪNG |
| 17 | Ngày Landsat sau ngày ERA5 mới nhất bị loại âm thầm | Không biết vì sao mất ngày gần đây | Báo rõ, bỏ qua tạm, chạy lại stage `features` khi ERA5 cập nhật |
| 18 | Mật khẩu DB ghi cứng trong 09 | Lộ thông tin | Biến môi trường `LST_DB_URL` / `--db_url` |
| 19 | Chuẩn hóa tên xóa luôn chữ số (03, 01) | "Quận 1", "Quận 3", "Quận 12" cùng thành "quan" → chọn `--spatial_districts "Quan 12"` sẽ kéo theo mọi quận đánh số | Giữ chữ số; vẫn khớp tên không dấu/không khoảng trắng |
| 20 | Bảng `lst_predictions` chỉ có `ADM2_NAME` (09) | Tên quận trùng giữa các tỉnh cũ (vd Tân Thành) nối nhầm với `admin_boundaries` | Thêm `ADM1_NAME` làm khóa nối |

---

## 6. Luồng dữ liệu bổ sung

1. **Overlay QA** (`qa_overlay.py`, tự chạy 2 cổng trong 04): ngoài kiểm tra hình thức, có kiểm tra bằng chứng thống kê (NDVI–LST âm, LST_t–LST_target hơn hẳn ghép ngẫu nhiên theo kiểm định hoán vị, ERA5 đồng pha LST sau khi bỏ chu kỳ mùa, vùng xây dựng nóng hơn thực vật) để cảnh báo sớm khi ghép lệch mà vẫn "hợp lệ" về hình thức.
2. **Khung điểm cố định** (`sampling_frame.csv`): cùng một điểm được đọc ở mọi ngày → các vị trí xuất hiện xuyên suốt cả 3 tập (đúng kịch bản Temporal của đề cương).
3. **Độ tươi nguồn** (`check_filter_stages.py`): ngày mới nhất của ERA5-Land, Landsat 8/9, S2 SR, S2 cloud-probability.
4. **Lưới dự báo toàn vùng** (`10`): mỗi ô có `has_data` và `reason` (`no_lst_t`, `no_s2`, `no_era5`, `ok`) để frontend vẽ "không có dữ liệu"; dự báo nhiều delta_t; cảnh báo khi delta_t nằm ngoài khoảng đã học.
5. **Đường vào CSDL cho Tuần 13**: 09 đẩy `lst_predictions`, `lst_forecast_grid`, `admin_boundaries`, `suhii_by_date`, `suhii_monthly`, `model_metrics`, `ablation_results`, `shap_importance` (GiST cho hình học SRID 4326, B-tree cho cột ngày kiểu DATE). `lst_predictions` và `lst_forecast_grid` nối với `admin_boundaries` qua (`ADM1_NAME`, `ADM2_NAME`).
6. **Cache + resume + đa luồng** cho phần GEE (`--workers`, `--limit_dates`), `ee_retry` có backoff khi GEE báo quá tải/timeout.

---

## 7. Đã kiểm thử / CHƯA kiểm thử

**Đã chạy thật (trên dữ liệu giả có cấu trúc: ~700 điểm, 12 đơn vị, ~200 ngày):**
- Biên dịch cả 14 file.
- Toàn bộ phần pandas/geopandas: khung điểm, độ phủ, ghép cặp, Moran, chia tập, QA hai cổng (21 PASS + 4 PASS, 0 FAIL).
- **Tiêm lỗi vào dữ liệu: 19/19 lỗi bị QA bắt đúng kiểm tra** (dòng trùng, đảo lon/lat, nhãn xáo trộn, ERA5 lệch giờ, S2 nhìn tương lai, sai quận, quận spatial lọt vào Train, nhãn Train chạm Val, v.v.) và dữ liệu sạch thì không báo động giả.
- Luồng GEE với `ee` GIẢ (chỉ kiểm tra luồng Python): cache, resume sau lỗi từng ngày, manifest chặn đổi tham số, đa luồng, cache S2 cũ khi đổi ngưỡng, `diagnose_mask_components`, `check_filter_stages`, `10_build_inference_grid`.
- 01 trên file GADM giả (3 tỉnh, polygon lỗi, tên trùng, chồng lấn).
- Chạy sạch từ thư mục giải nén (đúng bố cục `backend/` + `data/`): 04 offline → 05 → 06 → 07 → 08 → 09.
- **09 ghi thật vào PostgreSQL 16 + PostGIS 3.4 cục bộ**: 8 bảng, hình học SRID 4326, cột ngày kiểu DATE, chỉ mục GiST + B-tree không trùng, ghi lại lần 2 (replace), `--mode append`, chuỗi `postgres://`, sai mật khẩu báo lỗi gọn, cả driver psycopg2 và psycopg v3. (Đây là DB cục bộ chứ không phải Supabase của bạn.)

**CHƯA chạy được (cần máy bạn):**
- Mọi lệnh gọi GEE THẬT. Mình đã đọc kỹ cú pháp (`reduceRegions`, `Join.saveFirst`, `reduceNeighborhood`, `resample`…) nhưng chưa có môi trường chạy với tài khoản GEE. Lần đầu có thể gặp lỗi kiểu quota/timeout/tên band — hãy chạy bước B (3 ngày) trước.
- Ghi vào Supabase thật của bạn (chỉ khác ở chuỗi kết nối/SSL/quyền; có thể cần `?sslmode=require`).
- Thời gian chạy thật: chưa đo được; dùng bước B để đo thời gian 1 ngày rồi nhân lên.
- Số liệu mô hình trong các lần chạy thử là trên dữ liệu GIẢ nên không có ý nghĩa khoa học — chỉ để kiểm tra luồng.

---

## 8. Giả định chưa xác minh (kiểm tra khi chạy thật)

1. **QA60 của S2** không còn được ESA điền cho ảnh mới từ đầu 2022 (theo hiểu biết của mình, chưa tự kiểm chứng). Khi đó mặt nạ mây S2 thực tế chỉ còn s2cloudless + SCL. `diagnose_mask_components.py --s2` tự cảnh báo nếu QA60 không loại pixel nào ở ngày đó.
2. **Ngưỡng ST_QA ≤ 3 K, z-score 3, bán kính 2 pixel** là mặc định hợp lý chứ chưa tối ưu — chốt bằng số liệu bước D.
3. **ERA5-Land trên GEE trễ vài tháng** so với Landsat; `check_filter_stages.py` in ngày mới nhất.
4. **`ee.Algorithms.If` với điều kiện là ảnh `cloud_prob`** (null → nhánh dự phòng không có s2cloudless): dựa trên hành vi GEE mình biết, chưa chạy thật.
5. **Cửa sổ S2 [t−3, t] thay vì ±3** (xem mục 9).

---

## 9. Quyết định của bạn (đề cương ↔ code)

1. **Cửa sổ S2**: đề cương ghi ±3 ngày quanh t, code dùng [t−3, t]. Khuyến nghị sửa đề cương thành "3 ngày trước t" — vừa tránh rò rỉ, vừa khớp lúc triển khai thật (khi dự báo ở thời điểm t thì ảnh S2 sau t chưa tồn tại; `10` luôn dùng cửa sổ lùi). `--s2_fwd_days 3` vẫn có để giữ đúng chữ đề cương (tự chặn ≤ delta_t−1) nhưng KHÔNG nên dùng.
2. **Persistence** (đề cương §4/Tuần 6) còn ghi "LST(t+7) = LST(t)" → sửa thành "LST(t+Δt) = LST(t)".
3. **Số liệu delta_t ở mục 1.1** ("1–71 ngày, TB ~10.6"): thay bằng số thật từ `check_t7_pairing.py` sau khi chạy lại.
4. **`--min_valid_frac`**: mặc định 0.10, chốt theo bảng quét ở bước D.
5. **Quận Test không gian**: Cần Giờ, Củ Chi, Dầu Tiếng đều nông thôn/ven đô → Test không gian KHÔNG đo được khả năng khái quát sang đô thị lõi. Cân nhắc thêm 1 quận ven đô thị (vd `Go Vap`) qua `--spatial_districts`.
6. **Lấy mẫu đều theo quận** (100 điểm/quận) làm quận nhỏ đô thị được đại diện nhiều hơn diện tích thật — có chủ đích nhưng cần nêu trong báo cáo; SUHII đã trọng số diện tích. `diagnose_dataset_balance.py` in cột `dai_dien_x` để trích số.
7. **Local Moran's I** trên điểm lấy mẫu thưa (cách nhau hàng km) chủ yếu bắt tương phản thật (công viên giữa khu xây dựng, mặt nước), không nhất thiết là nhiễu → 04 chỉ BÁO CÁO, mặc định không loại (`--drop_moran_outliers` nếu muốn).
8. **Nếu S2 sẵn có quá ít** (nhất là mùa mưa — xem `diagnose_mask_components.py --s2`): phương án dự phòng là tính NDVI/NDBI/NDWI từ chính Landsat OLI SR cùng ngày. Code chưa có phương án này và nó đổi đề cương (bỏ S2), nên chỉ làm nếu số liệu thật cho thấy cần.
9. **Ghi vào "Giới hạn của đề tài"**: tên đơn vị phân vùng theo địa giới CŨ (GADM trước sáp nhập 1/7/2025); SUHII chỉ dựa trên điểm lấy mẫu chứ không phải toàn raster; phân loại bề mặt bằng ngưỡng đơn giản NDVI/NDBI/NDWI (đất trống có thể bị xếp "Xây dựng"); SUHII là phân tích MÔ TẢ, không dùng kết luận "rủi ro nhiệt".
