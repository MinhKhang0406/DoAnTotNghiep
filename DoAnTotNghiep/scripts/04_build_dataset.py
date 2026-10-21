"""
BƯỚC 4 — Ghép dataset (dự báo linh hoạt cho kỳ quan sát khả dụng tiếp theo)
             + chia Train/Val/Test/Spatial
============================================================================
CẬP NHẬT 2026-09-09 — đồng bộ lại với bản đề cương mới nhất (ChinhSua):

  1. BỎ CƠ CHẾ CỬA SỔ DUNG SAI [t+5, t+9]: bản trước chỉ ghép mẫu nếu ảnh
     tiếp theo rơi đúng trong cửa sổ này, ngày nào không có ảnh trong cửa
     sổ bị BỎ QUA hoàn toàn — cách này khớp với bản đề cương CŨ ("mốc t+7
     danh nghĩa, có dung sai"), nhưng bản đề cương MỚI NHẤT đã đổi hẳn định
     nghĩa bài toán: "Đề tài xác định bài toán dự báo linh hoạt: Dự báo giá
     trị LST cho kỳ quan sát khả dụng tiếp theo (next available
     observation)" — nghĩa là GHÉP MỖI ngày t hợp lệ với NGÀY HỢP LỆ LIỀN
     KỀ TIẾP THEO trong toàn bộ chuỗi, bất kể cách nhau bao xa (1 ngày hay
     71 ngày), và dùng delta_t LÀM FEATURE cho model học, KHÔNG dùng
     delta_t làm điều kiện lọc mẫu. Chỉ bỏ qua ĐÚNG 1 trường hợp: ngày t là
     ngày CUỐI CÙNG trong toàn bộ chuỗi hợp lệ (không có "ngày tiếp theo").

  2. delta_t giờ NẰM TRONG FEATURE_COLS (03_prepare_features.py) — không
     còn là cột "chỉ giữ lại để báo cáo" như bản trước. RAW_HEADER_COLS và
     cols_to_keep đã bỏ chỗ liệt kê "delta_t" riêng để tránh trùng cột.

  3. BỔ SUNG bước kiểm tra tự tương quan không gian cục bộ (Local Moran's
     I / LISA) sau khi ghép xong dataset — đúng đề cương, phần xử lý nhiễu
     loại (3): "...đồng thời kiểm tra tự tương quan không gian (Moran's I
     cục bộ) để phát hiện các bước nhảy giá trị bất hợp lý giữa các pixel
     lân cận." Mặc định CHỈ BÁO CÁO (không tự xóa dữ liệu) — bật cờ
     --drop_moran_outliers nếu muốn loại tự động sau khi đã xem số liệu.
     Cần cài: pip install libpysal esda --break-system-packages

  4. SỬA LỖI QUAN TRỌNG (2026-09-12) — stratifiedSample() BỎ SÓT gần hết
     quận/huyện: với vùng rộng (~6772 km², gộp cả Bình Dương/BR-VT) chia
     thành 40 lớp (district_idx) chênh lệch diện tích tới ~180 lần, cơ chế
     tự dò danh sách lớp mặc định của stratifiedSample (1 lượt lấy mẫu sơ
     bộ ngẫu nhiên) BỎ SÓT gần hết các quận nhỏ — đã xác nhận thực tế
     25/40 quận ra đúng 0 mẫu suốt 310 cặp ngày (7 năm dữ liệu), trong khi
     1-2 quận khác lại có tỷ lệ điểm/km² cao bất thường (Quận 9: 23.6
     điểm/km², gấp ~8 lần quận đứng nhì). Sửa bằng cách truyền rõ
     classValues + classPoints để ép lấy đúng n_points_per_district cho
     TỪNG lớp đã biết trước, không để nó tự dò nữa (xem build_raw_dataset).
     Nếu bạn đã có sẵn raw_dataset.csv/train.csv/... từ TRƯỚC bản sửa này,
     PHẢI XÓA HẾT và chạy lại --rebuild_raw — dữ liệu cũ không dùng được.

Cách chạy:
    # Bước A — thu thập dữ liệu thô từ GEE (chậm, có thể chạy lại nếu đứt giữa chừng)
    python 04_build_dataset.py --project YOUR_ID --rebuild_raw

    # Bước B — chỉ chia lại Train/Val/Test từ raw_dataset.csv đã có sẵn
    python 04_build_dataset.py --project YOUR_ID

    # Bước B + tự động loại luôn các điểm bị đánh dấu spatial outlier (Moran's I)
    python 04_build_dataset.py --project YOUR_ID --drop_moran_outliers
"""

import argparse
import importlib.util
import json
import os
from datetime import datetime, timedelta

import ee
import geopandas as gpd
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prepare_features", os.path.join(_THIS_DIR, "03_prepare_features.py")
)
prepare_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_features)

RAW_HEADER_COLS = ["lon", "lat", "date_t"] + prepare_features.FEATURE_COLS + ["LST_target"]
SPATIAL_TEST_DISTRICTS = ["Can Gio", "Cu Chi"]
# TODO: cân nhắc thêm 1 đơn vị thuộc Bình Dương/Bà Rịa-Vũng Tàu cũ (vd
# "Dau Tieng") để test khả năng khái quát hóa sang vùng mới sáp nhập — SAU
# KHI đã chạy 01_setup_gee_boundary.py và xem đúng tên GADM dùng (in ra ở
# bước đó, dòng "Các tỉnh/thành cũ được gộp vào") để tránh gõ sai tên gây
# lọc rỗng dữ liệu spatial test.


def _print_collection_range(col: "ee.ImageCollection", label: str) -> None:
    from datetime import datetime, timezone

    n = col.size().getInfo()
    if n == 0:
        print(f"   [{label}] 0 ảnh — RỖNG! (kiểm tra lại điều kiện lọc ở bước này)")
        return
    ts_list = col.aggregate_array("system:time_start").getInfo()
    dates = [datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for t in ts_list]
    print(f"   [{label}] {n} ảnh | sớm nhất={min(dates)} | mới nhất={max(dates)}")


def load_boundary_with_idx(geojson_path: str) -> dict:
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    for i, feat in enumerate(gj["features"]):
        feat["properties"]["district_idx"] = i
    return gj


def flag_local_spatial_outliers(
    df: pd.DataFrame, value_col: str = "LST_t", k_neighbors: int = 8, sig_level: float = 0.05
) -> pd.Series:
    """Local Moran's I (LISA) tính RIÊNG cho từng ngày (date_t) — 'không
    gian' = k hàng xóm gần nhất TRONG CÙNG 1 ngày, 'thời gian' = tách biệt
    theo từng date_t (không trộn các ngày khác nhau vào cùng 1 phép tính
    tương quan không gian) — đúng ý "cửa sổ không gian-thời gian cục bộ".
    Trả về Series bool: True = điểm rơi vào cụm High-Low hoặc Low-High của
    LISA (giá trị lệch bất thường so với hàng xóm) với p_sim <= sig_level.
    CHỈ ĐÁNH DẤU, không tự xóa — xem lại số liệu rồi quyết định."""
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran_Local
    except ImportError:
        print(
            "   ⚠️  Chưa cài libpysal/esda (pip install libpysal esda --break-system-packages) "
            "-> BỎ QUA bước kiểm tra Moran's I. Cần cài để đúng đề cương."
        )
        return pd.Series(False, index=df.index)

    flags = pd.Series(False, index=df.index)
    for date_t, group in df.groupby("date_t"):
        if len(group) < k_neighbors + 1:
            continue
        coords = group[["lon", "lat"]].to_numpy()
        try:
            w = KNN.from_array(coords, k=k_neighbors)
        except Exception:
            continue
        w.transform = "r"
        y = group[value_col].to_numpy()
        try:
            lisa = Moran_Local(y, w, permutations=199, seed=42)
        except Exception:
            continue
        is_outlier = ((lisa.q == 2) | (lisa.q == 3)) & (lisa.p_sim <= sig_level)
        flags.loc[group.index] = is_outlier
    return flags


def build_raw_dataset(
    project_id: str,
    boundary_path: str,
    raw_csv_path: str,
    start: str,
    end: str,
    n_points_per_district: int,
    chunk_months: int = 3,
) -> None:
    prepare_features.init_ee(project_id)
    gj = load_boundary_with_idx(boundary_path)
    hcmc_fc = ee.FeatureCollection(gj)
    roi = hcmc_fc.geometry()
    n_districts = len(gj["features"])
    print(f"   Tổng số đơn vị hành chính (district_idx 0..{n_districts - 1}): {n_districts}")
    district_img = (
        ee.Image(0)
        .paint(hcmc_fc, "district_idx")
        .rename("district_idx")
        .reproject(crs="EPSG:32648", scale=30)
    )

    print("1. Đang kiểm tra từng bước lọc Landsat...")
    l8_raw = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start, end)
    l9_raw = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(start, end)
    _print_collection_range(l8_raw, "Landsat 8 — sau filterBounds+filterDate")
    _print_collection_range(l9_raw, "Landsat 9 — sau filterBounds+filterDate")

    merged_raw = l8_raw.merge(l9_raw).sort("system:time_start")
    _print_collection_range(merged_raw, "L8+L9 — sau merge (TRƯỚC lọc mây/lỗi cảm biến)")

    # `landsat` (đã mask) dùng LẠI Ở DƯỚI cho từng cặp ngày cụ thể (filterDate
    # theo 1-2 ngày, rẻ) — KHÔNG evaluate toàn bộ collection này ngay bây giờ.
    landsat = merged_raw.map(prepare_features.mask_landsat_c2l2)
    landsat = landsat.map(lambda img: img.set("date_str", img.date().format("YYYY-MM-dd")))

    # CẬP NHẬT 2026-09-11: xác định valid_dates bằng hàm CHIA NHỎ theo quý
    # (get_valid_landsat_dates) thay vì gọi checked.aggregate_array(...).getInfo()
    # MỘT LẦN cho toàn bộ collection — cách cũ dễ bị GEE báo "Computation
    # timed out" với ROI lớn (TP.HCM mới) + >1000 ảnh nhiều năm.
    print(f"\n2. Đang xác định ngày hợp lệ (chia theo từng {chunk_months} tháng để tránh timeout)...")
    valid_dates = prepare_features.get_valid_landsat_dates(roi, start, end, chunk_months=chunk_months)
    print(f"\n   Tổng số NGÀY hợp lệ (đã gộp ảnh theo ngày): {len(valid_dates)}")
    if valid_dates:
        print(f"   -> Ngày sớm nhất: {valid_dates[0]} | Ngày mới nhất: {valid_dates[-1]}")

    if not os.path.exists(raw_csv_path):
        pd.DataFrame(columns=RAW_HEADER_COLS).to_csv(raw_csv_path, index=False)
        processed = set()
    else:
        existing = pd.read_csv(raw_csv_path)
        processed = set(existing["date_t"].unique().tolist()) if not existing.empty else set()

    n_paired, n_skipped_last_day = 0, 0
    delta_t_values = []
    for i, t_str in enumerate(valid_dates):
        if t_str in processed:
            continue
        if i + 1 >= len(valid_dates):
            # t là ngày cuối cùng trong toàn bộ chuỗi -> không có "kỳ ảnh
            # tiếp theo" để ghép nhãn, buộc phải bỏ (không phải do cửa sổ
            # dung sai như bản trước, mà vì đơn giản không còn ảnh nào sau).
            n_skipped_last_day += 1
            continue
        t_next_str = valid_dates[i + 1]

        t_date = datetime.strptime(t_str, "%Y-%m-%d")
        t_next_date = datetime.strptime(t_next_str, "%Y-%m-%d")
        delta_t = (t_next_date - t_date).days
        delta_t_values.append(delta_t)

        print(f"Ghép mẫu: {t_str} -> {t_next_str} (delta_t={delta_t})...", end=" ")
        try:
            img_t_col = landsat.filterDate(t_str, (t_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            img_t_meta = img_t_col.first()
            ee_date_t = ee.Date(img_t_meta.get("system:time_start"))
            lst_t = img_t_col.mosaic().select("LST_C").rename("LST_t")

            img_next_col = landsat.filterDate(t_next_str, (t_next_date + timedelta(days=1)).strftime("%Y-%m-%d"))
            lst_target = img_next_col.mosaic().select("LST_C").rename("LST_target")

            s2_t = prepare_features.get_sentinel2_features(roi, t_str)
            era5 = prepare_features.get_era5_features(roi, ee_date_t)
            dt_img = ee.Image.constant(delta_t).rename("delta_t").toFloat()

            combined = ee.Image.cat([lst_t, s2_t, era5, dt_img, lst_target, district_img])

            # SỬA LỖI QUAN TRỌNG (2026-09-12): stratifiedSample() mặc định TỰ DÒ danh
            # sách lớp (district_idx) bằng 1 lượt lấy mẫu sơ bộ NGẪU NHIÊN trên toàn
            # vùng — với vùng rộng (~6772 km², gộp cả Bình Dương/BR-VT) chia thành 40
            # lớp chênh lệch diện tích tới ~180 lần (Quận 3 ~4km² vs Cần Giờ ~718km²),
            # lượt dò ngẫu nhiên này gần như chắc chắn BỎ SÓT hoàn toàn các quận nhỏ
            # (đã xác nhận thực tế: 25/40 quận ra đúng 0 mẫu suốt 310 cặp ngày, trong
            # khi 1-2 quận khác lại nhận SỐ MẪU BẤT THƯỜNG). Truyền rõ classValues +
            # classPoints ép nó lấy đúng n_points_per_district cho TỪNG lớp đã biết
            # trước (0..n_districts-1), không để nó tự dò nữa.
            #
            # LƯU Ý: numPoints VẪN BẮT BUỘC phải truyền dù đã có classValues/classPoints
            # (đã kiểm tra chữ ký hàm thật — numPoints không có giá trị mặc định trong
            # earthengine-api). Vì classValues liệt kê ĐỦ CẢ 40 lớp nên numPoints thực
            # chất không áp dụng cho lớp nào (mọi lớp đều đã có classPoints riêng) —
            # chỉ cần truyền cho đúng cú pháp, giá trị không ảnh hưởng kết quả.
            samples = combined.stratifiedSample(
                numPoints=n_points_per_district,
                classBand="district_idx",
                region=roi,
                scale=30,
                seed=42,
                geometries=True,
                tileScale=4,
                dropNulls=True,
                classValues=list(range(n_districts)),
                classPoints=[n_points_per_district] * n_districts,
            )
            features = samples.getInfo().get("features", [])

            if features:
                rows = []
                for feat in features:
                    props = feat["properties"]
                    coords = feat["geometry"]["coordinates"]
                    props["lon"] = coords[0]
                    props["lat"] = coords[1]
                    props["date_t"] = t_str
                    rows.append(props)
                df_day = pd.DataFrame(rows)
                missing = [c for c in RAW_HEADER_COLS if c not in df_day.columns]
                if missing:
                    print(f"⚠️ thiếu cột {missing} (có thể do mây che hết vùng đó) -> bỏ dòng lỗi.")
                    df_day = df_day.reindex(columns=RAW_HEADER_COLS)
                df_day = df_day[RAW_HEADER_COLS]
                df_day.to_csv(raw_csv_path, mode="a", header=False, index=False)
                print(f"Xong ({len(df_day)} pixels, {df_day.shape[0]} dòng).")
                n_paired += 1
            else:
                print("Bị mây che hoàn toàn hoặc không có điểm hợp lệ, 0 pixels.")
        except Exception as e:
            print(f"Lỗi: {str(e)[:100]}... -> Bỏ qua ngày này.")

    print(f"\nTổng kết: {n_paired} cặp ngày ghép được mẫu.")
    print(f"   -> {n_skipped_last_day} ngày bị bỏ (là ngày CUỐI CÙNG trong chuỗi, không có kỳ ảnh tiếp theo để ghép nhãn).")
    if delta_t_values:
        s = pd.Series(delta_t_values)
        print(f"\nPhân phối delta_t thực tế đạt được ({len(s)} cặp):")
        print(f"   min={s.min()} | mean={s.mean():.1f} | median={s.median():.0f} | max={s.max()}")
        print("   (Số liệu này nên khớp với con số ~10.6 ngày trung bình / 1-71 ngày đã trích dẫn trong đề cương mục 1.1)")
        print(f"   ⚠️  Đây là delta_t của {len(s)} cặp MỚI ghép ở LƯỢT CHẠY NÀY (cơ chế resume ở trên bỏ qua")
        print(f"      những ngày đã có sẵn trong {raw_csv_path}) — KHÔNG phải phân phối của toàn bộ file.")

    # CẬP NHẬT 2026-09-22: cảnh báo khả năng raw_dataset.csv đang LẪN dữ liệu
    # thu thập TỪ TRƯỚC bản sửa classValues/classPoints (2026-09-12, xem
    # dòng 31-42). Vì cơ chế resume ở trên chỉ tính n_paired của phần MỚI,
    # nếu file đã tồn tại từ trước và không được xóa sạch trước khi chạy lại
    # --rebuild_raw, dữ liệu cũ (thiếu classValues/classPoints -> gần như bỏ
    # sót hết các quận nhỏ) vẫn nằm nguyên trong file mà không lượt chạy nào
    # tự phát hiện ra — phải tự so sánh mới thấy. In cảnh báo ngay tại đây
    # để không cần làm việc đó thủ công nữa.
    try:
        full_df = pd.read_csv(raw_csv_path)
        n_total_rows = len(full_df)
        n_total_days = full_df["date_t"].nunique() if "date_t" in full_df.columns else None
        print(f"\n📊 Toàn bộ {raw_csv_path} hiện có {n_total_rows} dòng"
              + (f", {n_total_days} ngày khác nhau" if n_total_days is not None else "")
              + " (gồm cả dữ liệu từ các lượt chạy trước, nếu có).")
        print("   -> Chạy `python diagnose_dataset_balance.py` để xem phân bố theo quận/huyện của TOÀN BỘ file này.")
        print("   -> Nếu thấy 1-2 quận áp đảo (vd Quận9 >20 điểm/km²) VÀ nhiều quận = 0 mẫu: rất có thể file")
        print(f"      đang lẫn dữ liệu thu thập TỪ TRƯỚC bản sửa 2026-09-12 — XÓA HẾT {raw_csv_path}")
        print("      (và train/val/test_temporal/test_spatial.csv) rồi chạy lại --rebuild_raw từ đầu, không resume.")
    except Exception:
        pass


def split_dataset(
    boundary_path: str,
    raw_csv_path: str,
    out_dir: str,
    spatial_districts=SPATIAL_TEST_DISTRICTS,
    drop_moran_outliers: bool = False,
) -> None:
    print("\n2. Đang phân chia tập dữ liệu (Data Splitting)...")
    df = pd.read_csv(raw_csv_path)
    if len(df) == 0:
        raise SystemExit(
            f"❌ {raw_csv_path} RỖNG (0 dòng) — có lỗi xảy ra ở Bước 1 (thu thập từ GEE), "
            f"xem lại log '--rebuild_raw' phía trên (mục 'Lỗi:' ở mỗi ngày) trước khi chia "
            f"dữ liệu. KHÔNG chạy tiếp 05/06/07/08 với dữ liệu rỗng — sẽ chỉ ra lỗi khó hiểu."
        )

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    hcmc = gpd.read_file(boundary_path)
    joined = gpd.sjoin(gdf, hcmc[["ADM2_NAME", "geometry"]], how="inner", predicate="intersects")
    joined["year"] = pd.to_datetime(joined["date_t"]).dt.year

    print("\n2b. Đang kiểm tra tự tương quan không gian cục bộ (Local Moran's I / LISA)...")
    moran_flags = flag_local_spatial_outliers(joined, value_col="LST_t")
    n_flagged = int(moran_flags.sum())
    pct_flagged = n_flagged / len(joined) * 100 if len(joined) else 0
    print(f"   Số điểm bị đánh dấu 'spatial outlier' (High-Low/Low-High LISA, p<=0.05): {n_flagged}/{len(joined)} ({pct_flagged:.2f}%).")
    if drop_moran_outliers:
        joined = joined[~moran_flags]
        print("   -> Đã LOẠI các điểm này khỏi dataset (--drop_moran_outliers được bật).")
    else:
        print("   -> Đang GIỮ LẠI (mặc định chỉ báo cáo) — bật --drop_moran_outliers nếu muốn loại tự động.")

    # So khớp quận/huyện spatial-test bằng tên ĐÃ CHUẨN HÓA (bỏ dấu/khoảng
    # trắng) — GADM có thể ghi ADM2_NAME viết liền có dấu kiểu "CầnGiờ"
    # trong khi SPATIAL_TEST_DISTRICTS ở trên ghi "Can Gio" (ASCII, có
    # khoảng trắng); so .isin() trực tiếp sẽ KHÔNG khớp và làm test_spatial
    # rỗng một cách âm thầm. Chuẩn hóa cả 2 phía cho chắc.
    spatial_keywords_norm = {prepare_features.normalize_vn_name(d) for d in spatial_districts}
    adm2_norm = joined["ADM2_NAME"].apply(prepare_features.normalize_vn_name)
    is_spatial = adm2_norm.isin(spatial_keywords_norm)

    df_spatial = joined[is_spatial]
    df_temporal = joined[~is_spatial]

    if len(df_spatial) == 0:
        unique_adm2 = sorted(joined["ADM2_NAME"].unique().tolist())
        print(
            f"\n⚠️  CẢNH BÁO: test_spatial.csv sẽ RỖNG — không quận/huyện nào trong "
            f"{spatial_districts} khớp (kể cả sau khi chuẩn hóa dấu/khoảng trắng) với "
            f"ADM2_NAME thật trong dữ liệu. Danh sách ADM2_NAME thật để bạn sửa lại "
            f"SPATIAL_TEST_DISTRICTS cho khớp:\n   {unique_adm2}"
        )

    df_train = df_temporal[df_temporal["year"] <= 2023]
    df_val = df_temporal[df_temporal["year"] == 2024]
    df_test_time = df_temporal[df_temporal["year"] >= 2025]

    cols_to_keep = ["lon", "lat", "date_t", "ADM2_NAME"] + prepare_features.FEATURE_COLS + ["LST_target"]

    df_train[cols_to_keep].to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_val[cols_to_keep].to_csv(os.path.join(out_dir, "val.csv"), index=False)
    df_test_time[cols_to_keep].to_csv(os.path.join(out_dir, "test_temporal.csv"), index=False)
    df_spatial[cols_to_keep].to_csv(os.path.join(out_dir, "test_spatial.csv"), index=False)

    print("\n✅ HOÀN TẤT! Đã xuất 4 file CSV:")
    print(f" - Train (2019-2023): {len(df_train)} dòng")
    print(f" - Validation (2024): {len(df_val)} dòng")
    print(f" - Test Thời gian (2025-2026): {len(df_test_time)} dòng")
    print(f" - Test Không gian ({', '.join(spatial_districts)}): {len(df_spatial)} dòng")
    print("\nPhân bố theo quận/huyện trong Train (kiểm tra lại độ cân bằng):")
    print(df_train["ADM2_NAME"].value_counts())
    print("\nPhân bố delta_t thực tế trong toàn bộ dữ liệu (nay là 1 feature, không chỉ để báo cáo):")
    print(joined["delta_t"].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--outdir", default="../data/")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--n_points_per_district", type=int, default=15)
    parser.add_argument(
        "--chunk_months", type=int, default=3,
        help="Chia truy vấn GEE (xác định valid_dates) thành từng đợt N tháng để tránh "
        "'Computation timed out'. Nếu vẫn timeout, giảm xuống 1.",
    )
    parser.add_argument(
        "--rebuild_raw",
        action="store_true",
        help="Chạy lại toàn bộ thu thập dữ liệu từ GEE (chậm). Không truyền cờ này thì chỉ chia lại "
        "Train/Val/Test từ raw_dataset.csv đã có sẵn.",
    )
    parser.add_argument(
        "--drop_moran_outliers",
        action="store_true",
        help="Tự động loại các điểm bị đánh dấu 'spatial outlier' bởi Local Moran's I. "
        "Mặc định KHÔNG bật (chỉ báo cáo số lượng) — xem số liệu rồi quyết định.",
    )
    args = parser.parse_args()

    raw_path = os.path.join(args.outdir, "raw_dataset.csv")

    if args.rebuild_raw:
        build_raw_dataset(
            args.project, args.boundary, raw_path, args.start, args.end,
            args.n_points_per_district, chunk_months=args.chunk_months,
        )
    elif not os.path.exists(raw_path):
        raise SystemExit(f"Không thấy {raw_path} — chạy lại với --rebuild_raw trước.")

    split_dataset(args.boundary, raw_path, args.outdir, drop_moran_outliers=args.drop_moran_outliers)
