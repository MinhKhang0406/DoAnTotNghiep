"""
BƯỚC 3 (module lõi) — Trích xuất đặc trưng Landsat / Sentinel-2 / ERA5-Land
============================================================================
File này là MODULE DÙNG CHUNG — 04, 07, 08, 10, qa_overlay.py, các script
check_/diagnose_ import trực tiếp từ đây, không copy lại logic.

CẬP NHẬT 2026-10-19 (review toàn bộ luồng lấy dữ liệu) — các thay đổi chính:

  1. S2 CHỐNG RÒ RỈ TƯƠNG LAI: bản cũ lấy cửa sổ [t-3, t+3] quanh t. Với
     delta_t = 1..3 ngày thì ảnh S2 chụp SAU ngày nhãn (t+delta_t) vẫn lọt vào
     feature -> rò rỉ tương lai (đề cương cấm). Giờ cửa sổ mặc định là
     [t - S2_BACK_DAYS, t + S2_FWD_DAYS] = [t-3, t] (S2_FWD_DAYS = 0, TUYỆT ĐỐI
     không nhìn về sau t). Nếu muốn giữ "±3" đúng chữ đề cương thì đặt
     S2_FWD_DAYS=3 NHƯNG khi đó 04 tự chặn thêm: fwd = min(fwd, delta_t - 1)
     để ảnh S2 không bao giờ chụp sau ngày nhãn. (Khuyến nghị: sửa đề cương
     thành "3 ngày trước t" cho khớp code. Lý do thứ hai để giữ fwd = 0: khi
     DỰ BÁO THẬT ở thời điểm t thì ảnh S2 sau t chưa tồn tại — 10_build_inference_grid.py
     luôn dùng fwd = 0, nên huấn luyện với fwd > 0 sẽ lệch với lúc triển khai.)
  2. S2/ERA5 RỖNG KHÔNG CÒN LÀM VĂNG CẢ NGÀY: bản cũ nếu cửa sổ S2 không có ảnh
     nào thì s2.median() ra ảnh 0 băng -> normalizedDifference ném exception ->
     MẤT CẢ NGÀY t. ERA5 chưa phát hành (ERA5-Land trên GEE trễ vài tháng) thì
     .first() = null -> cũng văng. Giờ cả hai trả về ảnh bị mask hoàn toàn (đúng
     tinh thần "không nội suy, pixel thiếu thì loại khỏi mẫu").
  3. S2 thêm mặt nạ bóng mây/lỗi bằng dải SCL (class 3 = cloud shadow,
     1 = saturated/defective) — s2cloudless chỉ bắt mây, không bắt bóng mây.
     QA60 giữ lại theo đề cương (lưu ý: ESA không còn điền QA60 cho sản phẩm
     mới từ đầu 2022 nên từ lúc đó mặt nạ mây S2 thực tế chỉ còn s2cloudless +
     SCL — chưa tự kiểm chứng được trong môi trường của tôi, hãy xem lại trang
     dataset trên GEE catalog).
  4. RESAMPLE BILINEAR: .resample() phải đặt trên ẢNH ĐẦU VÀO (ảnh có projection
     cố định). Bản cũ gọi .resample() trên ảnh composite (median) / ảnh đã tính
     biểu thức — composite không có projection cố định nên lệnh này nhiều khả
     năng không có tác dụng và S2 vẫn rơi về nearest-neighbor. Giờ resample gắn
     vào từng ảnh S2 TRƯỚC khi median, và gắn vào ảnh ERA5 gốc TRƯỚC khi tính
     biểu thức. Không còn .reproject() ép tính toán trên cả ROI — lưới EPSG:32648
     30 m được chỉ định ngay lúc lấy mẫu tại điểm (reduce_at_points).
  5. mask_landsat_c2l2: thêm tham số apply_zscore (False khi chỉ cần biết "ngày
     có ảnh hay không", nhẹ hơn rất nhiều), max_st_unc_k (mặc định 3.0 K — bản
     cũ 2.0 K; ST_QA ở vùng ẩm nhiệt đới và bề mặt đô thị thường lớn hơn nên
     ngưỡng 2 K dễ loại thiên lệch pixel đô thị; đo bằng
     diagnose_mask_components.py rồi chỉnh), xuất thêm dải ST_UNC_K để phân tích.
  6. BỔ SUNG helper dùng chung: ee_retry, reduce_at_points (lấy giá trị tại danh
     sách điểm — thay cho stratifiedSample), list_landsat_scene_times (liệt kê
     ngày chụp chỉ bằng metadata, không tính pixel), get_collection_max_date
     (biết ERA5/Landsat có dữ liệu tới đâu), load_districts (dissolve + district_idx
     ổn định + diện tích), load_boundary_roi (ROI đã đơn giản hóa, payload nhỏ).

Giữ nguyên từ bản trước (đã đúng): Sentinel-2 dùng S2_SR_HARMONIZED +
s2cloudless; Landsat mask QA_PIXEL (bit 1/3/4) + QA_RADSAT + giới hạn vật lý
-10..60°C; z-score cục bộ bán kính 2 pixel trên ảnh đã mask mây; ERA5 chỉ tại t,
cửa sổ ±60 phút; classify_season/classify_surface dùng chung.
"""

import re
import unicodedata
from datetime import datetime, timezone

import ee
import numpy as np
import pandas as pd

CRS_UTM = "EPSG:32648"
GRID_SCALE_M = 30
DEFAULT_MAX_ST_UNC_K = 3.0
S2_BACK_DAYS = 3
S2_FWD_DAYS = 0


def normalize_vn_name(text: str) -> str:
    """Chuẩn hóa tên riêng tiếng Việt (bỏ dấu, khoảng trắng, gạch nối, hạ
    chữ thường) để so khớp KHÔNG phụ thuộc cách nguồn dữ liệu ghi tên — vd
    GADM ghi viết LIỀN không khoảng trắng kiểu "HồChíMinh"/"BìnhDương", trong
    khi code có thể hard-code "Can Gio"/"Cu Chi". Dùng ở CẢ 2 phía trước khi
    so sánh. GIỮ chữ số (bản cũ xóa luôn chữ số nên "Quận 1", "Quận 3", "Quận 12"
    đều thành "quan" và bị coi là MỘT đơn vị khi so khớp tên)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = re.sub(r"[^a-zA-Z0-9]", "", text)
    return text.lower()


# ---------------------------------------------------------------------
# TIỆN ÍCH CHUNG CHO EARTH ENGINE
# ---------------------------------------------------------------------

def ee_retry(fn, tries: int = 4, base_sleep: float = 5.0, label: str = ""):
    """Chạy fn() (thường là lambda: x.getInfo()), thử lại với backoff khi GEE
    báo lỗi tạm thời (429 / 'Too many concurrent' / timeout / 5xx). Lỗi logic
    (band không tồn tại, cú pháp...) vẫn ném ra sau lần thử cuối."""
    import time

    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - GEE ném nhiều loại exception khác nhau
            last = e
            msg = str(e).lower()
            transient = any(
                k in msg
                for k in ("timed out", "timeout", "429", "too many", "quota", "503", "500", "unavailable", "reset")
            )
            if attempt == tries or not transient:
                break
            wait = base_sleep * (2 ** (attempt - 1))
            print(f"   ⚠️  GEE lỗi tạm thời{(' (' + label + ')') if label else ''}: {str(e)[:90]} -> thử lại sau {wait:.0f}s ({attempt}/{tries})")
            time.sleep(wait)
    raise last


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        print("Chưa xác thực hoặc chưa Initialize được — chạy Authenticate...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print(f"✅ Earth Engine sẵn sàng (project: {project_id})")


def _masked_image(band_names: list) -> "ee.Image":
    """Ảnh bị mask HOÀN TOÀN với đúng các băng band_names — dùng làm nhánh
    'không có dữ liệu' của ee.Algorithms.If để lấy mẫu ra null (bị loại khỏi
    dataset) thay vì ném exception làm mất cả ngày."""
    return (
        ee.Image.constant([0] * len(band_names))
        .rename(band_names)
        .toFloat()
        .updateMask(ee.Image.constant(0))
    )


def reduce_at_points(
    image: "ee.Image",
    pts: pd.DataFrame,
    band_names: list,
    id_col: str = "point_id",
    lon_col: str = "lon",
    lat_col: str = "lat",
    scale: float = GRID_SCALE_M,
    crs: str = CRS_UTM,
    chunk_size: int = 2000,
    tile_scale: int = 4,
    label: str = "",
) -> pd.DataFrame:
    """Lấy giá trị của `image` tại danh sách điểm `pts` (lưới EPSG:32648, 30 m).

    Thay cho stratifiedSample(): GEE chỉ tính pixel tại đúng các điểm cần lấy
    (nhẹ hơn rất nhiều so với tính cả ROI ~6772 km² x 30 m) và KHÔNG bao giờ
    'bốc trúng pixel bị mây rồi loại' làm hụt số mẫu theo quận. Pixel bị mask
    -> giá trị NaN (để phía gọi tự dropna). Chia lô chunk_size điểm/lần vì
    getInfo() giới hạn 5000 feature."""
    if len(pts) == 0:
        return pd.DataFrame(columns=[id_col] + list(band_names))

    rows = []
    for start in range(0, len(pts), chunk_size):
        part = pts.iloc[start : start + chunk_size]
        feats = [
            ee.Feature(ee.Geometry.Point([float(lo), float(la)]), {id_col: int(pid)})
            for pid, lo, la in zip(part[id_col], part[lon_col], part[lat_col])
        ]
        fc = ee.FeatureCollection(feats)
        out = image.reduceRegions(
            collection=fc, reducer=ee.Reducer.first(), scale=scale, crs=crs, tileScale=tile_scale
        )
        info = ee_retry(lambda: out.getInfo(), label=label)
        for f in info.get("features", []):
            props = f.get("properties", {})
            rec = {id_col: props.get(id_col)}
            for b in band_names:
                v = props.get(b)
                if v is None and len(band_names) == 1:
                    v = props.get("first")  # ảnh 1 băng: GEE đặt tên thuộc tính theo reducer
                rec[b] = v
            rows.append(rec)

    df = pd.DataFrame(rows, columns=[id_col] + list(band_names))
    for b in band_names:
        df[b] = pd.to_numeric(df[b], errors="coerce")
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype("Int64")
    return df


def get_collection_max_date(collection_id: str, roi: "ee.Geometry" = None):
    """Ngày (YYYY-MM-DD, UTC) của bản ghi MỚI NHẤT trong collection — để biết
    ERA5-Land / Landsat đã có dữ liệu tới đâu. ERA5-Land trên GEE phát hành
    trễ vài tháng nên ngày t gần đây có thể chưa có ERA5 (trước đây âm thầm
    bị loại, giờ phát hiện và báo rõ)."""
    col = ee.ImageCollection(collection_id)
    if roi is not None:
        col = col.filterBounds(roi)
    last = col.limit(1, "system:time_start", False).first()
    ms = ee_retry(lambda: last.get("system:time_start").getInfo(), label=f"max_date {collection_id}")
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------
# RANH GIỚI: đơn vị phân vùng không gian ổn định + ROI nhẹ cho GEE
# ---------------------------------------------------------------------

def load_districts(geojson_path: str):
    """Đọc ranh giới, sửa polygon lỗi, GỘP (dissolve) các feature cùng
    (ADM1_NAME, ADM2_NAME) thành 1 đơn vị, gán district_idx ỔN ĐỊNH (sắp theo
    tên, không phụ thuộc thứ tự trong file) và tính district_area_km2.

    Lý do dissolve: diagnose_boundary.py cho thấy có tên bị TÁCH nhiều
    feature (đảo/mảnh) — nếu mỗi mảnh là 1 'quận' riêng thì thống kê theo quận
    và chia spatial-test đều sai. Trả về GeoDataFrame EPSG:4326 với cột
    ADM1_NAME, ADM2_NAME, district_idx, district_area_km2, geometry."""
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    if "ADM2_NAME" not in gdf.columns:
        raise ValueError("Ranh giới thiếu cột ADM2_NAME — chạy lại 01_setup_gee_boundary.py.")
    if "ADM1_NAME" not in gdf.columns:
        gdf["ADM1_NAME"] = ""
    gdf["ADM1_NAME"] = gdf["ADM1_NAME"].fillna("")

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf.dissolve(by=["ADM1_NAME", "ADM2_NAME"], as_index=False)[["ADM1_NAME", "ADM2_NAME", "geometry"]]
    gdf = gdf.sort_values(["ADM1_NAME", "ADM2_NAME"]).reset_index(drop=True)
    gdf["district_idx"] = range(len(gdf))
    gdf["district_area_km2"] = gdf.to_crs(CRS_UTM).geometry.area / 1e6
    return gdf


def load_boundary_geom(geojson_path: str) -> "ee.Geometry":
    """ROI đầy đủ (polygon chi tiết). NẶNG khi nhúng vào request GEE — chỉ dùng
    cho các script chẩn đoán cũ; luồng chính dùng load_boundary_roi()."""
    import json

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ee.FeatureCollection(data).geometry()


def load_boundary_roi(geojson_path: str, simplify_deg: float = 0.005, buffer_deg: float = 0.01) -> "ee.Geometry":
    """ROI ĐƠN GIẢN HÓA (hợp các quận, simplify ~500 m, buffer ~1 km) chỉ để
    filterBounds trên GEE. Bản cũ nhúng polygon GADM chi tiết (hàng MB) vào MỌI
    request -> chậm và dễ chạm giới hạn kích thước request. Giá trị pixel luôn
    được lấy tại các điểm trong khung lấy mẫu (nằm trong ranh giới thật) nên
    ROI này không ảnh hưởng độ chính xác."""
    from shapely.geometry import mapping

    gdf = load_districts(geojson_path)
    union = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.unary_union
    simple = union.simplify(simplify_deg, preserve_topology=True).buffer(buffer_deg)
    return ee.Geometry(mapping(simple))


# ---------------------------------------------------------------------
# LANDSAT: lọc mây/bóng mây + lỗi cảm biến (QA_RADSAT + z-score cục bộ) +
#          giới hạn vật lý + độ bất định ST_QA
# ---------------------------------------------------------------------

def mask_landsat_c2l2(
    image: "ee.Image",
    zscore_threshold: float = 3.0,
    focal_radius_px: int = 2,
    apply_zscore: bool = True,
    max_st_unc_k: float = DEFAULT_MAX_ST_UNC_K,
) -> "ee.Image":
    """Trả về ảnh có dải LST_C (°C) và ST_UNC_K (K), đã mask.

    apply_zscore=False: bỏ bước z-score cục bộ (reduceNeighborhood x2 — phần nặng
    nhất, nguyên nhân chính gây 'Computation timed out'). Dùng khi chỉ cần biết
    ngày nào có pixel hợp lệ. Khi trích dữ liệu thật luôn để True.
    max_st_unc_k=None: không lọc theo ST_QA."""
    qa = image.select("QA_PIXEL")
    cloud_shadow_bit = 1 << 4
    cloud_bit = 1 << 3
    dilated_cloud_bit = 1 << 1
    cloud_mask = (
        qa.bitwiseAnd(cloud_shadow_bit).eq(0)
        .And(qa.bitwiseAnd(cloud_bit).eq(0))
        .And(qa.bitwiseAnd(dilated_cloud_bit).eq(0))
    )
    radsat_mask = image.select("QA_RADSAT").eq(0)

    lst_c = (
        image.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST_C")
    )
    st_unc_k = image.select("ST_QA").multiply(0.01).rename("ST_UNC_K")
    range_mask = lst_c.gte(-10).And(lst_c.lte(60))

    pre_mask = cloud_mask.And(radsat_mask).And(range_mask)
    if max_st_unc_k is not None:
        pre_mask = pre_mask.And(st_unc_k.lte(max_st_unc_k))

    if apply_zscore:
        # z-score CỤC BỘ KHÔNG GIAN trên ảnh ĐÃ mask các lỗi thô (mây/bão hòa/
        # bất định/ngoài khoảng vật lý) — reduceNeighborhood tự bỏ qua pixel
        # đã bị mask khi tính mean/std hàng xóm.
        lst_c_clean = lst_c.updateMask(pre_mask)
        kernel = ee.Kernel.square(radius=focal_radius_px, units="pixels")
        local_mean = lst_c_clean.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kernel)
        local_std = lst_c_clean.reduceNeighborhood(reducer=ee.Reducer.stdDev(), kernel=kernel)
        z_score = lst_c_clean.subtract(local_mean).divide(local_std.max(0.01)).abs()
        final_mask = pre_mask.And(z_score.lte(zscore_threshold))
    else:
        final_mask = pre_mask

    return image.addBands(lst_c).addBands(st_unc_k).updateMask(final_mask)


def build_landsat_collection(aoi_geom: "ee.Geometry", start: str, end: str, **mask_kwargs) -> "ee.ImageCollection":
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi_geom).filterDate(start, end)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(aoi_geom).filterDate(start, end)
    return l8.merge(l9).sort("system:time_start").map(lambda im: mask_landsat_c2l2(im, **mask_kwargs))


def _add_months(dt: "datetime", months: int) -> "datetime":
    """Cộng tháng cho 1 datetime, tự xử lý tràn năm/ngày cuối tháng."""
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(dt.day, days_in_month[month - 1])
    return dt.replace(year=year, month=month, day=day)


def list_landsat_scene_times(
    roi: "ee.Geometry",
    start: str,
    end: str,
    chunk_months: int = 6,
    max_scene_cloud: float = 100.0,
    verbose: bool = True,
) -> dict:
    """{ 'YYYY-MM-DD': system:time_start (ms, UTC) của cảnh SỚM NHẤT trong ngày }
    cho mọi cảnh Landsat 8/9 chạm ROI giữa start và end.

    CHỈ ĐỌC METADATA (aggregate_array) — không tính pixel nên rất nhanh, không
    timeout. Đây là danh sách NGÀY ỨNG VIÊN; 'ngày hợp lệ' (có đủ pixel sạch)
    được xác định ở 04 bằng số điểm lấy mẫu hợp lệ thực tế.
    max_scene_cloud < 100: bỏ cảnh có CLOUD_COVER (cả cảnh) >= ngưỡng (mặc
    định 100 = tắt, vì tiêu chí 'hợp lệ' của đồ án là theo PIXEL, không theo %
    mây cả cảnh)."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    out = {}
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(_add_months(cursor, chunk_months), end_dt)
        a, b = cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(a, b)
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(a, b)
        col = l8.merge(l9)
        if max_scene_cloud < 100:
            col = col.filter(ee.Filter.lt("CLOUD_COVER", max_scene_cloud))
        times = ee_retry(lambda: col.aggregate_array("system:time_start").getInfo(), label=f"scene_times {a}")
        for ms in times:
            d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            if d not in out or ms < out[d]:
                out[d] = int(ms)
        if verbose:
            print(f"   [{a} -> {b}] {len(times)} cảnh")
        cursor = chunk_end
    return dict(sorted(out.items()))


def get_valid_landsat_dates(
    roi: "ee.Geometry",
    start: str,
    end: str,
    count_scale: float = 100,
    chunk_months: int = 3,
    verbose: bool = True,
    apply_zscore: bool = False,
    max_st_unc_k: float = DEFAULT_MAX_ST_UNC_K,
) -> list:
    """Danh sách các ngày có ít nhất 1 pixel Landsat hợp lệ trong roi (giữ lại
    cho check_filter_stages.py / check_t7_pairing.py). CHIA NHỎ theo
    `chunk_months` tháng để tránh 'Computation timed out'.

    ⚠️ Khác bản cũ: mặc định apply_zscore=False (nhẹ hơn nhiều; việc 'ngày có ảnh
    hay không' gần như không phụ thuộc z-score). Luồng chính (04) KHÔNG dùng
    hàm này nữa mà xác định ngày hợp lệ theo độ phủ thực tế của các điểm lấy
    mẫu — chính xác hơn vì phân biệt được ngày chỉ phủ 1 dải mỏng ở rìa ROI
    (path/row khác) với ngày phủ phần lớn ROI."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    all_valid_dates = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end_dt = min(_add_months(cursor, chunk_months), end_dt)
        c_start_str = cursor.strftime("%Y-%m-%d")
        c_end_str = chunk_end_dt.strftime("%Y-%m-%d")

        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(c_start_str, c_end_str)
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(c_start_str, c_end_str)
        merged = l8.merge(l9).map(
            lambda im: mask_landsat_c2l2(im, apply_zscore=apply_zscore, max_st_unc_k=max_st_unc_k)
        )
        merged = merged.map(lambda img: img.set("date_str", img.date().format("YYYY-MM-dd")))

        def _has_valid_pixel(img):
            cnt = img.select("LST_C").reduceRegion(
                reducer=ee.Reducer.count(), geometry=roi, scale=count_scale, maxPixels=1e9, bestEffort=True
            ).get("LST_C")
            return img.set("valid_pixel_count", cnt)

        checked = merged.map(_has_valid_pixel).filter(ee.Filter.gt("valid_pixel_count", 0))

        try:
            chunk_dates = sorted(set(ee_retry(lambda: checked.aggregate_array("date_str").getInfo())))
        except Exception as e:
            print(
                f"   ⚠️  Lỗi ở đợt {c_start_str} -> {c_end_str}: {str(e)[:150]}"
                f" -> thử lại đợt này với chunk_months nhỏ hơn (vd 1)."
            )
            chunk_dates = []

        all_valid_dates.extend(chunk_dates)
        if verbose:
            print(f"   [{c_start_str} -> {c_end_str}] {len(chunk_dates)} ngày hợp lệ")
        cursor = chunk_end_dt

    return sorted(set(all_valid_dates))


# ---------------------------------------------------------------------
# SENTINEL-2: Surface Reflectance + QA60 + s2cloudless + SCL; cửa sổ CHỈ
#             LÙI về quá khứ (không nhìn sau t); resample bilinear trên ẢNH ĐẦU VÀO
# ---------------------------------------------------------------------

S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
S2_BANDS = ["B3", "B4", "B8", "B11"]


def _join_s2_with_cloud_prob(roi: "ee.Geometry", start: "ee.Date", end: "ee.Date") -> "ee.ImageCollection":
    s2_sr = ee.ImageCollection(S2_SR_COLLECTION).filterBounds(roi).filterDate(start, end)
    s2_prob = ee.ImageCollection(S2_CLOUD_PROB_COLLECTION).filterBounds(roi).filterDate(start, end)
    joined = ee.Join.saveFirst("cloud_prob").apply(
        primary=s2_sr,
        secondary=s2_prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined)


def _mask_s2_clouds_full(image: "ee.Image", prob_threshold: float = 40.0, use_scl: bool = True) -> "ee.Image":
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa60_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    # Test property gốc làm condition, chỉ ép ee.Image ở nhánh TRUE (nếu không
    # có ảnh cloud-prob khớp system:index thì không ném exception).
    cloud_prob_raw = image.get("cloud_prob")
    s2cloudless_mask = ee.Algorithms.If(
        cloud_prob_raw,
        ee.Image(cloud_prob_raw).select("probability").lt(prob_threshold),
        ee.Image.constant(1),
    )
    final_mask = qa60_mask.And(ee.Image(s2cloudless_mask))

    if use_scl:
        scl = image.select("SCL")
        # SCL: 1 = saturated/defective, 3 = cloud shadows (s2cloudless không bắt bóng mây)
        final_mask = final_mask.And(scl.neq(3)).And(scl.neq(1))
    return image.updateMask(final_mask)


def _prep_s2_image(image: "ee.Image") -> "ee.Image":
    return _mask_s2_clouds_full(image).select(S2_BANDS).resample("bilinear")


def get_sentinel2_features(
    roi: "ee.Geometry",
    target_date_str: str,
    back_days: int = S2_BACK_DAYS,
    fwd_days: int = S2_FWD_DAYS,
) -> "ee.Image":
    """NDVI/NDBI/NDWI (+ dải QA s2_n_img = số ảnh S2 trong cửa sổ) từ Sentinel-2
    L2A, median trong [t - back_days, t + fwd_days]. fwd_days mặc định 0 = KHÔNG
    dùng ảnh sau t (chống rò rỉ tương lai). Cửa sổ không có ảnh nào -> ảnh bị
    mask hoàn toàn (mẫu bị loại, KHÔNG nội suy, KHÔNG văng exception).

    Không .reproject(): lưới EPSG:32648/30 m do reduce_at_points chỉ định lúc
    lấy mẫu; bilinear đã gắn vào từng ảnh đầu vào."""
    t_date = ee.Date(target_date_str)
    start = t_date.advance(-back_days, "day")
    end = t_date.advance(fwd_days + 1, "day")

    s2_col = _join_s2_with_cloud_prob(roi, start, end).map(_prep_s2_image)
    n_img = s2_col.size()
    s2_img = ee.Image(ee.Algorithms.If(n_img.gt(0), s2_col.median(), _masked_image(S2_BANDS)))

    ndvi = s2_img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = s2_img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = s2_img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    n_band = ee.Image.constant(n_img).rename("s2_n_img").toFloat()
    return ee.Image.cat([ndvi, ndbi, ndwi, n_band])


# ---------------------------------------------------------------------
# ERA5-LAND: 5 biến, CHỈ tại thời điểm hiện tại t (không lag), khớp theo
#            giờ chụp thật trong cửa sổ ±60 phút; thiếu ERA5 -> ảnh mask
# ---------------------------------------------------------------------

ERA5_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
ERA5_VARS = ["Temp_2m", "Precip", "Humidity", "Wind", "Solar"]
ERA5_RAW_BANDS = [
    "temperature_2m",
    "dewpoint_temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "total_precipitation_hourly",
    "surface_solar_radiation_downwards_hourly",
]


def _get_era5_sorted(roi: "ee.Geometry", ee_date: "ee.Date", window_minutes: int = 60) -> "ee.ImageCollection":
    col = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterBounds(roi)
        .filterDate(ee_date.advance(-window_minutes, "minute"), ee_date.advance(window_minutes, "minute"))
    )

    def add_diff(img):
        diff = ee.Number(img.get("system:time_start")).subtract(ee_date.millis()).abs()
        return img.set("time_diff", diff)

    return col.map(add_diff).sort("time_diff")


def get_era5_features(roi: "ee.Geometry", ee_date_t: "ee.Date", window_minutes: int = 60) -> "ee.Image":
    """ERA5-Land CHỈ tại thời điểm hiện tại t (không lag, tuyệt đối không lấy
    ERA5 ở t+delta_t -> không rò rỉ). Trả về 5 biến ERA5_VARS + dải QA
    era5_dt_min (độ lệch phút giữa bản ghi ERA5 và giờ chụp thật). Không có bản
    ghi ERA5 trong cửa sổ -> ảnh mask hoàn toàn (mẫu bị loại)."""
    col = _get_era5_sorted(roi, ee_date_t, window_minutes)
    img = ee.Image(
        ee.Algorithms.If(
            col.size().gt(0),
            col.first().select(ERA5_RAW_BANDS),
            _masked_image(ERA5_RAW_BANDS).set("time_diff", -1),
        )
    )
    dt_min = ee.Number(img.get("time_diff")).divide(60000)

    # Resample bilinear gắn lên ẢNH GỐC (projection cố định ~9 km) rồi mới tính
    # biểu thức -> khi lấy mẫu ở lưới 30 m, GEE nội suy song tuyến từ lưới ERA5.
    img = img.resample("bilinear")

    temp_c = img.select("temperature_2m").subtract(273.15)
    dew_c = img.select("dewpoint_temperature_2m").subtract(273.15)

    # Độ ẩm tương đối từ nhiệt độ + điểm sương (Magnus-Tetens)
    humidity = temp_c.expression(
        "100 * exp((17.625*Td)/(243.04+Td)) / exp((17.625*T)/(243.04+T))",
        {"T": temp_c, "Td": dew_c},
    ).rename("Humidity")

    wind = img.select("u_component_of_wind_10m").hypot(img.select("v_component_of_wind_10m")).rename("Wind")
    precip = img.select("total_precipitation_hourly").multiply(1000).rename("Precip")  # m/h -> mm/h
    solar = img.select("surface_solar_radiation_downwards_hourly").divide(3600).rename("Solar")  # J/m2 -> W/m2
    dt_band = ee.Image.constant(dt_min).rename("era5_dt_min").toFloat()

    return ee.Image.cat([temp_c.rename("Temp_2m"), precip, humidity, wind, solar, dt_band])


# ---------------------------------------------------------------------
# Phân loại mùa / bề mặt — DÙNG CHUNG cho 07 (SUHII) và 08 (ablation)
# ---------------------------------------------------------------------

def classify_season(date_series) -> "np.ndarray":
    """Mùa mưa TP.HCM: tháng 5-11. Mùa khô: tháng 12-4."""
    month = pd.to_datetime(date_series).dt.month
    return np.where(month.isin([5, 6, 7, 8, 9, 10, 11]), "Mùa mưa", "Mùa khô")


def classify_surface(df) -> "np.ndarray":
    """Phân loại bề mặt bằng ngưỡng đơn giản trên NDWI/NDBI/NDVI."""
    conditions = [df["NDWI"] > 0.1, df["NDBI"] > df["NDVI"]]
    choices = ["Mặt nước", "Xây dựng"]
    return np.select(conditions, choices, default="Thực vật")


# ---------------------------------------------------------------------
# Danh sách feature cột chuẩn — 04/05/06/07/08/09/10 PHẢI dùng đúng danh sách
# này (import trực tiếp, không copy lại)
# ---------------------------------------------------------------------

FEATURE_COLS = ["LST_t", "NDVI", "NDBI", "NDWI"] + ERA5_VARS + ["delta_t"]
# delta_t NẰM TRONG FEATURE_COLS — đúng mục 3 đề cương (feature độc lập cho biết
# khoảng dự báo xa hay gần).

# Cột QA/metadata đi kèm dataset (KHÔNG phải feature của model)
QA_COLS = ["era5_dt_min", "s2_n_img", "st_unc_t", "st_unc_target"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    args = parser.parse_args()

    init_ee(args.project)
    roi = load_boundary_roi(args.boundary)

    test_date = "2024-03-01"
    print(f"\n[TESTING] Đang trích xuất đặc trưng cho ngày: {test_date}")

    s2 = get_sentinel2_features(roi, test_date)
    era5 = get_era5_features(roi, ee.Date(test_date))

    print("\n--- Sentinel-2 (SR) Features ---")
    print("Bands:", ee_retry(lambda: s2.bandNames().getInfo()), "(kỳ vọng NDVI, NDBI, NDWI, s2_n_img)")

    print("\n--- ERA5-Land Features (chỉ tại t, cửa sổ ±60 phút) ---")
    print("Bands:", ee_retry(lambda: era5.bandNames().getInfo()), f"(kỳ vọng {len(ERA5_VARS)} biến + era5_dt_min)")

    print("\n--- Ngày mới nhất của từng nguồn (để biết giới hạn dữ liệu) ---")
    for name, cid in [("ERA5-Land", ERA5_COLLECTION), ("Landsat 8", "LANDSAT/LC08/C02/T1_L2"),
                      ("Landsat 9", "LANDSAT/LC09/C02/T1_L2"), ("Sentinel-2 SR", S2_SR_COLLECTION)]:
        print(f"   {name:<14}: {get_collection_max_date(cid)}")

    print("\n--- Danh sách FEATURE_COLS chuẩn dùng cho model ---")
    print(FEATURE_COLS, f"(tổng {len(FEATURE_COLS)} feature — gồm cả delta_t)")
    print("✅ Nếu chạy tới dòng này không lỗi -> tên băng/collection dùng trong file là đúng.")
