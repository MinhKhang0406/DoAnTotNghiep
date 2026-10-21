"""
BƯỚC 3 (module lõi) — Trích xuất đặc trưng Landsat / Sentinel-2 / ERA5-Land
============================================================================
File này là MODULE DÙNG CHUNG — 04, 07, 08 import trực tiếp từ đây, không
copy lại logic, để tránh 2 nơi lệch nhau.

CẬP NHẬT 2026-09-18 — sửa 2 lỗi phát hiện khi review lại toàn bộ backend:
  1. `_mask_s2_clouds_full`: bản trước ép `ee.Image(image.get("cloud_prob"))`
     NGAY khi gán biến rồi mới đưa vào `ee.Algorithms.If` để test — nếu
     ngày nào S2_SR_HARMONIZED không có ảnh S2_CLOUD_PROBABILITY khớp
     system:index (join không ra kết quả), ép ee.Image(null) lỗi ngay tại
     bước tính condition, trước khi If kịp rẽ nhánh false -> văng exception,
     mất nguyên ngày đó. Sửa: test property gốc trước, ép kiểu Image chỉ ở
     nhánh true.
  2. `mask_landsat_c2l2`: z-score cục bộ (reduceNeighborhood) bản trước
     tính trên lst_c CHƯA mask mây/bóng mây/bão hòa/bất định -> mean/std
     hàng xóm dùng để tính z-score bị lẫn pixel mây trong cửa sổ 5x5. Sửa:
     mask các lỗi thô đó trước, mới tính z-score trên ảnh đã sạch mây.

CẬP NHẬT 2026-09-09 — đồng bộ lại với bản đề cương mới nhất (ChinhSua):

  1. BÀI TOÁN DỰ BÁO: đổi từ "t+7 cố định + cửa sổ dung sai t+5..t+9" sang
     "dự báo linh hoạt cho kỳ quan sát khả dụng tiếp theo" (next available
     observation). Biến delta_t (số ngày từ t đến kỳ ảnh tiếp theo) giờ là
     1 FEATURE trong FEATURE_COLS (không còn bị loại như bản trước) — đúng
     mục 3 "Đối tượng và phạm vi nghiên cứu" của đề cương: "Biến thời gian
     (Delta t)... được đưa vào mô hình như một đặc trưng (feature) độc lập
     giúp mô hình nhận thức được khoảng thời gian dự báo xa hay gần."

  2. BỎ ERA5-Land lag t-1/t-3/t-7 — đề cương bản này CHỈ nói lấy ERA5-Land
     "tại thời điểm hiện tại t" (mục 3, vai trò dữ liệu ERA5-Land), không hề
     nhắc tới độ trễ nào cả. Giữ lag sẽ khiến feature set (và cả % feature
     importance trong SHAP) không khớp với những gì đề cương mô tả.

  3. Sửa cửa sổ khớp thời gian ERA5: bản trước dùng ±90 phút nhưng lại ghi
     comment nhầm là khớp "±1 giờ" — 1 giờ = 60 phút, không phải 90. Đổi lại
     đúng ±60 phút như đề cương ghi ở mục Phương pháp hợp nhất dữ liệu.

  4. Sentinel-2 (NDVI/NDBI/NDWI): BỔ SUNG `.resample("bilinear")` trước khi
     `.reproject()` — bản trước thiếu bước này nên mặc định dùng nearest-
     neighbor cho biến liên tục, sai với đề cương ("dùng phương pháp
     resample bilinear cho biến liên tục (NDVI, NDBI, biến khí tượng)").
     ERA5 vốn đã gọi `.resample("bilinear")` đúng rồi, giữ nguyên.

  5. BỔ SUNG lọc ngoại lệ thống kê (z-score cục bộ không gian) cho Landsat
     LST — đúng đề cương, phần xử lý nhiễu loại (2) "lỗi cảm biến":
     "...lọc ngoại lệ thống kê theo z-score/IQR trong từng cửa sổ không
     gian--thời gian cục bộ". Cài đặt: focal window (reduceNeighborhood)
     bán kính 2 pixel (~60m ở lưới 30m, tương đương cửa sổ 5x5) để tính
     trung bình + độ lệch chuẩn cục bộ, loại pixel lệch quá `zscore_threshold`
     độ lệch chuẩn (mặc định 3.0). ĐÃ GIẢM bán kính từ 5 xuống 2 pixel so
     với thử nghiệm đầu tiên — bán kính 5 (cửa sổ 11x11) làm phép tính
     reduceNeighborhood quá nặng trên ROI lớn (TP.HCM mới gộp cả Bình
     Dương + BR-VT), gây lỗi "Computation timed out" của GEE khi chạy
     đồng thời trên nhiều ảnh (check_filter_stages.py, 04_build_dataset.py).
     Bán kính 2 vẫn đủ bắt các "đốm" giá trị bất hợp lý cục bộ, chỉ nhẹ
     tính toán hơn nhiều. Đây là z-score CỤC BỘ KHÔNG GIAN (trong 1 ảnh/1
     thời điểm) — KHÔNG mở rộng sang trục thời gian (so giữa các ngày khác
     nhau), vì LST biến động mạnh theo ngày do nắng/mưa thực tế, z-score
     theo thời gian dễ loại nhầm biến động thời tiết thật thành "nhiễu".
     Nêu rõ lựa chọn thiết kế này khi viết mục 3.2 báo cáo. Đề cương ghi
     "z-score/IQR" (một trong hai), dùng z-score là đủ.

  6. classify_season()/classify_surface() chuyển từ 08_ablation_shap.py về
     đây làm hàm DÙNG CHUNG — để 07_compute_suhii.py (SUHII) và 08 (ablation)
     dùng chung đúng 1 định nghĩa "bề mặt xây dựng/thực vật/mặt nước", tránh
     2 nơi định nghĩa lệch nhau.

  7. BỔ SUNG get_valid_landsat_dates() — hàm DÙNG CHUNG để lấy danh sách
     ngày Landsat hợp lệ, CHIA NHỎ theo từng quý (mặc định 3 tháng/đợt) rồi
     gọi GEE riêng từng đợt thay vì gọi .getInfo() một lần cho toàn bộ
     khoảng thời gian — tránh lỗi "Computation timed out" khi ROI lớn +
     nhiều ảnh (>1000 ảnh, nhiều năm). check_filter_stages.py,
     check_t7_pairing.py, 04_build_dataset.py đều dùng chung hàm này thay
     vì mỗi nơi tự viết lại 1 kiểu (từng gây lệch logic "valid pixel" giữa
     các script trước đây).

Giữ nguyên từ bản trước (đã đúng, không cần sửa):
  - Sentinel-2 dùng S2_SR_HARMONIZED (Surface Reflectance) + s2cloudless
    kết hợp QA60.
  - Landsat: mask mây/bóng mây/dilated cloud qua QA_PIXEL bitwise +
    QA_RADSAT (lỗi cảm biến bão hòa) + giới hạn vật lý -10..60°C + ST_QA
    (độ bất định nhiệt độ) <= 2K.
"""

import re
import unicodedata

import ee
import numpy as np


def normalize_vn_name(text: str) -> str:
    """Chuẩn hóa tên riêng tiếng Việt (bỏ dấu, khoảng trắng, gạch nối, hạ
    chữ thường) để so khớp KHÔNG phụ thuộc cách nguồn dữ liệu ghi tên — vd
    GADM ghi tỉnh/thành/quận/huyện viết LIỀN không khoảng trắng kiểu
    "HồChíMinh"/"BìnhDương" (đã xác nhận thực tế khi chạy 01), trong khi
    code có thể hard-code so sánh kiểu "Can Gio"/"Cu Chi" (không dấu, có
    khoảng trắng) — nếu so trực tiếp .isin()/== sẽ KHÔNG BAO GIỜ khớp.
    Dùng hàm này ở CẢ 2 phía (dữ liệu thật + danh sách lọc) trước khi so
    sánh. Dùng chung cho 01 (lọc tỉnh/thành) và 04 (lọc quận/huyện spatial
    test)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = re.sub(r"[^a-zA-Z]", "", text)
    return text.lower()

# ---------------------------------------------------------------------
# LANDSAT: lọc mây/bóng mây + lỗi cảm biến (QA_RADSAT + z-score cục bộ) +
#          giới hạn vật lý
# ---------------------------------------------------------------------

def mask_landsat_c2l2(image: "ee.Image", zscore_threshold: float = 3.0, focal_radius_px: int = 2) -> "ee.Image":
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
    st_uncertainty_k = image.select("ST_QA").multiply(0.01)
    uncertainty_mask = st_uncertainty_k.lte(2.0)
    range_mask = lst_c.gte(-10).And(lst_c.lte(60))

    # SỬA LỖI (2026-09-18): mask các lỗi "thô" (mây/bóng mây/bão hòa/bất
    # định/ngoài khoảng vật lý) TRƯỚC khi tính z-score cục bộ. Bản trước
    # reduceNeighborhood chạy trên lst_c CHƯA mask mây -> mean/std hàng xóm
    # dùng để tính z-score bị lẫn cả pixel mây trong cửa sổ 5x5, làm ngưỡng
    # z-score kém chính xác (mây làm lệch mean/std cục bộ, có thể che mất
    # đốm lỗi cảm biến thật hoặc ngược lại loại nhầm pixel hợp lệ cạnh mây).
    # reduceNeighborhood tự bỏ qua pixel đã bị mask khi tính reducer, nên
    # chỉ cần updateMask(pre_mask) trước là đủ, không cần đổi gì khác.
    pre_mask = cloud_mask.And(radsat_mask).And(uncertainty_mask).And(range_mask)
    lst_c_clean = lst_c.updateMask(pre_mask)

    # Nhiễu loại (2) bổ sung — lọc ngoại lệ thống kê z-score CỤC BỘ KHÔNG
    # GIAN: so mỗi pixel với trung bình + độ lệch chuẩn của các pixel lân
    # cận (đã sạch mây) trong bán kính focal_radius_px (cùng 1 ảnh/thời
    # điểm). Bắt được các "đốm" giá trị bất hợp lý do lỗi cảm biến mà
    # QA_RADSAT/ST_QA bỏ sót.
    kernel = ee.Kernel.square(radius=focal_radius_px, units="pixels")
    local_mean = lst_c_clean.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=kernel)
    local_std = lst_c_clean.reduceNeighborhood(reducer=ee.Reducer.stdDev(), kernel=kernel)
    z_score = lst_c_clean.subtract(local_mean).divide(local_std.max(0.01)).abs()
    zscore_mask = z_score.lte(zscore_threshold)

    final_mask = pre_mask.And(zscore_mask)
    return image.addBands(lst_c).updateMask(final_mask)


def build_landsat_collection(aoi_geom: "ee.Geometry", start: str, end: str) -> "ee.ImageCollection":
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi_geom).filterDate(start, end)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(aoi_geom).filterDate(start, end)
    return l8.merge(l9).sort("system:time_start").map(mask_landsat_c2l2)


def _add_months(dt: "datetime", months: int) -> "datetime":
    """Cộng tháng cho 1 datetime, tự xử lý tràn năm/ngày cuối tháng — viết
    tay để khỏi phụ thuộc thêm gói python-dateutil."""
    month_index = dt.month - 1 + months
    year = dt.year + month_index // 12
    month = month_index % 12 + 1
    days_in_month = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(dt.day, days_in_month[month - 1])
    return dt.replace(year=year, month=month, day=day)


def get_valid_landsat_dates(
    roi: "ee.Geometry",
    start: str,
    end: str,
    count_scale: float = 100,
    chunk_months: int = 3,
    verbose: bool = True,
) -> list:
    """Danh sách các ngày (YYYY-MM-DD) có ít nhất 1 pixel Landsat hợp lệ
    (sau mask_landsat_c2l2) trong roi, giữa start và end.

    CHIA NHỎ theo từng đợt `chunk_months` tháng (mặc định 1 quý) rồi gọi
    .getInfo() RIÊNG cho từng đợt — KHÔNG gọi 1 lần cho toàn bộ khoảng thời
    gian, vì với ROI lớn (TP.HCM mới, gộp cả Bình Dương + BR-VT) và nhiều
    năm dữ liệu (>1000 ảnh), gộp 1 lần rất dễ bị GEE trả lỗi "Computation
    timed out" (giới hạn ~5 phút/request đồng bộ của Earth Engine).
    Dùng CHUNG cho check_filter_stages.py, check_t7_pairing.py và
    04_build_dataset.py — để "ngày nào tính là hợp lệ" luôn nhất quán giữa
    lúc kiểm tra và lúc build dataset thật."""
    from datetime import datetime

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
        merged = l8.merge(l9).map(mask_landsat_c2l2)
        merged = merged.map(lambda img: img.set("date_str", img.date().format("YYYY-MM-dd")))

        def _has_valid_pixel(img):
            cnt = img.select("LST_C").reduceRegion(
                reducer=ee.Reducer.count(), geometry=roi, scale=count_scale, maxPixels=1e9, bestEffort=True
            ).get("LST_C")
            return img.set("valid_pixel_count", cnt)

        checked = merged.map(_has_valid_pixel).filter(ee.Filter.gt("valid_pixel_count", 0))

        try:
            chunk_dates = sorted(set(checked.aggregate_array("date_str").getInfo()))
        except Exception as e:
            print(
                f"   ⚠️  Lỗi ở đợt {c_start_str} -> {c_end_str}: {str(e)[:150]}"
                f" -> thử lại đợt này với chunk_months nhỏ hơn (vd 1), hoặc bỏ qua nếu chỉ 1 đợt lẻ bị lỗi."
            )
            chunk_dates = []

        all_valid_dates.extend(chunk_dates)
        if verbose:
            print(f"   [{c_start_str} -> {c_end_str}] {len(chunk_dates)} ngày hợp lệ")

        cursor = chunk_end_dt

    return sorted(set(all_valid_dates))


# ---------------------------------------------------------------------
# SENTINEL-2: Surface Reflectance + QA60 kết hợp s2cloudless, resample
#             bilinear trước khi đưa về lưới 30m (biến liên tục)
# ---------------------------------------------------------------------

S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"


def _join_s2_with_cloud_prob(roi: "ee.Geometry", start: "ee.Date", end: "ee.Date") -> "ee.ImageCollection":
    s2_sr = ee.ImageCollection(S2_SR_COLLECTION).filterBounds(roi).filterDate(start, end)
    s2_prob = ee.ImageCollection(S2_CLOUD_PROB_COLLECTION).filterBounds(roi).filterDate(start, end)
    joined = ee.Join.saveFirst("cloud_prob").apply(
        primary=s2_sr,
        secondary=s2_prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    return ee.ImageCollection(joined)


def _mask_s2_clouds_full(image: "ee.Image", prob_threshold: float = 40.0) -> "ee.Image":
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa60_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    # SỬA LỖI (2026-09-18): test property GỐC (chưa ép kiểu) làm condition
    # trước, chỉ ép ee.Image(...) ở NHÁNH TRUE. Bản trước ép ee.Image(...)
    # NGAY khi gán biến rồi mới đưa vào ee.Algorithms.If — nếu ngày nào đó
    # S2_SR_HARMONIZED không có ảnh S2_CLOUD_PROBABILITY khớp system:index
    # (join saveFirst không ra kết quả, hiếm nhưng có thể xảy ra ở vùng/ngày
    # rìa dữ liệu), property "cloud_prob" là null -> ép ee.Image(null) lỗi
    # ngay tại bước tính condition, TRƯỚC KHI If kịp rẽ sang nhánh false ->
    # nhánh "test rồi mới ép kiểu" mất tác dụng, cả ngày đó bị văng exception.
    cloud_prob_raw = image.get("cloud_prob")
    s2cloudless_mask = ee.Algorithms.If(
        cloud_prob_raw,
        ee.Image(cloud_prob_raw).select("probability").lt(prob_threshold),
        ee.Image.constant(1),
    )
    final_mask = qa60_mask.And(ee.Image(s2cloudless_mask))
    return image.updateMask(final_mask)


def get_sentinel2_features(roi: "ee.Geometry", target_date_str: str, window_days: int = 3) -> "ee.Image":
    """NDVI/NDBI/NDWI từ Sentinel-2 Level-2A (Surface Reflectance), lọc mây
    bằng QA60 + s2cloudless, median trong cửa sổ +-window_days quanh t.
    Resample BILINEAR trước khi đưa về lưới 30m/EPSG:32648 — đúng đề cương
    (biến liên tục dùng bilinear, không phải nearest-neighbor mặc định)."""
    t_date = ee.Date(target_date_str)
    start = t_date.advance(-window_days, "day")
    end = t_date.advance(window_days + 1, "day")

    s2_col = _join_s2_with_cloud_prob(roi, start, end).map(_mask_s2_clouds_full)
    s2_img = s2_col.median()

    ndvi = s2_img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = s2_img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = s2_img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return (
        ee.Image.cat([ndvi, ndbi, ndwi])
        .resample("bilinear")
        .reproject(crs="EPSG:32648", scale=30)
    )


# ---------------------------------------------------------------------
# ERA5-LAND: 5 biến, CHỈ tại thời điểm hiện tại t (không lag), khớp theo
#            giờ chụp thật trong cửa sổ ±60 phút (±1 giờ đúng đề cương)
# ---------------------------------------------------------------------

ERA5_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
ERA5_VARS = ["Temp_2m", "Precip", "Humidity", "Wind", "Solar"]


def _get_era5_image_at(roi: "ee.Geometry", ee_date: "ee.Date", window_minutes: int = 60) -> "ee.Image":
    """Bản ghi ERA5-Land gần nhất với ee_date, trong cửa sổ +-window_minutes
    phút. Mặc định 60 phút = đúng "±1 giờ" ghi trong đề cương (bản trước
    dùng 90 phút do nhầm lẫn)."""
    col = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterBounds(roi)
        .filterDate(ee_date.advance(-window_minutes, "minute"), ee_date.advance(window_minutes, "minute"))
    )

    def add_diff(img):
        diff = ee.Number(img.get("system:time_start")).subtract(ee_date.millis()).abs()
        return img.set("time_diff", diff)

    return col.map(add_diff).sort("time_diff").first()


def get_era5_features(roi: "ee.Geometry", ee_date_t: "ee.Date", window_minutes: int = 60) -> "ee.Image":
    """ERA5-Land CHỈ tại thời điểm hiện tại t (đúng đề cương — không lấy độ
    trễ t-1/t-3/t-7, tuyệt đối không lấy dữ liệu ERA5 ở thời điểm tương lai
    t+delta_t để tránh Data Leakage)."""
    img = _get_era5_image_at(roi, ee_date_t, window_minutes)

    temp_k = img.select("temperature_2m")
    dew_k = img.select("dewpoint_temperature_2m")
    temp_c = temp_k.subtract(273.15)
    dew_c = dew_k.subtract(273.15)

    # Độ ẩm tương đối tính từ nhiệt độ + điểm sương bằng công thức Magnus-Tetens
    humidity = temp_c.expression(
        "100 * exp((17.625*Td)/(243.04+Td)) / exp((17.625*T)/(243.04+T))",
        {"T": temp_c, "Td": dew_c},
    ).rename("Humidity")

    u = img.select("u_component_of_wind_10m")
    v = img.select("v_component_of_wind_10m")
    wind = u.hypot(v).rename("Wind")

    # total_precipitation_hourly: mét/giờ -> đổi sang mm/giờ cho dễ đọc
    precip = img.select("total_precipitation_hourly").multiply(1000).rename("Precip")

    # surface_solar_radiation_downwards_hourly: J/m2 cộng dồn trong giờ -> W/m2 trung bình
    solar = img.select("surface_solar_radiation_downwards_hourly").divide(3600).rename("Solar")

    temp_c_named = temp_c.rename("Temp_2m")

    return (
        ee.Image.cat([temp_c_named, precip, humidity, wind, solar])
        .resample("bilinear")
        .reproject(crs="EPSG:32648", scale=30)
    )


# ---------------------------------------------------------------------
# Phân loại mùa / bề mặt — DÙNG CHUNG cho 07 (SUHII) và 08 (ablation), để
# tránh 2 nơi định nghĩa lệch nhau
# ---------------------------------------------------------------------

def classify_season(date_series) -> "np.ndarray":
    """Mùa mưa TP.HCM: tháng 5-11. Mùa khô: tháng 12-4."""
    import pandas as pd

    month = pd.to_datetime(date_series).dt.month
    return np.where(month.isin([5, 6, 7, 8, 9, 10, 11]), "Mùa mưa", "Mùa khô")


def classify_surface(df) -> "np.ndarray":
    """Phân loại bề mặt bằng ngưỡng đơn giản trên NDWI/NDBI/NDVI."""
    conditions = [df["NDWI"] > 0.1, df["NDBI"] > df["NDVI"]]
    choices = ["Mặt nước", "Xây dựng"]
    return np.select(conditions, choices, default="Thực vật")


# ---------------------------------------------------------------------
# Danh sách feature cột chuẩn — 04/05/06/07/08/09 PHẢI dùng đúng danh sách
# này (import trực tiếp, không copy lại)
# ---------------------------------------------------------------------

FEATURE_COLS = ["LST_t", "NDVI", "NDBI", "NDWI"] + ERA5_VARS + ["delta_t"]
# delta_t (số ngày chờ tới kỳ ảnh khả dụng tiếp theo) NẰM TRONG FEATURE_COLS
# — đúng mục 3 đề cương: "Biến thời gian (Delta t)... được đưa vào mô hình
# như một đặc trưng (feature) độc lập giúp mô hình nhận thức được khoảng
# thời gian dự báo xa hay gần." (KHÁC bản trước — bản trước loại delta_t ra
# do hiểu theo bản đề cương CŨ "bài toán t+7 cố định").


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        print("Chưa xác thực hoặc chưa Initialize được — chạy Authenticate...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print(f"✅ Earth Engine sẵn sàng (project: {project_id})")


def load_boundary_geom(geojson_path: str) -> "ee.Geometry":
    import json

    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ee.FeatureCollection(data).geometry()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    args = parser.parse_args()

    init_ee(args.project)
    roi = load_boundary_geom(args.boundary)

    test_date = "2024-03-01"
    print(f"\n[TESTING] Đang trích xuất đặc trưng cho ngày: {test_date}")

    s2 = get_sentinel2_features(roi, test_date)
    era5 = get_era5_features(roi, ee.Date(test_date))

    print("\n--- Sentinel-2 (SR) Features (đã bilinear resample) ---")
    print("Bands:", s2.bandNames().getInfo())

    print("\n--- ERA5-Land Features (chỉ tại t, cửa sổ ±60 phút) ---")
    era5_bands = era5.bandNames().getInfo()
    print("Bands:", era5_bands, f"(kỳ vọng {len(ERA5_VARS)})")

    print("\n--- Danh sách FEATURE_COLS chuẩn dùng cho model ---")
    print(FEATURE_COLS, f"(tổng {len(FEATURE_COLS)} feature — gồm cả delta_t)")

    print("\n⚠️  QUAN TRỌNG: kiểm tra danh sách band ERA5 ở trên có báo lỗi gì không")
    print("   (band name sai sẽ khiến getInfo() phía trên ném exception ngay tại đây).")
    print("✅ Nếu chạy tới dòng này không lỗi -> tên băng ERA5-Land dùng trong file là đúng.")
