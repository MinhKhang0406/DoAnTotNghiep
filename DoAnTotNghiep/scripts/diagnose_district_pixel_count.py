"""
Chẩn đoán: đếm số pixel LST HỢP LỆ (đã mask đầy đủ bằng mask_landsat_c2l2 —
CHÍNH XÁC như 03/04 đang dùng) TRỰC TIẾP trong ranh giới TỪNG quận, KHÔNG
qua stratifiedSample(). diagnose_paint_overlap.py đã loại được giả thuyết
.paint() đè mất quận (%_bị_đè_mất cao nhất chỉ 4.7%) — file này tách bạch
2 nghi vấn còn lại:

  (a) Mask (đặc biệt z-score cục bộ trong mask_landsat_c2l2) đang loại gần
      hết pixel ở khu đô thị (bề mặt không đồng nhất: nhà cao tầng/đường
      nhựa/công viên xen kẽ dễ tạo chênh lệch nhiệt độ cục bộ lớn, dễ bị
      hiểu nhầm là lỗi cảm biến) -> nếu đúng, n_valid_pixels ở đây CŨNG gần 0
      ở đúng các quận đang ra 0 mẫu trong raw_dataset.csv.
  (b) stratifiedSample() không lấy được mẫu ở các quận nhỏ/nằm rải rác dù
      pixel hợp lệ vẫn còn nhiều (giới hạn/cách quét theo tile nội bộ của
      GEE với lớp phân loại diện tích quá chênh lệch) -> nếu đúng,
      n_valid_pixels ở đây sẽ KHÁC XA 0 dù stratifiedSample ra 0 mẫu.

Mặc định kiểm tra ngày 2026-09-19 (ngày "t" của cặp mẫu MỚI NHẤT ghép được
"Xong" trong log — xác nhận hôm đó có đủ ảnh/không bị mây che toàn vùng).

Cách chạy (cần GEE, chỉ 1 ngày nên nhanh — khoảng 1-2 phút):
    python diagnose_district_pixel_count.py --project dacn-475523
    python diagnose_district_pixel_count.py --project dacn-475523 --date 2019-02-13
"""
import argparse
import importlib.util
import os
from datetime import datetime, timedelta

import ee
import geopandas as gpd
from shapely.geometry import mapping

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prepare_features", os.path.join(_THIS_DIR, "03_prepare_features.py")
)
prepare_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_features)

# Các quận luôn ra 0 mẫu trong raw_dataset.csv (theo diagnose_dataset_balance.py)
# + 2 quận "may mắn" (CầnGiờ, Quận9) để đối chiếu — nếu 2 quận này cũng ra
# n_valid_pixels cao như thực tế thì phép đo ở đây đáng tin cậy.
TARGET_DISTRICTS = [
    "Quận1", "Quận3", "Quận4", "Quận5", "Quận6", "Quận7", "Quận8", "Quận10", "Quận11",
    "TânBình", "TânPhú", "BìnhThạnh", "GòVấp", "PhúNhuận", "BìnhTân",
    "NhàBè", "HócMôn", "BìnhChánh", "BếnCát", "BắcTânUyên", "ThủDầuMột",
    "ThuậnAn", "TânUyên", "DầuTiếng", "BàRịa",
    "CầnGiờ", "Quận9",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--date", default="2026-09-19", help="Ngày Landsat cần kiểm tra (YYYY-MM-DD)")
    args = parser.parse_args()

    prepare_features.init_ee(args.project)

    gdf = gpd.read_file(args.boundary)  # GeoJSON gốc đã là EPSG:4326, không cần to_crs

    date_dt = datetime.strptime(args.date, "%Y-%m-%d")
    next_day = (date_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(args.date, next_day)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(args.date, next_day)
    merged = l8.merge(l9).map(prepare_features.mask_landsat_c2l2)
    n_imgs = merged.size().getInfo()
    print(f"Số ảnh Landsat 8/9 phủ ngày {args.date}: {n_imgs}")
    if n_imgs == 0:
        print(f"❌ Không có ảnh nào ngày {args.date} — đổi --date sang 1 ngày khác trong "
              f"danh sách 'ngày hợp lệ' của check_t7_pairing.py.")
        return
    lst_t = merged.mosaic().select("LST_C").rename("LST_t")

    print(f"\nĐang đếm pixel hợp lệ (scale=30m, đã mask đầy đủ) trong từng quận...\n")
    rows = []
    for name in TARGET_DISTRICTS:
        match = gdf[gdf["ADM2_NAME"] == name]
        if match.empty:
            print(f"⚠️  Không thấy '{name}' trong {args.boundary} — bỏ qua.")
            continue
        geom = ee.Geometry(mapping(match.geometry.iloc[0]))
        count = lst_t.reduceRegion(
            reducer=ee.Reducer.count(), geometry=geom, scale=30, maxPixels=1e9, bestEffort=True
        ).get("LST_t").getInfo()
        rows.append((name, count or 0))

    rows_sorted = sorted(rows, key=lambda r: -r[1])
    print(f"{'ADM2_NAME':<15}{'n_valid_pixels_30m':>20}")
    for name, count in rows_sorted:
        flag = "  <= đối chiếu (đang CÓ mẫu trong raw_dataset.csv)" if name in ("CầnGiờ", "Quận9") else ""
        print(f"{name:<15}{count:>20}{flag}")

    print(
        "\n=> Nếu các quận KHÔNG phải CầnGiờ/Quận9 ở trên CŨNG có n_valid_pixels_30m gần 0:\n"
        "   xác nhận MASK (nghi nhất là z-score cục bộ) đang loại gần hết pixel đô thị —\n"
        "   cần nới lỏng/sửa lại z-score trong mask_landsat_c2l2.\n"
        "   Nếu các quận đó CÓ nhiều pixel hợp lệ (hàng trăm/nghìn) nhưng stratifiedSample()\n"
        "   ở 04_build_dataset.py vẫn ra 0 mẫu cho đúng các quận này: xác nhận lỗi nằm ở\n"
        "   stratifiedSample(), không phải ở mask -> cần đổi cách lấy mẫu (vd sample() riêng\n"
        "   từng quận thay vì stratifiedSample 1 lần cho cả 40 quận).\n"
        "   Dán TOÀN BỘ output này lên chat."
    )


if __name__ == "__main__":
    main()
