"""
Chẩn đoán khả năng .paint() vẽ ĐÈ mất hẳn 1 quận/huyện khỏi ảnh district_idx.
==============================================================================
04_build_dataset.py dùng:
    district_img = ee.Image(0).paint(hcmc_fc, "district_idx")...
`.paint(featureCollection, property)` vẽ theo THỨ TỰ feature trong
collection — feature đứng SAU (district_idx LỚN HƠN, do gán bằng
enumerate() trên thứ tự file) sẽ ĐÈ LÊN feature đứng trước tại phần diện
tích chồng lấn. diagnose_boundary.py đã tìm thấy 51 cặp polygon chồng lấn
đáng kể (>0.01 km2) nhưng chỉ in ra 20 cặp lớn nhất — file này tính CHO
TỪNG quận: bao nhiêu % diện tích của quận đó bị 1 quận khác có district_idx
LỚN HƠN che mất, dùng ĐÚNG thứ tự district_idx như 04_build_dataset.py
(load_boundary_with_idx) để kết quả phản ánh đúng thực tế .paint() sẽ làm.

Nếu 1 quận có %_bị_đè_mất cao (đặc biệt >50%) VÀ quận đó đang ra 0 mẫu
trong diagnose_dataset_balance.py -> xác nhận .paint() đang xóa mất quận đó.

Cách chạy (không cần GEE):
    python diagnose_paint_overlap.py --boundary ../data/hcmc_boundary.geojson
"""
import argparse
import json

import geopandas as gpd

parser = argparse.ArgumentParser()
parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
args = parser.parse_args()

with open(args.boundary, "r", encoding="utf-8") as f:
    gj = json.load(f)
for i, feat in enumerate(gj["features"]):
    feat["properties"]["district_idx"] = i

gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326").to_crs("EPSG:32648")
gdf["area_km2"] = gdf.geometry.area / 1e6
gdf = gdf.reset_index(drop=True)

sindex = gdf.sindex
rows = []
for i, geom in enumerate(gdf.geometry):
    own_area = geom.area
    if own_area <= 0:
        continue
    idx_i = gdf["district_idx"].iloc[i]
    covered_by_later = None
    for j in sindex.intersection(geom.bounds):
        if j == i:
            continue
        idx_j = gdf["district_idx"].iloc[j]
        if idx_j <= idx_i:
            # Chỉ feature có district_idx LỚN HƠN mới "đè" lên feature này khi paint
            continue
        other = gdf.geometry.iloc[j]
        if not geom.intersects(other):
            continue
        inter = geom.intersection(other)
        if inter.is_empty:
            continue
        covered_by_later = inter if covered_by_later is None else covered_by_later.union(inter)
    pct_lost = (covered_by_later.area / own_area * 100) if covered_by_later is not None else 0.0
    rows.append((gdf["ADM2_NAME"].iloc[i], idx_i, gdf["area_km2"].iloc[i], pct_lost))

rows.sort(key=lambda r: -r[3])
print(f"{'ADM2_NAME':<15}{'district_idx':>13}{'area_km2':>12}{'%_bị_đè_mất':>15}")
for name, idx, area, pct in rows:
    flag = "  <=== NGHI VẤN (bị đè mất phần lớn diện tích)" if pct > 20 else ""
    print(f"{name:<15}{idx:>13}{area:>12.2f}{pct:>14.1f}%{flag}")

print(
    "\n=> Nếu 1 quận có %_bị_đè_mất CAO (đặc biệt >50%) VÀ trùng với quận đang ra 0 mẫu\n"
    "   trong diagnose_dataset_balance.py: xác nhận .paint() đang đè mất quận đó thật —\n"
    "   cần đổi cách xây district_img (vd: rasterize riêng từng quận rồi ghép, hoặc\n"
    "   dissolve/fix polygon chồng lấn trước khi paint) thay vì sửa ở bước sampling.\n"
    "   Nếu %_bị_đè_mất THẤP (gần 0) ở đúng các quận đang ra 0 mẫu: paint không phải\n"
    "   nguyên nhân, cần chuyển sang nghi vấn khác (mask z-score quá chặt trên nền đô\n"
    "   thị không đồng nhất, hoặc stratifiedSample bỏ sót lớp nhỏ nằm rải rác).\n"
    "   Dán TOÀN BỘ output này lên chat."
)
