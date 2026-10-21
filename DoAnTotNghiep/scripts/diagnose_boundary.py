"""
Chẩn đoán nhanh file ranh giới (KHÔNG cần GEE) — chạy ngay bằng
hcmc_boundary.geojson đã có sẵn, để tìm nguyên nhân 1-2 quận/huyện
chiếm tỷ lệ bất thường trong dataset đã build.

Cách chạy:
    python diagnose_boundary.py --boundary ../data/hcmc_boundary.geojson
"""
import argparse
import geopandas as gpd

parser = argparse.ArgumentParser()
parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
args = parser.parse_args()

gdf = gpd.read_file(args.boundary)
print(f"Tổng số feature (polygon) trong file: {len(gdf)}")
print(f"Số ADM2_NAME DUY NHẤT: {gdf['ADM2_NAME'].nunique()}")

dup = gdf["ADM2_NAME"].value_counts()
dup = dup[dup > 1]
if len(dup) > 0:
    print(f"\n⚠️  {len(dup)} tên ADM2_NAME bị TÁCH thành nhiều feature riêng "
          f"(đây rất có thể là nguyên nhân chính gây lệch):")
    print(dup)
else:
    print("\n✅ Không có ADM2_NAME nào bị trùng lặp trên nhiều feature "
          "-> không phải do lỗi tách feature, cần xem tiếp phần diện tích/chồng lấn bên dưới.")

print("\nDiện tích (km2) theo từng đơn vị, sắp giảm dần (top 15):")
gdf_m = gdf.to_crs("EPSG:32648")
gdf_m["area_km2"] = gdf_m.geometry.area / 1e6
area_by_name = gdf_m.groupby("ADM2_NAME")["area_km2"].sum().sort_values(ascending=False)
print(area_by_name.head(15))

print("\nKiểm tra hình học không hợp lệ (invalid geometry):")
n_invalid = (~gdf_m.is_valid).sum()
print(f"   Số polygon KHÔNG hợp lệ: {n_invalid}/{len(gdf_m)}")

print("\nKiểm tra chồng lấn (overlap) giữa các polygon khác tên nhau:")
sindex = gdf_m.sindex
overlap_found = []
for i, geom in enumerate(gdf_m.geometry):
    for j in sindex.intersection(geom.bounds):
        if j <= i:
            continue
        name_i, name_j = gdf_m["ADM2_NAME"].iloc[i], gdf_m["ADM2_NAME"].iloc[j]
        if name_i == name_j:
            continue
        other = gdf_m.geometry.iloc[j]
        if geom.overlaps(other):
            overlap_area = geom.intersection(other).area / 1e6
            if overlap_area > 0.01:  # bỏ qua sai số làm tròn cực nhỏ
                overlap_found.append((name_i, name_j, round(overlap_area, 2)))

if overlap_found:
    print(f"⚠️  {len(overlap_found)} cặp polygon CHỒNG LẤN đáng kể (>0.01 km2):")
    for a, b, area in sorted(overlap_found, key=lambda x: -x[2])[:20]:
        print(f"   {a} <-> {b}: {area} km2 chồng lấn")
else:
    print("✅ Không phát hiện chồng lấn đáng kể giữa các polygon khác tên.")

print("\n=> Dán TOÀN BỘ output này lên chat để xác định chính xác nguyên nhân.")
