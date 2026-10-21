"""
Chẩn đoán độ cân bằng dữ liệu THẬT — xem TOÀN BỘ raw_dataset.csv phân bố
theo quận/huyện thế nào (KHÔNG tách theo Train/Val/Test/Spatial), so sánh
với diện tích để phân biệt "hiệu ứng diện tích bình thường" với "bug khác
cần điều tra thêm". Không cần GEE, chạy trong vài giây.

Cách chạy:
    python diagnose_dataset_balance.py
"""
import geopandas as gpd
import pandas as pd

raw = pd.read_csv("../data/raw_dataset.csv")
gdf = gpd.GeoDataFrame(raw, geometry=gpd.points_from_xy(raw.lon, raw.lat), crs="EPSG:4326")
boundary = gpd.read_file("../data/hcmc_boundary.geojson")
joined = gpd.sjoin(gdf, boundary[["ADM2_NAME", "geometry"]], how="inner", predicate="intersects")

print(f"Tổng số dòng trong raw_dataset.csv: {len(raw)}")
print(f"Tổng số dòng sau sjoin với ranh giới: {len(joined)}")
if len(joined) != len(raw):
    diff = len(joined) - len(raw)
    print(
        f"⚠️  Chênh lệch {diff} dòng so với raw_dataset.csv gốc — "
        f"{'nhiều hơn do 1 điểm rơi vào vùng giao nhau của 2 polygon bị đúp thành 2 dòng' if diff > 0 else 'ít hơn do có điểm rơi ngoài mọi polygon'}."
    )

counts = joined["ADM2_NAME"].value_counts()
print(f"\nSố dòng theo TỪNG quận/huyện (toàn bộ {len(joined)} dòng, chưa chia Train/Val/Test):")
print(counts.to_string())

boundary_m = boundary.to_crs("EPSG:32648")
boundary_m["area_km2"] = boundary_m.geometry.area / 1e6
area = boundary_m.set_index("ADM2_NAME")["area_km2"]

compare = pd.DataFrame({"so_diem": counts, "dien_tich_km2": area}).fillna(0)
compare["diem_tren_km2"] = (compare["so_diem"] / compare["dien_tich_km2"]).round(3)
compare = compare.sort_values("so_diem", ascending=False)

print("\nSo sánh số điểm / diện tích (điểm trên km2):")
print(compare.to_string())
print(
    "\n=> Nếu cột 'diem_tren_km2' TƯƠNG ĐỐI ĐỀU giữa các quận: bình thường, chỉ là hiệu\n"
    "   ứng diện tích (quận nhỏ dễ mất trắng vì mây). Nếu 1-2 quận có điểm/km2 CAO VỌT\n"
    "   (gấp nhiều lần so với phần còn lại): đó là bug thật, không phải diện tích."
)
