"""
BƯỚC 1 — Setup Earth Engine + lấy ranh giới hành chính THẬT của TP.HCM
=======================================================================
CẬP NHẬT 2026-10-19 (review luồng lấy dữ liệu) — bổ sung xử lý hình học để các
bước sau không phải đoán/chữa lỗi:

  1. GỘP (dissolve) các feature cùng (ADM1_NAME, ADM2_NAME) thành 1 đơn vị.
     diagnose_boundary.py từng phát hiện có tên bị TÁCH thành nhiều feature
     (đảo/mảnh). Nếu mỗi mảnh là 1 "quận" riêng thì thống kê theo quận và việc
     chọn quận Test không gian đều sai. Tắt bằng --no_dissolve nếu thật sự cần.
  2. SỬA polygon không hợp lệ (make_valid/buffer(0)) trước khi lưu — polygon lỗi
     làm sjoin/overlay/dissolve ở các bước sau ném lỗi hoặc cho kết quả sai.
  3. Thêm cột district_area_km2 (diện tích UTM 48N) — dùng tính trọng số diện
     tích cho SUHII (07) và độ phủ ảnh theo diện tích (04).
  4. CẢNH BÁO: (a) cùng ADM2_NAME xuất hiện ở NHIỀU tỉnh (so khớp tên đơn lẻ sẽ
     nhầm), (b) cặp polygon chồng lấn đáng kể (khung điểm ở 04 tự loại điểm nằm
     vùng chồng lấn, nhưng nên biết), (c) IN DANH SÁCH TÊN ĐƠN VỊ để bạn chọn đúng
     tên cho --spatial_districts ở 04.
  5. XUẤT THÊM bản NHẸ (đơn giản hóa hình học) cho WebGIS/PostGIS:
     hcmc_boundary_web.geojson (mặc định sai số ~30 m). File phân tích
     hcmc_boundary.geojson GIỮ NGUYÊN độ chi tiết. 09_export_to_postgis.py tự ưu
     tiên bản nhẹ khi đẩy vào CSDL.

Giữ nguyên từ bản 2026-09-08 — xử lý việc TP.HCM sáp nhập Bình Dương + Bà Rịa -
Vũng Tàu từ 1/7/2025 (cấp quận/huyện đã bỏ):
  - GAUL (--source gaul) quá thiếu chi tiết cho TP.HCM -> không dùng.
  - GADM chỉ có ranh giới TRƯỚC sáp nhập -> lọc CẢ 3 tỉnh/thành cũ rồi dùng
    ranh giới quận/huyện cũ của cả 3 nơi làm đơn vị phân vùng không gian (tên
    đơn vị con theo địa giới cũ — cần ghi rõ trong "Giới hạn của đề tài").
  - Nếu chỉ cần đường viền ngoài: hcmc_new_outline_2025.geojson (nguồn:
    https://github.com/nguyenduy1133/Free-GIS-Data, đối chiếu NQ 202/2025/QH15).

Cách chạy (đã activate env lst-thesis):
    python 01_setup_gee_boundary.py --source gadm_file --gadm_path duong_dan/gadm41_VNM_2.json
    python 01_setup_gee_boundary.py --source gadm_file --gadm_path duong_dan/gadm41_VNM_2.json \
        --provinces "Ho Chi Minh|Hồ Chí Minh"

Tải GADM level-2 Việt Nam tại: https://gadm.org/download_country.html (Vietnam > GeoJSON)
"""

import argparse
import json
import re
import sys
import os
import unicodedata

import ee
import geopandas as gpd


def _normalize_vn_name(text: str) -> str:
    """Chuẩn hóa tên riêng tiếng Việt: bỏ dấu, bỏ khoảng trắng/gạch nối, hạ chữ
    thường — để so khớp không phụ thuộc cách từng nguồn ghi tên. GIỮ chữ số (giống
    03_prepare_features.normalize_vn_name) để "Quận 1" và "Quận 12" không bị coi là một."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = re.sub(r"[^a-zA-Z0-9]", "", text)
    return text.lower()


# 3 tỉnh/thành CŨ đã gộp thành TP.HCM mới từ 1/7/2025 — viết dạng từ khóa ĐÃ
# CHUẨN HÓA (không dấu, không khoảng trắng).
DEFAULT_PROVINCE_KEYWORDS = ["hochiminh", "binhduong", "baria", "vungtau"]


def init_ee(project_id: str) -> None:
    try:
        ee.Initialize(project=project_id)
    except Exception:
        print("Chưa xác thực hoặc chưa Initialize được — chạy Authenticate...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print(f"✅ Earth Engine sẵn sàng (project: {project_id})")


def inspect_gaul_names() -> list:
    """Giữ lại để tham khảo — KHÔNG khuyến khích dùng cho đồ án này."""
    gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
    vn = gaul2.filter(ee.Filter.eq("ADM0_NAME", "Viet Nam"))
    all_names = vn.aggregate_array("ADM1_NAME").distinct().getInfo()
    matches = [n for n in all_names if "ho chi minh" in n.lower()]
    print("Tất cả ADM1_NAME của Việt Nam trong GAUL:", sorted(all_names))
    print("=> Các tên khớp 'Ho Chi Minh':", matches)
    return matches


def get_hcmc_boundary_from_gaul(adm1_name: str) -> "ee.FeatureCollection":
    gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2")
    hcmc = gaul2.filter(ee.Filter.eq("ADM1_NAME", adm1_name))
    n = hcmc.size().getInfo()
    print(f"Số đơn vị hành chính (quận/huyện) tìm thấy: {n}")
    if n == 0:
        raise ValueError(
            "Lọc ra 0 quận/huyện — adm1_name không khớp dữ liệu gốc. "
            "Xem lại danh sách in ra ở inspect_gaul_names()."
        )
    return hcmc


def _fix_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    n_bad = int((~gdf.geometry.is_valid).sum())
    if n_bad:
        print(f"   🔧 Sửa {n_bad} polygon không hợp lệ (make_valid).")
        try:
            gdf["geometry"] = gdf.geometry.make_valid()
        except Exception:
            gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def _report_quality(gdf: gpd.GeoDataFrame) -> None:
    """Cảnh báo tên trùng giữa các tỉnh + polygon chồng lấn; in danh sách tên đơn vị."""
    dup_names = gdf.groupby("ADM2_NAME")["ADM1_NAME"].nunique()
    dup_names = dup_names[dup_names > 1]
    if len(dup_names):
        print(f"\n⚠️  {len(dup_names)} ADM2_NAME xuất hiện ở NHIỀU tỉnh (so khớp theo tên đơn lẻ sẽ nhầm): "
              f"{dup_names.index.tolist()}")

    gm = gdf.to_crs("EPSG:32648")
    sindex = gm.sindex
    overlaps = []
    for i, geom in enumerate(gm.geometry):
        for j in sindex.intersection(geom.bounds):
            if j <= i:
                continue
            other = gm.geometry.iloc[j]
            if geom.intersects(other):
                a = geom.intersection(other).area / 1e6
                if a > 0.01:
                    overlaps.append((gm["ADM2_NAME"].iloc[i], gm["ADM2_NAME"].iloc[j], round(a, 2)))
    if overlaps:
        print(f"⚠️  {len(overlaps)} cặp đơn vị CHỒNG LẤN > 0.01 km² (top 10): "
              f"{sorted(overlaps, key=lambda x: -x[2])[:10]}")
        print("   -> 04 tự loại điểm lấy mẫu nằm trong vùng chồng lấn; không cần sửa tay.")
    else:
        print("✅ Không có chồng lấn đáng kể giữa các đơn vị.")

    names = gdf.sort_values("district_area_km2")[["ADM1_NAME", "ADM2_NAME", "district_area_km2"]]
    print("\nDanh sách đơn vị (nhỏ -> lớn theo diện tích km²) — dùng để chọn --spatial_districts ở 04:")
    for _, r in names.iterrows():
        print(f"   {r.ADM1_NAME:<16} {r.ADM2_NAME:<22} {r.district_area_km2:>8.1f}")


def get_hcmc_boundary_from_gadm_file(
    gadm_geojson_path: str, province_pattern: str = None, dissolve: bool = True
) -> gpd.GeoDataFrame:
    """Lọc GADM level-2 Việt Nam theo tên tỉnh/thành CŨ (NAME_1), mặc định lấy cả
    3 tỉnh/thành đã sáp nhập vào TP.HCM mới. So khớp bằng tên ĐÃ chuẩn hóa ở cả
    2 phía (bỏ dấu + khoảng trắng + gạch nối)."""
    gdf = gpd.read_file(gadm_geojson_path)
    print("Các cột có sẵn trong file GADM:", list(gdf.columns))
    if "NAME_1" not in gdf.columns:
        raise ValueError("Không thấy cột NAME_1 — kiểm tra lại đúng file GADM level-2 chưa.")
    print("Giá trị NAME_1 duy nhất:", sorted(gdf["NAME_1"].unique().tolist()))

    if province_pattern:
        keywords = [_normalize_vn_name(p) for p in province_pattern.split("|") if p.strip()]
    else:
        keywords = DEFAULT_PROVINCE_KEYWORDS

    name1_norm = gdf["NAME_1"].apply(_normalize_vn_name)
    mask = name1_norm.apply(lambda n: any(kw in n for kw in keywords))
    hcmc_gdf = gdf[mask].copy()
    if len(hcmc_gdf) == 0:
        raise ValueError(
            "Lọc theo tên không ra kết quả (kể cả sau khi đã chuẩn hóa dấu/khoảng "
            "trắng) — xem danh sách NAME_1 in ở trên, chỉnh lại DEFAULT_PROVINCE_KEYWORDS "
            "hoặc --provinces cho khớp cách GADM ghi tên."
        )

    print(f"Số feature (quận/huyện cũ) tìm thấy trước khi gộp: {len(hcmc_gdf)}")
    print("   Các tỉnh/thành cũ được gộp vào:", sorted(hcmc_gdf["NAME_1"].unique().tolist()))

    # Chuẩn hóa tên cột về schema các bước sau mong đợi (ADM1_NAME/ADM2_NAME)
    rename_map = {}
    if "NAME_1" in hcmc_gdf.columns:
        rename_map["NAME_1"] = "ADM1_NAME"
    if "NAME_2" in hcmc_gdf.columns:
        rename_map["NAME_2"] = "ADM2_NAME"
    hcmc_gdf = hcmc_gdf.rename(columns=rename_map)
    if hcmc_gdf.crs is None:
        hcmc_gdf = hcmc_gdf.set_crs("EPSG:4326")
    hcmc_gdf = hcmc_gdf.to_crs("EPSG:4326")

    hcmc_gdf = _fix_geometry(hcmc_gdf)

    if dissolve:
        n_before = len(hcmc_gdf)
        hcmc_gdf = hcmc_gdf.dissolve(by=["ADM1_NAME", "ADM2_NAME"], as_index=False)[["ADM1_NAME", "ADM2_NAME", "geometry"]]
        print(f"   Gộp feature cùng (ADM1_NAME, ADM2_NAME): {n_before} -> {len(hcmc_gdf)} đơn vị.")
    else:
        hcmc_gdf = hcmc_gdf[["ADM1_NAME", "ADM2_NAME", "geometry"]]

    hcmc_gdf = hcmc_gdf.sort_values(["ADM1_NAME", "ADM2_NAME"]).reset_index(drop=True)
    hcmc_gdf["district_area_km2"] = (hcmc_gdf.to_crs("EPSG:32648").geometry.area / 1e6).round(3)

    _report_quality(hcmc_gdf)
    return hcmc_gdf


def export_boundary_geojson_from_ee(hcmc_fc: "ee.FeatureCollection", out_path: str) -> None:
    geojson = hcmc_fc.getInfo()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    _print_done(out_path, len(geojson.get("features", [])))


def _print_done(out_path: str, n_features: int) -> None:
    print(f"\n✅ Đã lưu ranh giới thật ({n_features} đơn vị phân vùng) vào: {out_path}")
    print("   Việc tiếp theo:")
    print("   1. Chạy: python 04_build_dataset.py --stage frame   (tạo khung điểm lấy mẫu từ ranh giới này).")
    print("   2. Chọn 2-3 đơn vị làm Test không gian từ danh sách tên in ở trên (--spatial_districts ở 04);")
    print("      nên có ít nhất 1 đơn vị thuộc Bình Dương/BR-VT cũ để kiểm tra khái quát hóa sang vùng mới sáp nhập.")
    print("   3. Đẩy ranh giới vào CSDL cho WebGIS: python 09_export_to_postgis.py (tự dùng bản _web nhẹ nếu có).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", help="GEE Cloud Project ID (bắt buộc nếu --source gaul)")
    parser.add_argument(
        "--source",
        choices=["gaul", "gadm_file"],
        default="gadm_file",
        help="Nguồn ranh giới: 'gadm_file' (khuyến nghị) hoặc 'gaul' (qua GEE, KHÔNG khuyến nghị cho TP.HCM)",
    )
    parser.add_argument("--gadm_path", help="Đường dẫn file GeoJSON GADM level-2 (nếu --source gadm_file)")
    parser.add_argument(
        "--provinces",
        default=None,
        help="Từ khóa tên tỉnh/thành (cột NAME_1), phân tách bởi '|' — tự bỏ dấu/khoảng trắng khi so khớp. "
        f"Mặc định: gộp cả 3 tỉnh/thành cũ đã sáp nhập vào TP.HCM mới ({'|'.join(DEFAULT_PROVINCE_KEYWORDS)})",
    )
    parser.add_argument("--out", default="../data/hcmc_boundary.geojson", help="Đường dẫn file GeoJSON xuất ra")
    parser.add_argument("--no_dissolve", action="store_true", help="KHÔNG gộp feature cùng tên (mặc định: gộp)")
    parser.add_argument(
        "--web_out", default="../data/hcmc_boundary_web.geojson",
        help="Bản NHẸ cho WebGIS/PostGIS (đơn giản hóa hình học); '' để bỏ qua",
    )
    parser.add_argument("--simplify_deg", type=float, default=0.0003, help="Dung sai đơn giản hóa bản web (độ, ~0.0003 ≈ 30 m)")
    args = parser.parse_args()

    if args.source == "gaul":
        if not args.project:
            sys.exit("Thiếu --project (GEE Cloud Project ID) khi dùng --source gaul")
        init_ee(args.project)
        matches = inspect_gaul_names()
        if not matches:
            sys.exit("❌ Không tìm thấy tên khớp trong GAUL — thử lại với --source gadm_file.")
        chosen = matches[0]
        print(f"-> Dùng ADM1_NAME = '{chosen}' (tự kiểm tra lại danh sách ở trên nếu nghi ngờ)")
        hcmc_fc = get_hcmc_boundary_from_gaul(chosen)
        export_boundary_geojson_from_ee(hcmc_fc, args.out)
    else:
        if not args.gadm_path:
            sys.exit("Thiếu --gadm_path khi dùng --source gadm_file")
        hcmc_gdf = get_hcmc_boundary_from_gadm_file(args.gadm_path, args.provinces, dissolve=not args.no_dissolve)
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        hcmc_gdf.to_file(args.out, driver="GeoJSON")
        if args.web_out:
            web = hcmc_gdf.copy()
            web["geometry"] = web.geometry.simplify(args.simplify_deg, preserve_topology=True)
            web.to_file(args.web_out, driver="GeoJSON")
            print(f"   (Bản nhẹ cho WebGIS: {args.web_out})")
        _print_done(args.out, len(hcmc_gdf))
