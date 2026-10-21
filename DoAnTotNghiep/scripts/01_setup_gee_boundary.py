"""
BƯỚC 1 — Setup Earth Engine + lấy ranh giới hành chính THẬT của TP.HCM
=======================================================================
BẢN CẬP NHẬT 2026-09-08 — xử lý việc TP.HCM đã sáp nhập với Bình Dương và
Bà Rịa - Vũng Tàu từ 1/7/2025 (và cả nước đã bỏ cấp quận/huyện, chỉ còn
tỉnh/thành -> xã/phường). Ảnh hưởng tới bước này:

  1. GAUL (--source gaul, mặc định cũ) chỉ trả về ~4 đơn vị cho TP.HCM —
     bộ dữ liệu FAO GAUL 2015 vốn không cập nhật từ 2015 và rất kém chi
     tiết ở cấp quận/huyện cho nhiều nước, trong đó có Việt Nam. Không
     dùng --source gaul nữa cho đồ án này.

  2. GADM (--source gadm_file) chi tiết hơn nhiều NHƯNG dữ liệu GADM vẫn
     là ranh giới TRƯỚC sáp nhập (chỉ có TP.HCM cũ ~22 quận/huyện, KHÔNG
     có Bình Dương/BR-VT gộp vào). Hiện chưa có bộ dữ liệu polygon cấp
     xã/phường MỚI (168 đơn vị) nào phát hành miễn phí đầy đủ để thay thế.

  => GIẢI PHÁP dùng trong bản này: lọc GADM lấy CẢ 3 tỉnh/thành CŨ đã sáp
     nhập (TP.HCM + Bình Dương + Bà Rịa - Vũng Tàu), rồi dùng UNION ranh
     giới quận/huyện cũ của cả 3 nơi này làm đơn vị phân vùng không gian
     (district_idx ở Bước 4, thay cho khái niệm "quận/huyện" nay không
     còn tồn tại về mặt hành chính). Tổng diện tích/hình dạng khớp với
     TP.HCM mới; chỉ có TÊN đơn vị con là dùng theo địa giới cũ — cần ghi
     rõ điều này trong phần "Giới hạn của đề tài" khi viết đồ án.

  Nếu chỉ cần đường viền ngoài (không cần chia theo đơn vị con) để làm
  layer hiển thị ranh giới TP.HCM mới trên WebGIS, dùng file
  hcmc_new_outline_2025.geojson (đã tách sẵn từ nguồn dưới đây) thay vì
  chạy lại script này — 1 feature polygon duy nhất, không có cột quận/
  huyện.
  Nguồn: https://github.com/nguyenduy1133/Free-GIS-Data (mục "Vietnam
  Administrative Divisions (Post-2025)"), đối chiếu Nghị quyết 202/2025/
  QH15 — diện tích/dân số/số xã trong file khớp với số liệu chính thức
  (~6.772 km², ~13,7-14,7 triệu dân tùy nguồn thống kê, 168 xã/phường).

Mục tiêu bước này (giữ nguyên từ bản gốc):
  1. Khởi tạo & xác thực Google Earth Engine (GEE).
  2. Lấy ranh giới cấp quận/huyện (cũ) của TP.HCM + Bình Dương + BR-VT,
     thay cho placeholder hình vuông đang dùng trong WebGIS.
  3. Xuất ra 1 file GeoJSON duy nhất, dùng xuyên suốt các bước sau.

Cách chạy (đã activate env lst-thesis):
    # Mặc định: lọc gộp cả 3 tỉnh/thành cũ (khớp TP.HCM mới)
    python 01_setup_gee_boundary.py --source gadm_file --gadm_path duong_dan/gadm41_VNM_2.json

    # Tự chọn tỉnh/thành khác (regex, phân tách bởi |), ví dụ chỉ lấy TP.HCM cũ:
    python 01_setup_gee_boundary.py --source gadm_file --gadm_path duong_dan/gadm41_VNM_2.json \
        --provinces "Ho Chi Minh|Hồ Chí Minh"

Tải GADM level-2 Việt Nam tại: https://gadm.org/download_country.html
  (chọn Vietnam > GeoJSON)
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
    """Chuẩn hóa tên riêng tiếng Việt để so khớp KHÔNG phụ thuộc cách ghi của
    từng nguồn dữ liệu: bỏ dấu, bỏ khoảng trắng/gạch nối, hạ chữ thường.
    Lý do cần hàm này: GADM có thể ghi "Bà Rịa - Vũng Tàu" (có dấu, có
    khoảng trắng quanh gạch nối) trong khi pattern cũ dùng để so khớp lại
    viết liền không khoảng trắng kiểu "BàRịa"/"VũngTàu" -> KHÔNG BAO GIỜ
    khớp được vì chuỗi thật luôn có khoảng trắng giữa các âm tiết. Chuẩn
    hóa cả 2 phía (dữ liệu thật + từ khóa lọc) về cùng 1 dạng thì mới chắc
    ăn, bất kể nguồn ghi có dấu hay không dấu, "Ba Ria-Vung Tau" hay
    "Bà Rịa - Vũng Tàu" hay "Ho Chi Minh City" hay "TP. Hồ Chí Minh"."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # bỏ dấu
    text = text.replace("đ", "d").replace("Đ", "D")
    text = re.sub(r"[^a-zA-Z]", "", text)  # bỏ khoảng trắng, gạch nối, dấu chấm...
    return text.lower()


# Mặc định: 3 tỉnh/thành CŨ đã gộp thành TP.HCM mới từ 1/7/2025 — viết dưới
# dạng từ khóa ĐÃ CHUẨN HÓA (không dấu, không khoảng trắng) để so khớp an
# toàn, không còn phụ thuộc GADM ghi có dấu/khoảng trắng/gạch nối hay không.
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
    """Giữ lại để tham khảo — KHÔNG khuyến khích dùng cho đồ án này vì GAUL
    quá thiếu chi tiết ở cấp quận/huyện cho TP.HCM (xem cảnh báo ở đầu file)."""
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


def get_hcmc_boundary_from_gadm_file(gadm_geojson_path: str, province_pattern: str = None) -> gpd.GeoDataFrame:
    """Lọc GADM level-2 Việt Nam theo tên tỉnh/thành CŨ (NAME_1).
    Mặc định lấy cả 3 tỉnh/thành đã sáp nhập vào TP.HCM mới (1/7/2025):
    TP.HCM + Bình Dương + Bà Rịa - Vũng Tàu — vì GADM chưa có ranh giới
    xã/phường mới, union quận/huyện cũ của cả 3 nơi này dùng thay thế làm
    đơn vị phân vùng không gian cho Bước 4 (district_idx).

    So khớp bằng tên đã CHUẨN HÓA (bỏ dấu + khoảng trắng + gạch nối, hạ chữ
    thường) ở cả 2 phía — chứ không .str.contains() trực tiếp trên NAME_1
    gốc — vì bản trước dùng pattern viết liền không khoảng trắng (kiểu
    "BàRịa") trong khi NAME_1 thật của GADM luôn có khoảng trắng giữa âm
    tiết (kiểu "Bà Rịa"), nên KHÔNG BAO GIỜ khớp được, dù case=False."""
    gdf = gpd.read_file(gadm_geojson_path)
    print("Các cột có sẵn trong file GADM:", list(gdf.columns))
    if "NAME_1" not in gdf.columns:
        raise ValueError("Không thấy cột NAME_1 — kiểm tra lại đúng file GADM level-2 chưa.")
    print("Giá trị NAME_1 duy nhất:", sorted(gdf["NAME_1"].unique().tolist()))

    if province_pattern:
        # Vẫn hỗ trợ truyền tay qua --provinces (phân tách bởi '|'), nhưng
        # chuẩn hóa luôn từng phần tử cho chắc, không bắt người dùng phải
        # tự gõ đúng dấu/khoảng trắng y hệt NAME_1 gốc.
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
            "hoặc --provinces cho khớp cách GADM ghi tên (vd nếu GADM ghi 'Ho Chi Minh City' "
            "thì keyword 'hochiminh' vẫn khớp bình thường vì chỉ cần là substring)."
        )

    print(f"Số quận/huyện (cũ, dùng làm đơn vị phân vùng không gian) tìm thấy: {len(hcmc_gdf)}")
    print("   Các tỉnh/thành cũ được gộp vào:", sorted(hcmc_gdf["NAME_1"].unique().tolist()))

    # Chuẩn hóa tên cột về đúng schema mà 04_build_dataset.py đang mong đợi
    # (04 hardcode "ADM2_NAME" theo schema GAUL — GADM dùng NAME_1/NAME_2,
    # đổi tên ở đây để 02/03/04 dùng lại được luôn, không cần sửa gì thêm).
    rename_map = {}
    if "NAME_1" in hcmc_gdf.columns:
        rename_map["NAME_1"] = "ADM1_NAME"
    if "NAME_2" in hcmc_gdf.columns:
        rename_map["NAME_2"] = "ADM2_NAME"
    hcmc_gdf = hcmc_gdf.rename(columns=rename_map)

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
    print(f"✅ Đã lưu ranh giới thật ({n_features} đơn vị phân vùng) vào: {out_path}")
    print("   Việc tiếp theo:")
    print("   1. Thay layer placeholder hình vuông trong WebGIS bằng file này")
    print("      (hoặc dùng hcmc_new_outline_2025.geojson nếu chỉ cần đường viền ngoài).")
    print("   2. Giữ nguyên file này để Bước 2 (thu thập Landsat/Sentinel-2) dùng làm vùng clip.")
    print("   3. Ghi lại tên 2-3 đơn vị sẽ dùng làm test set không gian sau này — cân nhắc")
    print("      chọn thêm 1 đơn vị thuộc Bình Dương/BR-VT cũ để test khả năng khái quát hóa")
    print("      sang vùng mới sáp nhập, không chỉ Cần Giờ/Củ Chi như bản trước.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", help="GEE Cloud Project ID (bắt buộc nếu --source gaul)")
    parser.add_argument(
        "--source",
        choices=["gaul", "gadm_file"],
        default="gadm_file",
        help="Nguồn ranh giới: 'gadm_file' (khuyến nghị — file GADM tự tải) hoặc 'gaul' (qua GEE, "
        "KHÔNG khuyến nghị cho TP.HCM vì quá thiếu chi tiết, xem đầu file)",
    )
    parser.add_argument("--gadm_path", help="Đường dẫn file GeoJSON GADM level-2 (nếu --source gadm_file)")
    parser.add_argument(
        "--provinces",
        default=None,
        help="Từ khóa tên tỉnh/thành (cột NAME_1) để lọc, phân tách bởi '|' — sẽ tự "
        "động bỏ dấu/khoảng trắng khi so khớp nên không cần gõ đúng y hệt NAME_1 gốc. "
        f"Mặc định: gộp cả 3 tỉnh/thành cũ đã sáp nhập vào TP.HCM mới ({'|'.join(DEFAULT_PROVINCE_KEYWORDS)})",
    )
    parser.add_argument("--out", default="../data/hcmc_boundary.geojson", help="Đường dẫn file GeoJSON xuất ra")
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
        if not args.project:
            sys.exit("Thiếu --project")
        hcmc_fc = get_hcmc_boundary_from_gaul(chosen)
        export_boundary_geojson_from_ee(hcmc_fc, args.out)
    else:
        if not args.gadm_path:
            sys.exit("Thiếu --gadm_path khi dùng --source gadm_file")
        hcmc_gdf = get_hcmc_boundary_from_gadm_file(args.gadm_path, args.provinces)
        hcmc_gdf.to_file(args.out, driver="GeoJSON")
        _print_done(args.out, len(hcmc_gdf))