"""
Kiểm tra nhanh điều kiện lọc + độ "tươi" của dữ liệu nguồn (KHÔNG build dataset)
==================================================================================
CẬP NHẬT 2026-10-19 (cùng đợt sửa luồng lấy dữ liệu):
  - Dùng ROI NHẸ (load_boundary_roi) thay polygon GADM chi tiết — request nhỏ hơn hàng chục lần.
  - Bước "ngày" chỉ đọc METADATA (list_landsat_scene_times) -> vài giây, không timeout.
  - Bước pixel (get_valid_landsat_dates) so sánh NHIỀU ngưỡng ST_QA (--compare_k 2,3,5): cho thấy
    ngưỡng độ bất định làm rụng bao nhiêu ngày. Lưu ý: "ngày có >= 1 pixel hợp lệ ở bất kỳ đâu trong
    ROI" chỉ là chỉ số THÔ để so sánh các ngưỡng với nhau — ngày hợp lệ THẬT của dataset được xác định
    bằng độ phủ theo diện tích (04 --stage lst + check_t7_pairing.py).
  - MỚI: in ngày mới nhất của ERA5-Land, Sentinel-2 SR, Sentinel-2 cloud-probability, Landsat 8/9.
    ERA5-Land trên GEE thường trễ vài tháng so với Landsat -> các ngày Landsat gần đây KHÔNG có ERA5
    (bản cũ âm thầm loại các ngày này; 04 bản mới tự bỏ qua và báo rõ).

In ra:
  1. L8 / L9: số cảnh + ngày sớm nhất/mới nhất (filterBounds + filterDate)
  2. Sau merge: số cảnh, số NGÀY khác nhau
  3. Số ngày có >= 1 pixel hợp lệ theo từng ngưỡng ST_QA (không z-score; --zscore để bật, nặng hơn)
  4. Ngày mới nhất của các nguồn + số ngày Landsat chưa có ERA5

Chạy TRƯỚC 04_build_dataset.py --rebuild_raw để chắc điều kiện lọc không cắt mất dữ liệu ngoài ý muốn.

Cách chạy:
    python check_filter_stages.py --project YOUR_ID
    python check_filter_stages.py --project YOUR_ID --compare_k 2,3,5 --chunk_months 1   # chia nhỏ nếu timeout
    python check_filter_stages.py --project YOUR_ID --quick                               # chỉ metadata, bỏ bước pixel
"""

import argparse
import importlib.util
import os
from datetime import datetime, timezone

import ee

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prepare_features", os.path.join(_THIS_DIR, "03_prepare_features.py")
)
prepare_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_features)


def print_range(col: "ee.ImageCollection", label: str) -> list:
    n = prepare_features.ee_retry(lambda: col.size().getInfo(), label=f"size {label}")
    if n == 0:
        print(f"[{label}] 0 cảnh — RỖNG!")
        return []
    ts_list = prepare_features.ee_retry(lambda: col.aggregate_array("system:time_start").getInfo(), label=f"dates {label}")
    dates = sorted({datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for t in ts_list})
    print(f"[{label}] {n} cảnh trên {len(dates)} ngày | sớm nhất={dates[0]} | mới nhất={dates[-1]}")
    return dates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--chunk_months", type=int, default=3,
                        help="Chia truy vấn pixel thành từng đợt N tháng để tránh timeout (giảm xuống 1 nếu vẫn timeout).")
    parser.add_argument("--compare_k", default="2,3,5", help="Các ngưỡng ST_QA (K) cần so sánh ở bước pixel")
    parser.add_argument("--zscore", action="store_true", help="Bật z-score cục bộ ở bước pixel (nặng hơn nhiều)")
    parser.add_argument("--quick", action="store_true", help="Chỉ đọc metadata (bỏ bước pixel)")
    args = parser.parse_args()

    prepare_features.init_ee(args.project)
    roi = prepare_features.load_boundary_roi(args.boundary)
    print(f"Khoảng ngày truy vấn: {args.start} -> {args.end}\n")

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(args.start, args.end)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(args.start, args.end)
    d8 = print_range(l8, "Landsat 8 — filterBounds+filterDate")
    d9 = print_range(l9, "Landsat 9 — filterBounds+filterDate")
    merged = l8.merge(l9).sort("system:time_start")
    dm = print_range(merged, "L8+L9 — sau merge (TRƯỚC mọi bước lọc pixel)")
    if not dm:
        raise SystemExit("❌ Không có cảnh Landsat nào chạm ROI trong khoảng ngày này.")

    valid_by_k = {}
    if args.quick:
        print("\n(--quick: bỏ qua bước lọc pixel)")
    else:
        ks = [float(x) for x in args.compare_k.split(",") if x.strip()]
        print(f"\nBước pixel: đếm NGÀY có >= 1 pixel hợp lệ (cửa sổ {args.chunk_months} tháng/lượt, "
              f"z-score {'BẬT' if args.zscore else 'tắt'}) cho từng ngưỡng ST_QA {ks} K ...")
        for k in ks:
            vd = prepare_features.get_valid_landsat_dates(
                roi, args.start, args.end, chunk_months=args.chunk_months, verbose=False,
                apply_zscore=args.zscore, max_st_unc_k=k,
            )
            valid_by_k[k] = vd
            if vd:
                print(f"   ST_QA <= {k:g} K: {len(vd):>4} ngày | sớm nhất={vd[0]} | mới nhất={vd[-1]}")
            else:
                print(f"   ST_QA <= {k:g} K:    0 ngày — RỖNG!")
        if len(valid_by_k) > 1 and all(valid_by_k.values()):
            ks_sorted = sorted(valid_by_k)
            base = len(valid_by_k[ks_sorted[-1]])
            print("   Mất so với ngưỡng nới nhất (" + f"{ks_sorted[-1]:g} K" + "): " + ", ".join(
                f"{k:g} K: {base - len(valid_by_k[k])} ngày ({(base - len(valid_by_k[k])) / max(base, 1) * 100:.1f}%)" for k in ks_sorted[:-1]))
        print("   (Chỉ số THÔ: 'có >= 1 pixel ở bất kỳ đâu trong ROI'. Ngày hợp lệ thật dựa trên độ phủ — xem check_t7_pairing.py.)")

    # --- độ tươi của các nguồn
    print("\nNgày MỚI NHẤT của các nguồn dữ liệu (UTC):")
    latest = {}
    for name, cid in [("ERA5-Land hourly", prepare_features.ERA5_COLLECTION), ("Landsat 8 C2L2", "LANDSAT/LC08/C02/T1_L2"),
                      ("Landsat 9 C2L2", "LANDSAT/LC09/C02/T1_L2"), ("Sentinel-2 SR", prepare_features.S2_SR_COLLECTION),
                      ("Sentinel-2 cloud-prob", prepare_features.S2_CLOUD_PROB_COLLECTION)]:
        try:
            latest[name] = prepare_features.get_collection_max_date(cid, roi if "Landsat" in name else None)
        except Exception as e:  # noqa: BLE001
            latest[name] = None
            print(f"   ⚠️  {name}: không đọc được ({str(e)[:80]})")
        if latest[name]:
            print(f"   {name:<24}{latest[name]}")
    era5 = latest.get("ERA5-Land hourly")
    if era5:
        no_era5 = [d for d in dm if d > era5]
        print(f"\n   Landsat có {len(no_era5)} ngày SAU ngày ERA5 mới nhất ({era5}) -> các ngày này KHÔNG dùng được làm t "
              f"(thiếu ERA5)" + (f": {no_era5[:4]}{' ...' if len(no_era5) > 4 else ''}" if no_era5 else ""))
        if no_era5:
            print("   -> 04 --stage features tự bỏ qua chúng và báo rõ; khi ERA5 cập nhật, chạy lại stage này (cache giữ nguyên phần đã xong).")
    s2 = latest.get("Sentinel-2 SR")
    prob = latest.get("Sentinel-2 cloud-prob")
    if s2 and prob and prob < s2:
        print(f"   ⚠️  Cloud-probability ({prob}) trễ hơn S2 SR ({s2}): ảnh S2 mới nhất chưa có xác suất mây — mặt nạ s2cloudless "
              f"không áp dụng được cho chúng (code vẫn chạy nhờ nhánh dự phòng, nhưng mặt nạ mây yếu hơn).")

    print(
        "\n=> Đọc kết quả:\n"
        "   - 'mới nhất' của L8+L9 sau merge và của bước pixel gần nhau: điều kiện lọc không cắt mất dữ liệu gần đây.\n"
        "   - Số ngày tụt mạnh khi hạ ngưỡng ST_QA (2 K so với 5 K): ngưỡng độ bất định là yếu tố quyết định số mẫu\n"
        "     — dùng diagnose_mask_components.py để xem chi tiết theo từng quận/bước mặt nạ.\n"
        "   - Có ngày Landsat sau ERA5 mới nhất: bình thường (ERA5 trễ), không phải lỗi lọc."
    )


if __name__ == "__main__":
    main()
