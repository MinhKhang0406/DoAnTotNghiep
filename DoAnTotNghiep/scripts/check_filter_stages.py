"""
Kiểm tra nhanh điều kiện lọc Landsat (KHÔNG chạy build dataset đầy đủ).
==========================================================================
CẬP NHẬT 2026-09-11: bước "SAU lọc mây/bóng mây/lỗi cảm biến" giờ dùng
prepare_features.get_valid_landsat_dates() — CHIA NHỎ theo từng quý thay vì
gọi .getInfo() 1 lần cho toàn bộ 2019-2026 — vì với ROI TP.HCM mới (gộp cả
Bình Dương + BR-VT, lớn hơn hẳn TP.HCM cũ) cộng bộ lọc z-score cục bộ mới
thêm, gọi 1 lần cho >1000 ảnh rất dễ bị GEE báo lỗi "Computation timed out"
(giới hạn ~5 phút/request đồng bộ).

In số ảnh + ngày sớm nhất/mới nhất sau từng bước lọc:
  1. filterBounds + filterDate riêng L8, riêng L9
  2. Sau merge (trước khi lọc mây/lỗi cảm biến)
  3. Sau lọc mây/bóng mây/lỗi cảm biến (đếm pixel thật còn lại, vì
     updateMask() không tự xóa ảnh khỏi collection) — CHIA NHỎ theo quý.

Chạy cái này TRƯỚC khi chạy 04_build_dataset.py --rebuild_raw (rất chậm)
để chắc chắn điều kiện lọc không cắt mất dữ liệu ngoài ý muốn.

Cách chạy:
    python check_filter_stages.py --project YOUR_ID
    python check_filter_stages.py --project YOUR_ID --chunk_months 1   # nếu vẫn timeout, chia nhỏ hơn nữa
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


def print_range(col: "ee.ImageCollection", label: str) -> None:
    n = col.size().getInfo()
    if n == 0:
        print(f"[{label}] 0 ảnh — RỖNG!")
        return
    ts_list = col.aggregate_array("system:time_start").getInfo()
    dates = [datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d") for t in ts_list]
    print(f"[{label}] {n} ảnh | sớm nhất={min(dates)} | mới nhất={max(dates)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument(
        "--chunk_months", type=int, default=3,
        help="Chia truy vấn GEE thành từng đợt N tháng để tránh timeout. Nếu vẫn timeout, giảm xuống 1.",
    )
    args = parser.parse_args()

    prepare_features.init_ee(args.project)
    roi = prepare_features.load_boundary_geom(args.boundary)
    print(f"Khoảng ngày truy vấn: {args.start} -> {args.end}\n")

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(args.start, args.end)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(args.start, args.end)
    print_range(l8, "Landsat 8 — filterBounds+filterDate")
    print_range(l9, "Landsat 9 — filterBounds+filterDate")

    merged = l8.merge(l9).sort("system:time_start")
    print_range(merged, "L8+L9 — sau merge (TRƯỚC lọc mây/lỗi cảm biến)")

    print("\nĐang tính 'SAU lọc mây/bóng mây/lỗi cảm biến' (chia theo từng "
          f"{args.chunk_months} tháng để tránh timeout)...")
    valid_dates = prepare_features.get_valid_landsat_dates(
        roi, args.start, args.end, chunk_months=args.chunk_months
    )
    if not valid_dates:
        print("\n[SAU lọc mây/bóng mây/lỗi cảm biến] 0 ngày — RỖNG!")
    else:
        print(
            f"\n[SAU lọc mây/bóng mây/lỗi cảm biến] {len(valid_dates)} ngày hợp lệ "
            f"| sớm nhất={valid_dates[0]} | mới nhất={valid_dates[-1]}"
        )

    print(
        "\n=> So sánh 'mới nhất' ở dòng cuối với dòng 'sau merge' phía trên:\n"
        "   - Nếu gần bằng nhau: điều kiện lọc ổn, không cắt mất dữ liệu gần đây.\n"
        "   - Nếu lệch xa (ví dụ mới nhất tụt hẳn về quá khứ): ngưỡng lọc\n"
        "     (ST_QA <= 2.0, QA_RADSAT, z-score cục bộ, khoảng vật lý -10..60°C)\n"
        "     đang quá chặt, cần xem lại trong 03_prepare_features.py."
    )


if __name__ == "__main__":
    main()
