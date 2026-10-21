"""
BƯỚC 2 — (FILE NÀY ĐÃ NGƯNG DÙNG, xem CẬP NHẬT bên dưới)
==========================================================================
CẬP NHẬT 2026-09-18 — NGƯNG DÙNG file này, KHÔNG chạy nữa.

Lý do: file này dùng CLOUD_COVER (tỷ lệ % mây của CẢ CẢNH ảnh, lấy trực
tiếp từ metadata) làm tiêu chí "ảnh hợp lệ" — khác với định nghĩa dùng ở
TOÀN BỘ phần còn lại của pipeline (03_prepare_features.mask_landsat_c2l2,
04_build_dataset.py, check_filter_stages.py, check_t7_pairing.py): "hợp
lệ" = còn ít nhất 1 PIXEL sau khi mask THẬT bằng QA_PIXEL + QA_RADSAT +
ST_QA + z-score cục bộ + giới hạn vật lý (-10..60°C). Hai định nghĩa này
cho ra 2 con số delta_t KHÁC NHAU — nếu dùng số liệu từ file này để trích
dẫn đề cương mục 1.1 sẽ KHÔNG khớp với dataset thật được build ở Bước 4.

Ngoài ra file này gọi `merged.map(extract_meta)` rồi `.getInfo()` 1 lần
cho TOÀN BỘ collection L8+L9 filterBounds trên ROI mới (đã gộp thêm Bình
Dương + BR-VT, ROI phức tạp hơn hẳn TP.HCM cũ) x 8 năm (2019-2026) —
không chia nhỏ theo quý như get_valid_landsat_dates() ở 03, nên có rủi ro
riêng về hiệu năng/timeout ngoài vấn đề số liệu không khớp ở trên.

check_t7_pairing.py đã thay thế đúng chức năng "phân tích gap/delta_t" của
file này từ 2026-09-09 (đọc docstring của check_t7_pairing.py) — dùng
get_valid_landsat_dates() (chia nhỏ theo quý, tránh "Computation timed
out") và dùng ĐÚNG 1 định nghĩa "hợp lệ" xuyên suốt toàn bộ pipeline.

  ==> DÙNG THAY: python check_t7_pairing.py --project YOUR_ID

Giữ lại file này (không xóa) để không gãy tham chiếu nếu đề cương/ghi chú
cũ có nhắc tên "Bước 2". Script bên dưới CHỈ in cảnh báo rồi dừng, KHÔNG
tự gọi Earth Engine — tránh mất công chờ 1 query không cần thiết và tránh
vô tình lấy nhầm số liệu không nhất quán với dataset thật.
"""

import sys

if __name__ == "__main__":
    print("=" * 74)
    print(" ⚠️  02_analyze_landsat_gaps.py ĐÃ NGƯNG DÙNG (xem docstring đầu file)")
    print("=" * 74)
    print(
        "Lý do: tiêu chí 'ảnh hợp lệ' của file này (CLOUD_COVER % cả cảnh, lấy từ\n"
        "metadata) KHÔNG khớp với mask_landsat_c2l2 (mask theo pixel) dùng ở toàn\n"
        "bộ phần còn lại của pipeline (03/04/check_filter_stages/check_t7_pairing).\n"
        "Số liệu delta_t ra từ file này sẽ SAI/KHÔNG khớp dataset thật ở Bước 4."
    )
    print("\n👉 Dùng thay:  python check_t7_pairing.py --project YOUR_ID")
    print("   (cùng cấu trúc tham số --boundary/--start/--end/--chunk_months)\n")
    sys.exit(1)
