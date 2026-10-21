"""
Kiểm tra PHÂN PHỐI delta_t thực tế (thay cho việc đếm cặp t+7 cũ)
==========================================================================
CẬP NHẬT 2026-09-09: File này TRƯỚC ĐÂY dùng để đếm % ngày ghép được cặp
đúng t+7 và % ghép được trong cửa sổ dung sai t+5..t+9 — để trả lời câu hỏi
"cố định 7 ngày có khả thi không". Câu hỏi đó ĐÃ ĐƯỢC TRẢ LỜI (không khả
thi) và đề cương đã chốt sang bài toán "dự báo linh hoạt cho kỳ quan sát
khả dụng tiếp theo" (xem 03_prepare_features.py, 04_build_dataset.py) —
nên khái niệm "cửa sổ dung sai" không còn áp dụng nữa, ghép mẫu giờ luôn
lấy NGÀY HỢP LỆ LIỀN KỀ TIẾP THEO bất kể cách xa bao nhiêu.

File này giờ chỉ còn 1 việc: thống kê PHÂN PHỐI delta_t thực tế (khoảng
cách ngày tới kỳ ảnh hợp lệ tiếp theo) trên TOÀN BỘ chuỗi ngày hợp lệ —
để có số liệu CHÍNH XÁC, TÁI LẶP ĐƯỢC cho câu trích dẫn ở đề cương mục 1.1:
"khoảng cách thời gian giữa hai bức ảnh khả dụng dao động rất lớn (từ 1
ngày đến 71 ngày, trung bình ~10.6 ngày)" — chạy file này ra đúng số liệu
đó (hoặc số liệu mới nếu biên dữ liệu/tham số lọc mây có thay đổi) để trích
dẫn chính xác trong báo cáo, tránh ghi số liệu cũ không khớp dữ liệu thật.

CẬP NHẬT 2026-09-11: dùng prepare_features.get_valid_landsat_dates() (CHIA
NHỎ theo từng quý) thay vì tự lấy danh sách ngày hợp lệ trong 1 lần gọi —
tránh lỗi "Computation timed out" của GEE với ROI lớn (TP.HCM mới) + nhiều
ảnh (>1000 ảnh, 2019-2026).

Cách chạy:
    python check_t7_pairing.py --project dacn-475523
    python check_t7_pairing.py --project dacn-475523 --chunk_months 1   # nếu vẫn timeout
"""

import argparse
import importlib.util
import os
from datetime import datetime

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prepare_features", os.path.join(_THIS_DIR, "03_prepare_features.py")
)
prepare_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_features)


def compute_delta_t_distribution(valid_dates: list) -> pd.Series:
    """delta_t giữa mỗi ngày hợp lệ và ngày hợp lệ LIỀN KỀ TIẾP THEO — đúng
    logic ghép mẫu thật sự dùng trong 04_build_dataset.py (không có cửa sổ
    dung sai, không bỏ qua cặp nào trừ ngày cuối cùng của chuỗi)."""
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in valid_dates]
    deltas = [(dates_dt[i + 1] - dates_dt[i]).days for i in range(len(dates_dt) - 1)]
    return pd.Series(deltas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument(
        "--chunk_months", type=int, default=3,
        help="Chia truy vấn GEE thành từng đợt N tháng để tránh 'Computation timed out'. Nếu vẫn timeout, giảm xuống 1.",
    )
    args = parser.parse_args()

    prepare_features.init_ee(args.project)
    roi = prepare_features.load_boundary_geom(args.boundary)

    print(f"Đang lấy danh sách ngày Landsat hợp lệ ({args.start} -> {args.end}, chia theo từng {args.chunk_months} tháng)...")
    valid_dates = prepare_features.get_valid_landsat_dates(roi, args.start, args.end, chunk_months=args.chunk_months)
    n_total = len(valid_dates)
    print(f"\nTổng số NGÀY có ảnh Landsat hợp lệ sau lọc mây/bóng mây/lỗi cảm biến: {n_total}\n")

    if n_total < 2:
        print("❌ Không đủ ngày hợp lệ để tính delta_t (cần ít nhất 2 ngày).")
        return

    deltas = compute_delta_t_distribution(valid_dates)

    print("=" * 65)
    print(" PHÂN PHỐI delta_t (khoảng cách ngày tới kỳ ảnh hợp lệ tiếp theo)")
    print("=" * 65)
    print(f"Số cặp (t, t+delta_t) sẽ tạo được: {len(deltas)} (= {n_total} ngày hợp lệ - 1 ngày cuối chuỗi)")
    print(f"min={deltas.min()} | mean={deltas.mean():.2f} | median={deltas.median():.0f} | "
          f"max={deltas.max()} | std={deltas.std():.2f}")
    print("\nCác mốc phần trăm (percentile):")
    for p in [10, 25, 50, 75, 90, 95]:
        print(f"   p{p:<3} = {deltas.quantile(p / 100):.1f} ngày")
    print("\nSố ngày có delta_t đúng bằng 7 (để đối chiếu với giả định 't+7' ban đầu):")
    n_exact7 = int((deltas == 7).sum())
    print(f"   {n_exact7}/{len(deltas)} ({n_exact7 / len(deltas) * 100:.1f}%)")
    print("=" * 65)

    print("\n=> DÙNG SỐ LIỆU NÀY để trích dẫn chính xác trong đề cương/báo cáo mục 1.1")
    print("   (thay vì ghi cứng '~10.6 ngày, 1-71 ngày' nếu dữ liệu/tham số lọc mây")
    print("   ở lần chạy này khác với lần đã dùng để viết đề cương).")


if __name__ == "__main__":
    main()
