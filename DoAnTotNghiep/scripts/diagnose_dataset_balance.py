"""
Chẩn đoán độ cân bằng dữ liệu THẬT — TOÀN BỘ raw_dataset.csv phân bố theo quận/huyện, năm, mùa,
và (nếu đã chia) theo từng tập. Không cần GEE, không cần ranh giới, chạy vài giây.

CẬP NHẬT 2026-10-19:
  - BỎ sjoin với ranh giới (bản cũ nhân đôi điểm nằm vùng polygon chồng lấn, nên chính phép kiểm tra
    này làm lệch số dòng). Bản mới dùng thẳng các cột ADM1_NAME/ADM2_NAME/district_area_km2/point_id
    mà 04 bản mới ghi sẵn vào raw_dataset.csv (đơn vị đã dissolve, diện tích đã tính ở UTM 48N).
  - Khung điểm mới có SỐ ĐIỂM ỨNG VIÊN BẰNG NHAU mỗi quận (mặc định 100), nên "dòng/km²" không còn là
    thước đo bất thường như trước. Hai chỉ số đúng để đọc:
        dong_moi_diem   = số dòng / số điểm khung của quận  -> mức độ quận bị MASK/mây "ăn" dữ liệu
        ty_le_diem_dung = số điểm khung có >= 1 dòng / số điểm khung
    Quận có dong_moi_diem thấp bất thường (< 25% trung vị) -> chạy diagnose_mask_components.py để biết
    bước mặt nạ nào loại pixel ở đó.
  - MỚI: phân bố theo năm/mùa (kích thước Train/Val/Test theo thời gian), theo tập đã chia, chỉ số tập
    trung (Gini, số đơn vị hiệu dụng).

LƯU Ý THIẾT KẾ: lấy mẫu đều theo quận khiến quận nhỏ (đô thị lõi) được đại diện NHIỀU hơn so với diện tích
của chúng, quận lớn nông thôn (Cần Giờ, Dầu Tiếng...) ít hơn. Đây là lựa chọn có chủ đích (để mô hình thấy đủ
đô thị) nhưng cần NÊU trong báo cáo; khi tính SUHII toàn vùng, 07 đã trọng số theo diện tích.

Cách chạy:
    python diagnose_dataset_balance.py
    python diagnose_dataset_balance.py --data_dir ../data --top 12
"""
import argparse
import os

import numpy as np
import pandas as pd


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    raw_path = os.path.join(args.data_dir, "raw_dataset.csv")
    if not os.path.exists(raw_path):
        raise SystemExit(f"❌ Không thấy {raw_path} — chạy 04_build_dataset.py trước.")
    raw = pd.read_csv(raw_path)
    need = ["point_id", "date_t", "ADM2_NAME", "district_area_km2"]
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise SystemExit(f"❌ raw_dataset.csv thiếu cột {miss} — file từ bản 04 CŨ; xóa và build lại bằng --rebuild_raw.")
    frame_path = os.path.join(args.data_dir, "sampling_frame.csv")
    frame = pd.read_csv(frame_path) if os.path.exists(frame_path) else None

    print(f"Tổng số dòng trong raw_dataset.csv: {len(raw):,} | {raw['point_id'].nunique():,} điểm | "
          f"{raw['date_t'].nunique()} ngày t | {raw['ADM2_NAME'].nunique()} đơn vị có dòng")

    # ---------- theo quận
    g = raw.groupby(["ADM1_NAME", "ADM2_NAME"]).agg(
        so_dong=("point_id", "size"), so_diem_dung=("point_id", "nunique"), dien_tich_km2=("district_area_km2", "first"),
    ).reset_index()
    if frame is not None:
        fr = frame.groupby(["ADM1_NAME", "ADM2_NAME"]).agg(so_diem_khung=("point_id", "nunique"),
                                                            dien_tich_khung=("district_area_km2", "first")).reset_index()
        g = fr.merge(g, on=["ADM1_NAME", "ADM2_NAME"], how="left")
        g["so_dong"] = g["so_dong"].fillna(0).astype(int)
        g["so_diem_dung"] = g["so_diem_dung"].fillna(0).astype(int)
        g["dien_tich_km2"] = g["dien_tich_km2"].fillna(g["dien_tich_khung"])
        g = g.drop(columns="dien_tich_khung")
    else:
        g["so_diem_khung"] = g["so_diem_dung"]
        print("   (Không có sampling_frame.csv — dùng số điểm có dòng làm số điểm khung; đơn vị 0 dòng sẽ không hiện.)")
    g["dong_moi_diem"] = (g["so_dong"] / g["so_diem_khung"].clip(lower=1)).round(1)
    g["ty_le_diem_dung_pct"] = (g["so_diem_dung"] / g["so_diem_khung"].clip(lower=1) * 100).round(1)
    g["dong_tren_km2"] = (g["so_dong"] / g["dien_tich_km2"].clip(lower=1e-6)).round(2)
    g["ty_le_dong_pct"] = (g["so_dong"] / max(g["so_dong"].sum(), 1) * 100).round(2)
    g["ty_le_dien_tich_pct"] = (g["dien_tich_km2"] / g["dien_tich_km2"].sum() * 100).round(2)
    g["dai_dien_x"] = (g["ty_le_dong_pct"] / g["ty_le_dien_tich_pct"].clip(lower=1e-9)).round(2)
    g = g.sort_values("so_dong", ascending=False).reset_index(drop=True)

    med = float(g["dong_moi_diem"].median())
    low = g[g["dong_moi_diem"] < 0.25 * med]
    zero = g[g["so_dong"] == 0]
    cols = ["ADM2_NAME", "so_diem_khung", "so_diem_dung", "so_dong", "dong_moi_diem", "dien_tich_km2", "dong_tren_km2", "ty_le_dong_pct", "dai_dien_x"]
    pd.set_option("display.width", 200)
    print(f"\n=== {args.top} đơn vị NHIỀU dòng nhất ===")
    print(g[cols].head(args.top).round(2).to_string(index=False))
    print(f"\n=== {args.top} đơn vị ÍT dòng nhất ===")
    print(g[cols].tail(args.top).round(2).to_string(index=False))

    print("\n=== Tóm tắt cân bằng ===")
    print(f"dòng/điểm khung: min={g['dong_moi_diem'].min():.1f} | trung vị={med:.1f} | max={g['dong_moi_diem'].max():.1f}")
    pr = g["so_dong"] / max(g["so_dong"].sum(), 1)
    print(f"Gini số dòng giữa các đơn vị = {gini(g['so_dong']):.3f} (0 = đều tuyệt đối) | số đơn vị hiệu dụng = {1 / float((pr ** 2).sum()):.1f}/{len(g)}")
    if len(zero):
        print(f"⚠️  {len(zero)} đơn vị 0 dòng: {zero['ADM2_NAME'].tolist()[:15]}{' ...' if len(zero) > 15 else ''}")
    if len(low):
        print(f"⚠️  {len(low)} đơn vị có dòng/điểm < 25% trung vị (bị mask/mây ăn nặng): {low['ADM2_NAME'].tolist()[:15]}"
              f"{' ...' if len(low) > 15 else ''}\n   -> chạy: python diagnose_mask_components.py --project ID --districts \"" +
              ",".join(low['ADM2_NAME'].head(6).tolist()) + "\"")
    if not len(zero) and not len(low):
        print("✅ Không có đơn vị nào bị mất dữ liệu bất thường — chênh lệch số dòng giữa các đơn vị chủ yếu do số điểm khung/độ phủ ngày.")
    top_share = g["ty_le_dong_pct"].head(3).sum()
    print(f"3 đơn vị nhiều dòng nhất chiếm {top_share:.1f}% tổng dòng (đều tuyệt đối ≈ {3 / len(g) * 100:.1f}%).")
    big = g[g["dai_dien_x"] > 3]
    if len(big):
        print(f"Đơn vị được đại diện > 3 lần so với tỉ lệ diện tích (do lấy mẫu đều theo quận): {big['ADM2_NAME'].head(8).tolist()}"
              f"{' ...' if len(big) > 8 else ''}")
    print("   Ghi chú báo cáo: lấy mẫu đều theo quận => đô thị lõi (diện tích nhỏ) được đại diện nhiều hơn diện tích thật; SUHII đã trọng số theo diện tích (07).")

    # ---------- theo năm / mùa
    raw["_year"] = pd.to_datetime(raw["date_t"]).dt.year
    raw["_season"] = np.where(pd.to_datetime(raw["date_t"]).dt.month.isin(range(5, 12)), "Mùa mưa", "Mùa khô")
    ty = raw.groupby("_year").agg(so_dong=("point_id", "size"), so_ngay_t=("date_t", "nunique")).assign(
        dong_moi_ngay=lambda x: (x["so_dong"] / x["so_ngay_t"]).round(0))
    print("\n=== Theo năm của ngày t ===")
    print(ty.to_string())
    ts = raw.groupby("_season").agg(so_dong=("point_id", "size"), so_ngay_t=("date_t", "nunique")).assign(
        dong_moi_ngay=lambda x: (x["so_dong"] / x["so_ngay_t"]).round(0))
    print("\n=== Theo mùa (mùa mưa 5-11, mùa khô 12-4) ===")
    print(ts.to_string())
    ratio = float(ts["dong_moi_ngay"].min() / ts["dong_moi_ngay"].max()) if len(ts) == 2 else 1.0
    if ratio < 0.5:
        print(f"⚠️  Một mùa chỉ có {ratio * 100:.0f}% số dòng/ngày so với mùa kia (mây/mưa) — kết quả 'theo mùa' ở 08 của mùa đó kém ổn định hơn.")
    low_year = ty[ty["so_dong"] < 0.4 * ty["so_dong"].median()]
    if len(low_year):
        print(f"⚠️  Năm ít dòng bất thường (< 40% trung vị): {low_year.index.tolist()} (năm chưa trọn vẹn như 2026 là bình thường).")

    # ---------- theo tập đã chia
    sp_rows = []
    for name, fn in [("Train", "train.csv"), ("Validation", "val.csv"), ("Test thời gian", "test_temporal.csv"), ("Test không gian", "test_spatial.csv")]:
        p = os.path.join(args.data_dir, fn)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            d = pd.read_csv(p)
            top1 = d["ADM2_NAME"].value_counts(normalize=True)
            sp_rows.append({"tap": name, "so_dong": len(d), "so_don_vi": d["ADM2_NAME"].nunique(), "so_diem": d["point_id"].nunique(),
                            "so_ngay_t": d["date_t"].nunique(), "don_vi_lon_nhat": top1.index[0] if len(top1) else "",
                            "ty_le_don_vi_lon_nhat_pct": round(float(top1.iloc[0]) * 100, 1) if len(top1) else 0})
    if sp_rows:
        print("\n=== Theo tập đã chia ===")
        print(pd.DataFrame(sp_rows).to_string(index=False))
    else:
        print("\n(Chưa có train.csv/val.csv/... — chạy 04 --stage split để xem phân bố theo tập.)")

    out = os.path.join(args.data_dir, "dataset_balance_by_district.csv")
    g.to_csv(out, index=False)
    print(f"\n✅ Đã lưu {out}")


if __name__ == "__main__":
    main()
