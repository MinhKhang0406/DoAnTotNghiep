"""
Kiểm tra PHÂN PHỐI delta_t thực tế + độ phủ ngày hợp lệ (đọc từ CACHE của 04)
================================================================================
CẬP NHẬT 2026-10-19 (viết lại cùng đợt sửa luồng lấy dữ liệu):

Bản cũ lấy "ngày hợp lệ" = có >= 1 pixel Landsat sạch Ở BẤT KỲ ĐÂU trong ROI. Ngày chỉ phủ 1 dải
mỏng ở rìa (path/row khác) vẫn được tính -> sinh delta_t = 1 "ảo" và làm số trích dẫn trong
đề cương mục 1.1 ("từ 1 ngày đến 71 ngày, trung bình ~10.6 ngày") phụ thuộc vào đúng cái lỗi này.
Bản mới dùng ĐỘ PHỦ theo diện tích của từng ngày (tính từ cache `04 --stage lst`, KHÔNG gọi GEE):

  1. BẢNG QUÉT NGƯỠNG: với mỗi ngưỡng độ phủ (0.01 ... 0.5) -> số ngày hợp lệ, số cặp, delta_t
     min/trung vị/trung bình/max, % delta_t = 1. Dùng để CHỌN --min_valid_frac của 04 một cách có cơ sở.
  2. PHÂN PHỐI delta_t tại ngưỡng đang dùng (--min_valid_frac) — số liệu CHÍNH XÁC, TÁI LẶP ĐƯỢC để
     trích dẫn vào đề cương/báo cáo (thay cho con số cũ).
  3. SỐ CẶP THEO NĂM và theo tập (Train nhãn <= 2023 | Val = 2024 | Test t >= 2025 | loại do khác giai
     đoạn) — để biết Validation 2024 sẽ có bao nhiêu cặp TRƯỚC khi chạy tiếp.
  4. (nếu đã chạy `04 --stage features`) Độ sẵn có của S2 + ERA5 trên các ngày hợp lệ — nút thắt thật
     của dataset thường là S2 (cửa sổ [t-3, t]), không phải Landsat.
  5. Các cặp delta_t = 1: độ phủ của CẢ HAI ngày — phân biệt cặp thật (2 path liền kề cùng phủ rộng) với
     "dải mỏng".
Lưu ../data/delta_t_distribution.csv (mỗi cặp: date_t, date_target, delta_t, độ phủ, năm, tập).

Chưa có cache (chưa chạy 04 --stage lst)? Thêm --project: script chỉ liệt kê NGÀY CÓ CẢNH Landsat
(metadata, rất nhanh, KHÔNG lọc pixel) — số này là CẬN TRÊN về số ngày, đừng trích dẫn làm số liệu cuối.

Cách chạy:
    python check_t7_pairing.py                                  # đọc cache của 04 (không cần GEE)
    python check_t7_pairing.py --min_valid_frac 0.15
    python check_t7_pairing.py --project YOUR_ID                # chưa có cache: chỉ đếm ngày có cảnh
"""

import argparse
import importlib.util
import os
from datetime import datetime

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_THIS_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prepare_features = _load("prepare_features", "03_prepare_features.py")
build_dataset = _load("build_dataset", "04_build_dataset.py")


def compute_delta_t_distribution(valid_dates: list) -> pd.Series:
    """delta_t giữa mỗi ngày hợp lệ và ngày hợp lệ LIỀN KỀ TIẾP THEO — đúng logic ghép mẫu của 04
    (giữ lại tên hàm cũ cho tương thích)."""
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in valid_dates]
    return pd.Series([(dates_dt[i + 1] - dates_dt[i]).days for i in range(len(dates_dt) - 1)])


def pairs_table(tbl: pd.DataFrame) -> pd.DataFrame:
    """Bảng các cặp (t, t_next) từ bảng ngày hợp lệ (cột date, coverage)."""
    v = tbl.sort_values("date").reset_index(drop=True)
    if len(v) < 2:
        return pd.DataFrame(columns=["date_t", "date_target", "delta_t", "cov_t", "cov_target"])
    d = pd.to_datetime(v["date"])
    return pd.DataFrame({
        "date_t": v["date"].iloc[:-1].to_numpy(), "date_target": v["date"].iloc[1:].to_numpy(),
        "delta_t": (d.iloc[1:].to_numpy() - d.iloc[:-1].to_numpy()).astype("timedelta64[D]").astype(int),
        "cov_t": v["coverage"].iloc[:-1].to_numpy(), "cov_target": v["coverage"].iloc[1:].to_numpy(),
    })


def label_split(pairs: pd.DataFrame) -> pd.Series:
    """Tập mà mỗi cặp sẽ rơi vào — đúng quy tắc của split_dataset() trong 04 (chưa tách Test không gian)."""
    yt, yn = pd.to_datetime(pairs["date_t"]).dt.year, pd.to_datetime(pairs["date_target"]).dt.year
    s = np.where(yn <= build_dataset.TRAIN_END_YEAR, "Train",
        np.where((yt == build_dataset.VAL_YEAR) & (yn == build_dataset.VAL_YEAR), "Validation",
        np.where(yt >= build_dataset.TEST_START_YEAR, "Test", "Loại (t và nhãn khác giai đoạn)")))
    return pd.Series(s, index=pairs.index)


def describe(deltas: pd.Series, label: str) -> None:
    print("=" * 72)
    print(f" PHÂN PHỐI delta_t — {label}")
    print("=" * 72)
    print(f"Số cặp (t, t+delta_t): {len(deltas)}")
    print(f"min={deltas.min()} | mean={deltas.mean():.2f} | median={deltas.median():.0f} | max={deltas.max()} | std={deltas.std():.2f}")
    print("Phân vị: " + " | ".join(f"p{p}={deltas.quantile(p / 100):.0f}" for p in (10, 25, 50, 75, 90, 95)))
    n7 = int((deltas == 7).sum())
    print(f"delta_t đúng bằng 7 (đối chiếu giả định 't+7' cũ): {n7}/{len(deltas)} ({n7 / len(deltas) * 100:.1f}%)")
    vc = deltas.value_counts().sort_index()
    print("Đếm theo delta_t: " + ", ".join(f"{int(k)}:{int(v)}" for k, v in vc.head(14).items()) + (" ..." if len(vc) > 14 else ""))
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--min_valid_frac", type=float, default=0.10, help="Ngưỡng độ phủ để 1 ngày được tính là hợp lệ (khớp 04)")
    parser.add_argument("--project", default=None, help="GEE project — chỉ cần khi CHƯA có cache của 04 (chế độ đếm ngày có cảnh)")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-12-31")
    args = parser.parse_args()

    out_csv = os.path.join(args.data_dir, "delta_t_distribution.csv")
    frame_path = os.path.join(args.data_dir, "sampling_frame.csv")

    # ---------- có cache của 04 -> dùng độ phủ
    try:
        dates, lst = build_dataset.load_lst_cache(args.data_dir)
        have_cache = os.path.exists(frame_path)
    except SystemExit:
        have_cache = False

    if have_cache:
        frame = pd.read_csv(frame_path)
        tbl = build_dataset.valid_dates_table(dates, lst, frame, args.min_valid_frac)
        print(f"Cache LST của 04: {len(tbl)} ngày Landsat có dữ liệu; độ phủ (theo diện tích) trung vị {tbl['coverage'].median() * 100:.1f}%, "
              f"p90 {tbl['coverage'].quantile(0.9) * 100:.1f}%.")
        build_dataset.print_threshold_sweep(tbl, args.min_valid_frac)

        valid = tbl[tbl["valid"]]
        if len(valid) < 2:
            raise SystemExit(f"\n❌ Dưới 2 ngày hợp lệ ở ngưỡng {args.min_valid_frac} — hạ --min_valid_frac.")
        pairs = pairs_table(valid)
        pairs["year_t"] = pd.to_datetime(pairs["date_t"]).dt.year
        pairs["split"] = label_split(pairs)
        print(f"\nNgưỡng đang dùng: {args.min_valid_frac:.2f} -> {len(valid)} ngày hợp lệ "
              f"({valid['date'].iloc[0]} … {valid['date'].iloc[-1]}).\n")
        describe(pairs["delta_t"], f"ngưỡng độ phủ >= {args.min_valid_frac:.2f}")

        print("\nSỐ CẶP THEO NĂM (năm của ngày t) và THEO TẬP — mỗi cặp sẽ cho nhiều dòng (số điểm hợp lệ của cả 2 ngày):")
        by_year = pairs.groupby("year_t").agg(n_cap=("delta_t", "size"), dt_tb=("delta_t", "mean"), dt_max=("delta_t", "max"))
        print(by_year.round(1).to_string())
        by_split = pairs["split"].value_counts()
        print("\n" + "  |  ".join(f"{k}: {int(v)} cặp" for k, v in by_split.items()))
        for s in ("Validation", "Test"):
            if by_split.get(s, 0) < 10:
                print(f"   ⚠️  {s} chỉ có {int(by_split.get(s, 0))} cặp — đánh giá trên tập này sẽ kém ổn định (kiểm tra lại dữ liệu năm đó).")
        n_cross = int((pairs["split"].str.startswith("Loại")).sum())
        if n_cross:
            print(f"   ({n_cross} cặp bị loại vì t và ngày nhãn khác giai đoạn — đúng cơ chế purge ở split_dataset())")

        d1 = pairs[pairs["delta_t"] == 1]
        if len(d1):
            thin = d1[(d1["cov_t"] < 0.3) | (d1["cov_target"] < 0.3)]
            print(f"\nCẶP delta_t = 1: {len(d1)} cặp; trong đó {len(thin)} cặp có ít nhất 1 ngày độ phủ < 30% (nghi dải mỏng). "
                  f"Còn lại {len(d1) - len(thin)} cặp phủ rộng cả 2 ngày (2 path liền kề — hợp lệ).")
            if len(thin) > 0.5 * len(d1):
                print("   -> Phần lớn là dải mỏng: nâng --min_valid_frac (xem bảng quét ngưỡng) rồi chạy lại 04.")

        # S2 + ERA5
        fdates = os.path.join(args.data_dir, build_dataset.FEAT_DATES)
        if os.path.exists(fdates):
            fd = pd.read_csv(fdates).drop_duplicates("date", keep="last")
            fd = fd[fd["status"] == "ok"].copy()
            fd["avail"] = fd["n_complete"] / fd["n_points"].clip(lower=1)
            m = pairs.merge(fd[["date", "avail", "n_complete"]], left_on="date_t", right_on="date", how="left")
            n_no_feat = int(m["avail"].isna().sum())
            a = m["avail"].dropna()
            print(f"\nĐỘ SẴN CÓ S2 + ERA5 trên ngày t (tỉ lệ điểm có LST hợp lệ mà còn đủ NDVI/NDBI/NDWI + ERA5): "
                  f"trung vị {a.median() * 100:.0f}%, p10 {a.quantile(0.1) * 100:.0f}%, p90 {a.quantile(0.9) * 100:.0f}% ({len(a)} ngày t)")
            low = int((a < 0.3).sum())
            print(f"   {low} ngày t (<30% điểm đủ đặc trưng) gần như mất trắng; {n_no_feat} cặp chưa có cache đặc trưng (chạy 04 --stage features).")
            if len(a):
                mo = pd.to_datetime(m.loc[m["avail"].notna(), "date_t"]).dt.month
                by_season = pd.Series(a.to_numpy(), index=mo.to_numpy()).groupby(level=0).median()
                wet = by_season[by_season.index.isin(range(5, 12))].median() * 100
                dry = by_season[~by_season.index.isin(range(5, 12))].median() * 100
                print(f"   Trung vị theo mùa: mùa mưa (5-11) {wet:.0f}% | mùa khô (12-4) {dry:.0f}%"
                      + ("  -> S2 là nút thắt, nhất là mùa mưa: xem diagnose_mask_components.py --s2" if wet < 40 else ""))
            pairs = m.drop(columns=["date"]).rename(columns={"avail": "feat_avail", "n_complete": "feat_n_complete"})
        else:
            print("\n(Chưa có cache đặc trưng S2/ERA5 — chạy `04 --stage features` rồi chạy lại script này để xem độ sẵn có.)")

        os.makedirs(args.data_dir, exist_ok=True)
        pairs.to_csv(out_csv, index=False)
        print(f"\n✅ Đã lưu {out_csv}")
        print("\n=> DÙNG số liệu 'PHÂN PHỐI delta_t' ở trên để trích dẫn đề cương mục 1.1 (thay cho '1–71 ngày, TB ~10.6' nếu khác),")
        print("   và bảng quét ngưỡng để chốt --min_valid_frac. Mỗi lần đổi ngưỡng chỉ cần chạy lại script này (không gọi GEE).")
        return

    # ---------- chưa có cache -> chỉ đếm ngày có cảnh (cận trên)
    if not args.project:
        raise SystemExit("❌ Chưa có cache LST của 04 (cache_lst.csv/cache_lst_dates.csv/sampling_frame.csv).\n"
                         "   Cách 1 (khuyên dùng): python 04_build_dataset.py --project ID --stage lst   (tạo khung điểm + cache), rồi chạy lại script này.\n"
                         "   Cách 2 (chỉ ước lượng thô): python check_t7_pairing.py --project ID   (đếm ngày có cảnh, chưa lọc pixel).")
    prepare_features.init_ee(args.project)
    roi = prepare_features.load_boundary_roi(args.boundary)
    print(f"Chưa có cache của 04 -> liệt kê NGÀY CÓ CẢNH Landsat 8/9 chạm ROI ({args.start} -> {args.end}), CHƯA lọc mây/pixel...")
    scene_dates = sorted(prepare_features.list_landsat_scene_times(roi, args.start, args.end, chunk_months=6))
    if len(scene_dates) < 2:
        raise SystemExit("❌ Không đủ ngày có cảnh.")
    deltas = compute_delta_t_distribution(scene_dates)
    print(f"\nTổng số NGÀY có cảnh: {len(scene_dates)} (CẬN TRÊN — chưa trừ ngày mây/dải mỏng)\n")
    describe(deltas, "ngày CÓ CẢNH (chưa lọc pixel — chỉ để ước lượng thô)")
    print("\n⚠️  Đây KHÔNG phải số liệu cuối để trích dẫn: delta_t thật (sau lọc mây, theo độ phủ) lớn hơn. "
          "Chạy 04 --stage lst rồi chạy lại script này không kèm --project.")


if __name__ == "__main__":
    main()
