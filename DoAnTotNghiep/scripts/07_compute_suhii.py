"""
BƯỚC 7 — Tính chỉ số Đảo nhiệt đô thị bề mặt (SUHII)
=========================================================
Định nghĩa (Voogt & Oke 2003, đề cương mục 4):
    SUHII = LST trung bình khu vực đô thị − LST trung bình khu vực nền/nông thôn
            tham chiếu, TRONG CÙNG 1 thời điểm.
"Đô thị"/"nền" xác định bằng classify_surface (NDVI/NDBI/NDWI, dùng chung với 08):
"Xây dựng" = đô thị; "Thực vật" = nền tham chiếu.

CẬP NHẬT 2026-10-19 (review) — SỬA 4 LỖI của bản cũ:

  1. SAI NGÀY: bản cũ nhóm theo date_t nhưng lấy LST_target — mà LST_target là LST
     ở ngày t+delta_t (1–71 ngày sau), không phải ngày t. SUHII gắn nhãn ngày bị
     lệch cả tuần/tháng. Giờ dùng LST tại đúng ngày quan sát t (cột LST_t) cùng
     chỉ số NDVI/NDBI/NDWI tại t.
  2. THIÊN LỆCH TRỌNG SỐ: khung điểm lấy số điểm BẰNG NHAU mỗi quận dù diện tích
     chênh ~180 lần -> trung bình không trọng số coi 4 km² (Quận 3) nặng bằng 718
     km² (Cần Giờ). Giờ trung bình có TRỌNG SỐ DIỆN TÍCH: mỗi điểm đại diện
     district_area_km2 / (số điểm hợp lệ của quận đó trong ngày).
  3. THAM CHIẾU CÓ MẶT NƯỚC: bản cũ gộp "Mặt nước" vào nền tham chiếu — nước mát
     hơn hẳn đất nên kéo SUHII lên (thực hành phổ biến loại mặt nước khỏi nền).
     Giờ SUHII chính (cột `suhii`) = đô thị − thực vật; vẫn xuất `suhii_incl_water`
     (nền = thực vật + mặt nước) để đối chiếu/chọn theo cách bạn muốn trình bày.
  4. CHỈ TRÊN 2025-2026: bản cũ chỉ tính trên test_temporal. SUHII là phân tích
     MÔ TẢ cho cả giai đoạn 2019–2026 nên giờ dùng toàn bộ quan sát (nguồn mặc định:
     cache LST + đặc trưng của 04 -> nhiều điểm hơn, không phụ thuộc điều kiện
     'ngày kế tiếp cũng hợp lệ'; hoặc raw_dataset.csv nếu không có cache).

Ghi vào báo cáo: phân loại bề mặt bằng ngưỡng đơn giản NDVI/NDBI/NDWI (đất trống
có thể bị xếp 'Xây dựng'); SUHII chỉ dựa trên điểm lấy mẫu chứ không phải toàn
raster; ngày thiếu hẳn 1 trong 2 nhóm hoặc dưới --min_n điểm/nhóm bị bỏ (không
nội suy — nhất quán với đồ án).

⚠️ SUHII là PHÂN TÍCH MÔ TẢ bổ sung, KHÔNG dùng để kết luận về "rủi ro nhiệt"
(đề cương yêu cầu tách bạch LST / UHI / rủi ro nhiệt).

Đầu ra: suhii_by_date.csv (theo ngày), suhii_monthly.csv (trung bình tháng — để vẽ
biểu đồ WebGIS Tuần 13).

Cách chạy:
    python 07_compute_suhii.py
    python 07_compute_suhii.py --source raw --min_n 20
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prepare_features", os.path.join(_THIS_DIR, "03_prepare_features.py")
)
prepare_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_features)

OUT_COLS = [
    "date_t", "season", "lst_urban_mean", "lst_rural_mean", "suhii",
    "lst_rural_incl_water_mean", "suhii_incl_water", "n_urban", "n_veg", "n_water", "n_districts_urban",
]


def load_observations(data_dir: str, source: str = "auto") -> pd.DataFrame:
    """Quan sát (ngày t, điểm) với LST tại t + NDVI/NDBI/NDWI tại t + quận/diện tích.
    Cột trả về: date_t, point_id, LST, NDVI, NDBI, NDWI, district_idx, ADM2_NAME, district_area_km2."""
    cache_lst = os.path.join(data_dir, "cache_lst.csv")
    cache_feat = os.path.join(data_dir, "cache_feat.csv")
    frame_path = os.path.join(data_dir, "sampling_frame.csv")
    have_cache = all(os.path.exists(p) and os.path.getsize(p) > 0 for p in (cache_lst, cache_feat, frame_path))

    if source in ("auto", "cache") and have_cache:
        lst = pd.read_csv(cache_lst).drop_duplicates(["date", "point_id"], keep="last")
        feat = pd.read_csv(cache_feat, usecols=["date", "point_id", "NDVI", "NDBI", "NDWI"]).drop_duplicates(
            ["date", "point_id"], keep="last")
        frame = pd.read_csv(frame_path)[["point_id", "district_idx", "ADM2_NAME", "district_area_km2"]]
        obs = lst.merge(feat, on=["date", "point_id"], how="inner").merge(frame, on="point_id", how="left")
        obs = obs.rename(columns={"date": "date_t", "LST_C": "LST"})
        print(f"Nguồn: cache LST + đặc trưng ({len(obs)} quan sát, {obs['date_t'].nunique()} ngày).")
    else:
        if source == "cache":
            raise SystemExit("❌ Không có cache_lst.csv/cache_feat.csv/sampling_frame.csv — chạy 04 trước hoặc dùng --source raw.")
        raw_path = os.path.join(data_dir, "raw_dataset.csv")
        if not os.path.exists(raw_path):
            raise SystemExit(f"❌ Không tìm thấy {raw_path} — chạy 04_build_dataset.py trước.")
        raw = pd.read_csv(raw_path)
        need = ["date_t", "point_id", "LST_t", "NDVI", "NDBI", "NDWI", "district_idx", "ADM2_NAME", "district_area_km2"]
        missing = [c for c in need if c not in raw.columns]
        if missing:
            raise SystemExit(f"❌ raw_dataset.csv thiếu cột {missing} — file từ bản 04 CŨ, build lại bằng 04 mới.")
        obs = raw[need].drop_duplicates(["date_t", "point_id"]).rename(columns={"LST_t": "LST"})
        print(f"Nguồn: raw_dataset.csv ({len(obs)} quan sát, {obs['date_t'].nunique()} ngày).")
    return obs.dropna(subset=["LST", "NDVI", "NDBI", "NDWI", "district_area_km2"]).reset_index(drop=True)


def compute_suhii_by_date(obs: pd.DataFrame, lst_col: str = "LST", date_col: str = "date_t", min_n: int = 10) -> pd.DataFrame:
    """SUHII theo từng ngày, trung bình có TRỌNG SỐ DIỆN TÍCH.

    Mỗi điểm i của quận d trong ngày đó có trọng số w_i = diện tích quận d / số điểm
    hợp lệ của quận d trong ngày — (ước lượng hậu phân tầng) mỗi quận đóng góp theo
    diện tích thật của nó thay vì theo số điểm lấy mẫu. Ngày thiếu hẳn nhóm đô thị
    hoặc nhóm thực vật, hoặc mỗi nhóm < min_n điểm, bị bỏ qua (không nội suy)."""
    df = obs.copy()
    df["surface_type"] = prepare_features.classify_surface(df)
    cnt = df.groupby([date_col, "district_idx"])[lst_col].transform("count")
    df["w"] = df["district_area_km2"] / cnt
    df["wl"] = df["w"] * df[lst_col]

    agg = df.groupby([date_col, "surface_type"]).agg(
        sw=("w", "sum"), swl=("wl", "sum"), n=(lst_col, "size"), nd=("district_idx", "nunique")
    ).reset_index()

    def piv(col):
        return agg.pivot(index=date_col, columns="surface_type", values=col)

    sw, swl, n, nd = piv("sw"), piv("swl"), piv("n"), piv("nd")
    for t in (sw, swl, n, nd):
        for c in ["Xây dựng", "Thực vật", "Mặt nước"]:
            if c not in t.columns:
                t[c] = np.nan

    out = pd.DataFrame(index=sw.index)
    out["lst_urban_mean"] = swl["Xây dựng"] / sw["Xây dựng"]
    out["lst_rural_mean"] = swl["Thực vật"] / sw["Thực vật"]
    out["suhii"] = out["lst_urban_mean"] - out["lst_rural_mean"]
    sw_rw = sw["Thực vật"].fillna(0) + sw["Mặt nước"].fillna(0)
    swl_rw = swl["Thực vật"].fillna(0) + swl["Mặt nước"].fillna(0)
    out["lst_rural_incl_water_mean"] = swl_rw / sw_rw.replace(0, np.nan)
    out["suhii_incl_water"] = out["lst_urban_mean"] - out["lst_rural_incl_water_mean"]
    out["n_urban"] = n["Xây dựng"].fillna(0).astype(int)
    out["n_veg"] = n["Thực vật"].fillna(0).astype(int)
    out["n_water"] = n["Mặt nước"].fillna(0).astype(int)
    out["n_districts_urban"] = nd["Xây dựng"].fillna(0).astype(int)

    ok = (out["n_urban"] >= min_n) & (out["n_veg"] >= min_n) & out["suhii"].notna()
    out = out[ok].reset_index().rename(columns={date_col: "date_t"})
    if out.empty:
        return pd.DataFrame(columns=OUT_COLS)
    out["season"] = prepare_features.classify_season(out["date_t"])
    return out[OUT_COLS].sort_values("date_t").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--source", choices=["auto", "cache", "raw"], default="auto")
    parser.add_argument("--min_n", type=int, default=10, help="Số điểm tối thiểu mỗi nhóm (đô thị/thực vật) trong 1 ngày")
    args = parser.parse_args()

    obs = load_observations(args.data_dir, args.source)
    surface_counts = pd.Series(prepare_features.classify_surface(obs)).value_counts().to_dict()
    print(f"Phân bố loại bề mặt trong quan sát: {surface_counts}")

    print("Đang tính SUHII theo từng ngày (LST_t thực đo, trung bình có trọng số diện tích)...")
    suhii_df = compute_suhii_by_date(obs, min_n=args.min_n)

    if suhii_df.empty:
        print("❌ Không đủ dữ liệu — thiếu hẳn nhóm 'Xây dựng' HOẶC 'Thực vật' (hoặc dưới --min_n điểm) ở MỌI ngày.")
        if surface_counts.get("Xây dựng", 0) == 0:
            print("   -> 0 điểm 'Xây dựng': các đơn vị đô thị đang thiếu mẫu. Chạy diagnose_dataset_balance.py "
                  "và diagnose_mask_components.py trước khi tin vào SUHII/ablation theo bề mặt.")
        return

    out_path = os.path.join(args.data_dir, "suhii_by_date.csv")
    suhii_df.to_csv(out_path, index=False)

    monthly = suhii_df.assign(month=pd.to_datetime(suhii_df["date_t"]).dt.to_period("M").astype(str)).groupby("month").agg(
        suhii_mean=("suhii", "mean"), suhii_incl_water_mean=("suhii_incl_water", "mean"),
        lst_urban_mean=("lst_urban_mean", "mean"), lst_rural_mean=("lst_rural_mean", "mean"), n_dates=("date_t", "size"),
    ).reset_index()
    monthly.to_csv(os.path.join(args.data_dir, "suhii_monthly.csv"), index=False)

    print("\n" + "=" * 78)
    print(" KẾT QUẢ SUHII (Surface Urban Heat Island Intensity) — đô thị − thực vật")
    print("=" * 78)
    print(f"Số ngày tính được: {len(suhii_df)} / {obs['date_t'].nunique()} ngày có quan sát")
    print(f"SUHII trung bình toàn giai đoạn: {suhii_df['suhii'].mean():.2f} °C (độ lệch chuẩn {suhii_df['suhii'].std():.2f})")
    idx_max, idx_min = suhii_df["suhii"].idxmax(), suhii_df["suhii"].idxmin()
    print(f"SUHII lớn nhất:  {suhii_df.loc[idx_max, 'suhii']:.2f} °C (ngày {suhii_df.loc[idx_max, 'date_t']})")
    print(f"SUHII nhỏ nhất:  {suhii_df.loc[idx_min, 'suhii']:.2f} °C (ngày {suhii_df.loc[idx_min, 'date_t']})")
    print(f"Nếu tính nền gồm cả mặt nước: trung bình {suhii_df['suhii_incl_water'].mean():.2f} °C")
    print("\nTheo mùa:")
    print(suhii_df.groupby("season")[["suhii", "suhii_incl_water"]].agg(["mean", "count"]).round(2).to_string())
    print(f"\n✅ Đã lưu: {out_path} và suhii_monthly.csv")
    print("   -> Dùng để vẽ biểu đồ SUHII theo thời gian trên WebGIS (Tuần 13); 09_export_to_postgis.py sẽ đẩy bảng suhii_by_date.")
    print("\n⚠️  SUHII là PHÂN TÍCH MÔ TẢ bổ sung — KHÔNG dùng để kết luận về 'rủi ro nhiệt'.")


if __name__ == "__main__":
    main()
