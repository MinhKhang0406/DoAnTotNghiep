"""
SINH DỮ LIỆU GIẢ (có cấu trúc giống dataset thật) để chạy thử nhóm script 11-16 mà KHÔNG cần GEE
====================================================================================================
Tạo 4 file đúng schema của 04_build_dataset.py (SPLIT_KEEP_COLS) trong thư mục --out:
    train.csv | val.csv | test_temporal.csv | test_spatial.csv

Cấu trúc giả được thiết kế để các kiểm tra có ý nghĩa:
  * 15 đơn vị (3 tỉnh cũ), đô thị lõi nhỏ / nông thôn lớn; 3 đơn vị nông thôn giữ làm Test không gian
    (Can Gio, Cu Chi, Dau Tieng) đúng như mặc định của 04;
  * LST = mùa vụ + hiệu ứng đô thị + hiệu ứng NDVI + dị thường thời tiết toàn vùng AR(1) + nhiễu;
  * ngày Landsat thưa hơn vào mùa mưa -> delta_t DÀI hơn vào mùa mưa (delta_t lẫn với mùa, giống dữ liệu thật);
  * chia tập theo đúng quy tắc split_dataset() của 04 (Train nhãn <= 2023 | Val = 2024 | Test t >= 2025).

⚠️ Số liệu mô hình chạy trên dữ liệu này KHÔNG có ý nghĩa khoa học — chỉ để kiểm tra luồng script.
Mặc định ghi vào ../data_fake (không đụng ../data thật). Chạy script khác trên dữ liệu giả bằng --data_dir.

Cách chạy (từ thư mục scripts/, cùng chỗ với 03_prepare_features.py):
    python ../tests/make_fake_data.py --out ../data_fake
    python 11_improved_models.py --data_dir ../data_fake --models xgb,lgbm
"""
import argparse
import os

import numpy as np
import pandas as pd

DISTRICTS = [  # (ADM1, ADM2, diện tích km²)
    ("Ho Chi Minh", "Quan 1", 4.9), ("Ho Chi Minh", "Quan 3", 4.9), ("Ho Chi Minh", "Tan Binh", 22.4),
    ("Ho Chi Minh", "Go Vap", 19.7), ("Ho Chi Minh", "Binh Thanh", 20.8), ("Ho Chi Minh", "Thu Duc", 211.0),
    ("Ho Chi Minh", "Hoc Mon", 109.0), ("Ho Chi Minh", "Binh Chanh", 252.0), ("Ho Chi Minh", "Nha Be", 100.0),
    ("Ho Chi Minh", "Can Gio", 704.0), ("Ho Chi Minh", "Cu Chi", 435.0), ("Binh Duong", "Thu Dau Mot", 118.0),
    ("Binh Duong", "Di An", 60.0), ("Binh Duong", "Dau Tieng", 720.0), ("Ba Ria - Vung Tau", "Vung Tau", 150.0),
]
SPATIAL = {"Can Gio", "Cu Chi", "Dau Tieng"}
TRAIN_END_YEAR, VAL_YEAR, TEST_START_YEAR = 2023, 2024, 2025

SPLIT_KEEP_COLS = (
    ["point_id", "lon", "lat", "date_t", "date_target", "ADM1_NAME", "ADM2_NAME", "district_area_km2"]
    + ["LST_t", "NDVI", "NDBI", "NDWI", "Temp_2m", "Precip", "Humidity", "Wind", "Solar", "delta_t"]
    + ["LST_target"] + ["era5_dt_min", "s2_n_img", "st_unc_t", "st_unc_target"]
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../data_fake")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pts", type=int, default=40, help="Số điểm mỗi đơn vị")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # ---------- điểm
    pts = []
    pid = 0
    for k, (a1, a2, area) in enumerate(DISTRICTS):
        u = float(np.clip(1 - np.log(area / 4.0) / np.log(720 / 4.0), 0, 1) * 0.9 + 0.05)
        cx, cy = rng.uniform(106.4, 107.2), rng.uniform(10.3, 11.3)
        for _ in range(args.pts):
            ndvi0 = float(np.clip(0.75 - 0.55 * u + rng.normal(0, 0.08), 0.05, 0.9))
            pts.append((pid, cx + rng.normal(0, 0.015), cy + rng.normal(0, 0.015), a1, a2, area, u, ndvi0, rng.normal(0, 0.6), k))
            pid += 1
    P = pd.DataFrame(pts, columns=["point_id", "lon", "lat", "ADM1_NAME", "ADM2_NAME", "district_area_km2", "u", "ndvi0", "peff", "k"])
    npts = len(P)

    # ---------- ngày Landsat (thưa hơn ở mùa mưa => delta_t dài hơn vào mùa mưa)
    end = pd.Timestamp("2026-09-20")
    d, days = pd.Timestamp("2019-01-02"), []
    while d <= end:
        rainy = d.month in range(5, 12)
        if rng.random() < (0.35 if rainy else 0.72):
            days.append(d)
            if rng.random() < 0.12 and d + pd.Timedelta(days=1) <= end:
                days.append(d + pd.Timedelta(days=1))
        d += pd.Timedelta(days=8)
    days = sorted(set(days))
    nd = len(days)

    # ---------- dị thường thời tiết toàn vùng AR(1) theo ngày lịch
    all_days = pd.date_range("2019-01-01", "2026-12-31")
    A = np.zeros(len(all_days))
    for i in range(1, len(all_days)):
        A[i] = 0.8 * A[i - 1] + rng.normal(0, 0.9)
    A += (all_days.year.to_numpy() - 2019) * 0.15
    A_by = pd.Series(A, index=all_days)

    def season(doy, rainy):
        return 33.0 + 3.2 * np.sin(2 * np.pi * (doy - 15) / 365.0) - np.where(rainy, 1.0, 0.0)

    # ---------- ma trận theo (ngày, điểm)
    doy = np.array([x.dayofyear for x in days])
    rainy = np.array([x.month in range(5, 12) for x in days])
    a_vec = A_by.reindex(days).to_numpy()
    ndvi = np.clip(P["ndvi0"].to_numpy()[None, :] + 0.06 * np.where(rainy, 1, -1)[:, None] + rng.normal(0, 0.03, (nd, npts)), -0.05, 0.95)
    u = P["u"].to_numpy()[None, :]
    ndbi = -0.2 + 0.55 * u - 0.25 * (ndvi - 0.45) + rng.normal(0, 0.04, (nd, npts))
    ndwi = -0.35 * ndvi + rng.normal(0, 0.04, (nd, npts))
    lst = (season(doy, rainy)[:, None] + 5.0 * u - 7.0 * (ndvi - 0.45) + a_vec[:, None]
           + P["peff"].to_numpy()[None, :] + rng.normal(0, 0.7, (nd, npts)))

    # ERA5 (thô theo ngày, nhiễu nhỏ theo đơn vị)
    dist_noise = rng.normal(0, 0.15, (nd, len(DISTRICTS)))
    dk = P["k"].to_numpy()
    temp2m = (26.5 + 2.0 * np.sin(2 * np.pi * (doy - 15) / 365.0) - np.where(rainy, 0.8, 0.0) + 0.6 * a_vec)[:, None] + dist_noise[:, dk]
    precip = np.where(rainy[:, None] & (rng.random((nd, npts)) < 0.4), rng.exponential(0.6, (nd, npts)), rng.random((nd, npts)) * 0.02)
    humid = np.clip(62 + 22 * rainy[:, None] - 2 * a_vec[:, None] + rng.normal(0, 4, (nd, 1)) + rng.normal(0, 0.5, (nd, npts)), 20, 100)
    wind = np.abs(2.5 + rng.normal(0, 0.8, (nd, 1)) + rng.normal(0, 0.1, (nd, npts)))
    solar = 600 - 150 * rainy[:, None] + 20 * a_vec[:, None] + rng.normal(0, 60, (nd, 1)) + rng.normal(0, 5, (nd, npts))

    valid = rng.random((nd, npts)) < 0.85

    # ---------- ghép cặp (t, ngày hợp lệ kế tiếp)
    chunks = []
    ptid = P["point_id"].to_numpy()
    for i in range(nd - 1):
        j = i + 1
        m = valid[i] & valid[j]
        if not m.any():
            continue
        n = int(m.sum())
        chunks.append(pd.DataFrame({
            "point_id": ptid[m], "lon": P["lon"].to_numpy()[m].round(7), "lat": P["lat"].to_numpy()[m].round(7),
            "date_t": days[i].strftime("%Y-%m-%d"), "date_target": days[j].strftime("%Y-%m-%d"),
            "ADM1_NAME": P["ADM1_NAME"].to_numpy()[m], "ADM2_NAME": P["ADM2_NAME"].to_numpy()[m],
            "district_area_km2": P["district_area_km2"].to_numpy()[m],
            "LST_t": lst[i][m], "NDVI": ndvi[i][m], "NDBI": ndbi[i][m], "NDWI": ndwi[i][m],
            "Temp_2m": temp2m[i][m], "Precip": precip[i][m], "Humidity": humid[i][m], "Wind": wind[i][m], "Solar": solar[i][m],
            "delta_t": float((days[j] - days[i]).days), "LST_target": lst[j][m],
            "era5_dt_min": rng.uniform(0, 30, n), "s2_n_img": rng.integers(1, 4, n).astype(float),
            "st_unc_t": rng.uniform(0.5, 2.9, n), "st_unc_target": rng.uniform(0.5, 2.9, n),
        }))
    raw = pd.concat(chunks, ignore_index=True)[SPLIT_KEEP_COLS]

    # ---------- chia tập đúng quy tắc 04
    d_t, d_n = pd.to_datetime(raw["date_t"]), pd.to_datetime(raw["date_target"])
    is_sp = raw["ADM2_NAME"].isin(SPATIAL)
    spatial, temporal = raw[is_sp], raw[~is_sp]
    dt_t, dt_n = d_t[~is_sp], d_n[~is_sp]
    m_train = dt_n.dt.year <= TRAIN_END_YEAR
    m_val = (dt_t.dt.year == VAL_YEAR) & (dt_n.dt.year == VAL_YEAR)
    m_test = dt_t.dt.year >= TEST_START_YEAR
    temporal[m_train].to_csv(os.path.join(args.out, "train.csv"), index=False)
    temporal[m_val].to_csv(os.path.join(args.out, "val.csv"), index=False)
    temporal[m_test].to_csv(os.path.join(args.out, "test_temporal.csv"), index=False)
    spatial.to_csv(os.path.join(args.out, "test_spatial.csv"), index=False)
    print(f"✅ Dữ liệu GIẢ ({nd} ngày, {npts} điểm, {len(raw):,} dòng) -> {args.out}")
    print(f"   train={int(m_train.sum()):,} | val={int(m_val.sum()):,} | test_temporal={int(m_test.sum()):,} | test_spatial={len(spatial):,}")
    dts = raw.drop_duplicates("date_t")["delta_t"]
    print(f"   delta_t: min={dts.min():.0f} trung vị={dts.median():.0f} max={dts.max():.0f}")


if __name__ == "__main__":
    main()
