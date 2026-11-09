"""
SINH forecast_grid_<t>.csv GIẢ đúng định dạng dài của 10_build_inference_grid.py (để thử 10b/16 mà không cần GEE)
=====================================================================================================================
Lấy các điểm của test_temporal trong một ngày t làm 'lưới', nhân lên theo các delta_t, thêm điểm has_data=False
(reason no_lst_t / no_s2 / no_era5) và cột LST_pred (giả lập dự báo của 06). Số liệu KHÔNG có ý nghĩa khoa học.

    python ../tests/make_fake_grid.py --data_dir ../data --deltas 1,7,14,60
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--deltas", default="1,7,14,60")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(os.path.join(args.data_dir, "test_temporal.csv"))
    t = sorted(df["date_t"].unique())[len(df["date_t"].unique()) // 2]
    g = df[df["date_t"] == t].drop_duplicates("point_id").reset_index(drop=True)
    keep = ["point_id", "lon", "lat", "ADM1_NAME", "ADM2_NAME", "LST_t", "NDVI", "NDBI", "NDWI", "Temp_2m", "Precip", "Humidity", "Wind", "Solar"]
    g = g[keep].copy()
    # ~12% điểm không có dữ liệu
    bad = rng.random(len(g)) < 0.12
    reason = np.where(bad, rng.choice(["no_lst_t", "no_s2", "no_era5"], len(g)), "ok")
    g.loc[reason == "no_lst_t", "LST_t"] = np.nan
    g.loc[reason == "no_s2", ["NDVI", "NDBI", "NDWI"]] = np.nan
    g.loc[reason == "no_era5", ["Temp_2m", "Precip", "Humidity", "Wind", "Solar"]] = np.nan
    g["reason"] = reason
    g["has_data"] = reason == "ok"
    g["surface_type"] = np.where(g["has_data"], "Thực vật", "Không có dữ liệu")

    parts = []
    for d in [float(x) for x in args.deltas.split(",")]:
        p = g.copy()
        p["date_t"] = t
        p["delta_t"] = d
        p["date_forecast"] = (pd.Timestamp(t) + pd.Timedelta(days=int(round(d)))).strftime("%Y-%m-%d")
        p["LST_pred"] = np.where(p["has_data"], p["LST_t"] + rng.normal(0, 0.8, len(p)), np.nan)
        p["model"] = "xgb"
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    path = os.path.join(args.data_dir, f"forecast_grid_{t}.csv")
    out.to_csv(path, index=False)
    print(f"✅ Lưới GIẢ {path}: {len(out):,} dòng ({len(g)} điểm × {len(parts)} delta_t), has_data={out['has_data'].mean():.0%}")


if __name__ == "__main__":
    main()
