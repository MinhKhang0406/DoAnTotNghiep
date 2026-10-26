"""
BƯỚC 10 — LƯỚI DỰ BÁO PHỦ TOÀN VÙNG + TRẠNG THÁI "KHÔNG CÓ DỮ LIỆU" (cho WebGIS)
=================================================================================
LUỒNG DỮ LIỆU CÒN THIẾU của bản cũ (phát hiện khi review): 09 chỉ đẩy dự báo trên
các điểm lấy mẫu của tập Test (~vài trăm điểm/ngày, nằm rải rác), trong khi đề
cương yêu cầu (mục "Phương pháp bản đồ hóa", Tuần 12):
  - bản đồ nhiệt phân cấp màu phủ toàn khu vực;
  - vùng không có ảnh hợp lệ (mây che hoàn toàn / ngoài dải quét) hiển thị "không
    có dữ liệu" thay vì nội suy.
Không script nào tạo được cái đó. Script này làm đúng việc đó:

  1. Tạo lưới điểm đều (mặc định 1 km) phủ toàn bộ ranh giới, gán quận/huyện.
  2. Với ngày quan sát t (mặc định: ngày hợp lệ MỚI NHẤT trong cache của 04, hoặc --date):
     lấy LST_t (cùng bộ mặt nạ đã dùng khi huấn luyện, đọc từ cache_manifest.json),
     NDVI/NDBI/NDWI (S2, chỉ nhìn về trước t), ERA5 tại t — đúng định nghĩa feature.
  3. Với mỗi delta_t yêu cầu (--delta_t 1,7,14...): dự báo LST tại ngày t+delta_t bằng
     ĐÚNG model đã tune ở 06 (best_model_*.pkl, không train lại).
     (Khi dự báo thật, delta_t là THAM SỐ do người dùng chọn: "dự báo cho kỳ quan sát
     cách t bao nhiêu ngày" — đúng ý nghĩa feature delta_t trong đề cương.)
  4. Điểm thiếu LST_t / S2 / ERA5 KHÔNG được dự báo: has_data=False + reason
     (no_lst_t | no_s2 | no_era5) -> frontend vẽ "không có dữ liệu".
  5. Ghi ../data/forecast_grid_<t>.csv (định dạng dài: mỗi (điểm, delta_t) 1 dòng, có
     LST_t, NDVI, NDBI, NDWI, ERA5, surface_type nên frontend dùng luôn làm layer
     NDVI/NDBI). 09_export_to_postgis.py đẩy vào bảng lst_forecast_grid.

Cách chạy:
    python 10_build_inference_grid.py --project YOUR_ID                       # ngày hợp lệ mới nhất, delta_t=8
    python 10_build_inference_grid.py --project YOUR_ID --date 2026-06-12 --delta_t 1,7,14 --grid_m 500
    python 10_build_inference_grid.py --project YOUR_ID --model rf

Lưu ý: lưới 500 m ≈ 27 nghìn điểm (~14 lần gọi GEE 2000 điểm) — vẫn nhẹ; hạ xuống 250 m
nếu cần mịn hơn. Muốn chạy định kỳ (cập nhật bản đồ mỗi khi có ảnh mới): lặp lại
04 --stage lst/features (chỉ xử lý ngày mới nhờ cache) -> 10 -> 09 --tables grid.
"""

import argparse
import importlib.util
import json
import os
import pickle
from datetime import datetime, timedelta

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

CRS_UTM = prepare_features.CRS_UTM
ERA5_VARS = prepare_features.ERA5_VARS
GRID_BANDS = ["LST_t", "NDVI", "NDBI", "NDWI", "s2_n_img"] + ERA5_VARS + ["era5_dt_min"]


def build_grid_points(boundary_path: str, grid_m: float) -> pd.DataFrame:
    import geopandas as gpd

    districts = prepare_features.load_districts(boundary_path).to_crs(CRS_UTM)
    minx, miny, maxx, maxy = districts.total_bounds
    xs = np.arange(minx + grid_m / 2, maxx, grid_m)
    ys = np.arange(miny + grid_m / 2, maxy, grid_m)
    gx, gy = np.meshgrid(xs, ys)
    pts = gpd.GeoDataFrame(
        {"x": gx.ravel(), "y": gy.ravel()}, geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=CRS_UTM
    )
    j = gpd.sjoin(pts, districts[["district_idx", "ADM1_NAME", "ADM2_NAME", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated(keep="first")]  # điểm vùng chồng lấn: giữ 1 đơn vị
    ll = j.to_crs("EPSG:4326")
    out = pd.DataFrame({
        "lon": ll.geometry.x.values.round(6), "lat": ll.geometry.y.values.round(6),
        "district_idx": j["district_idx"].values, "ADM1_NAME": j["ADM1_NAME"].values, "ADM2_NAME": j["ADM2_NAME"].values,
    }).reset_index(drop=True)
    out.insert(0, "point_id", np.arange(len(out)))
    return out


def _manifest_params(data_dir: str) -> dict:
    """Đọc tham số mặt nạ/S2 đã dùng khi dựng dataset để lưới dự báo dùng ĐÚNG cùng cấu hình."""
    mp = os.path.join(data_dir, "cache_manifest.json")
    if not os.path.exists(mp):
        print("   ⚠️  Không có cache_manifest.json — dùng tham số mặc định của 03 (có thể lệch cấu hình lúc huấn luyện).")
        return {}
    with open(mp, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--date", default=None, help="Ngày quan sát t (YYYY-MM-DD). Mặc định: ngày hợp lệ mới nhất trong cache.")
    parser.add_argument("--delta_t", default="8", help="Các delta_t (ngày) cần dự báo, phân tách bởi dấu phẩy, vd 1,7,14")
    parser.add_argument("--grid_m", type=float, default=1000.0, help="Khoảng cách lưới điểm (m)")
    parser.add_argument("--model", choices=["xgb", "rf"], default="xgb")
    parser.add_argument("--min_valid_frac", type=float, default=0.10, help="Độ phủ tối thiểu khi tự chọn ngày mới nhất")
    args = parser.parse_args()

    deltas = [float(x) for x in args.delta_t.split(",") if x.strip()]
    model_path = os.path.join(args.data_dir, f"best_model_{args.model}.pkl")
    if not os.path.exists(model_path):
        raise SystemExit(f"❌ Không thấy {model_path} — chạy 06_train_ml_models.py trước.")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model, features = saved["model"], saved["features"]
    print(f"Model '{args.model}' — {len(features)} feature: {features}")

    train_path = os.path.join(args.data_dir, "train.csv")
    if os.path.exists(train_path):
        tdt = pd.read_csv(train_path, usecols=["delta_t"])["delta_t"]
        lo, hi = float(tdt.min()), float(tdt.max())
        out_of = [d for d in deltas if d < lo or d > hi]
        if out_of:
            print(f"   ⚠️  delta_t {out_of} nằm NGOÀI khoảng đã học ({lo:.0f}–{hi:.0f} ngày) — mô hình phải ngoại suy, độ tin cậy thấp.")

    import ee

    prepare_features.init_ee(args.project)
    roi = prepare_features.load_boundary_roi(args.boundary)
    manifest = _manifest_params(args.data_dir)
    lst_p = manifest.get("lst", {})
    mask_kwargs = dict(
        zscore_threshold=lst_p.get("zscore_threshold", 3.0), focal_radius_px=lst_p.get("focal_radius_px", 2),
        apply_zscore=True, max_st_unc_k=lst_p.get("max_st_unc_k", prepare_features.DEFAULT_MAX_ST_UNC_K),
    )
    s2_back = manifest.get("feat", {}).get("s2_back_days", prepare_features.S2_BACK_DAYS)

    era5_max = prepare_features.get_collection_max_date(prepare_features.ERA5_COLLECTION)
    print(f"ERA5-Land trên GEE có dữ liệu tới: {era5_max}")

    # --- chọn ngày t và giờ chụp thật
    frame_path = os.path.join(args.data_dir, "sampling_frame.csv")
    t_str, t_ms = args.date, None
    have_cache = os.path.exists(os.path.join(args.data_dir, build_dataset.LST_DATES)) and os.path.exists(frame_path)
    if have_cache:
        dates, lst = build_dataset.load_lst_cache(args.data_dir)
        tbl = build_dataset.valid_dates_table(dates, lst, pd.read_csv(frame_path), args.min_valid_frac)
        if t_str is None:
            cand = tbl[tbl["valid"]]
            if era5_max:
                cand = cand[cand["date"] <= era5_max]
            if cand.empty:
                raise SystemExit("❌ Không có ngày hợp lệ nào (có ERA5) trong cache — chạy 04 hoặc truyền --date.")
            t_str, t_ms = cand.iloc[-1]["date"], int(cand.iloc[-1]["t_utc_ms"])
            print(f"Tự chọn ngày hợp lệ mới nhất: {t_str} (độ phủ {cand.iloc[-1]['coverage'] * 100:.1f}%)")
        else:
            row = tbl[tbl["date"] == t_str]
            if len(row):
                t_ms = int(row.iloc[0]["t_utc_ms"])
    if t_str is None:
        raise SystemExit("❌ Chưa có cache của 04 — hãy truyền --date YYYY-MM-DD.")
    if t_ms is None:
        d1 = (datetime.strptime(t_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        st = prepare_features.list_landsat_scene_times(roi, t_str, d1, chunk_months=1, verbose=False)
        if t_str not in st:
            raise SystemExit(f"❌ Không có cảnh Landsat 8/9 nào chạm ROI ngày {t_str}.")
        t_ms = st[t_str]
    if era5_max and t_str > era5_max:
        print(f"   ⚠️  Ngày {t_str} > {era5_max}: ERA5 chưa có -> TOÀN BỘ lưới sẽ là 'no_era5'.")

    # --- lưới điểm + lấy đặc trưng tại t
    grid = build_grid_points(args.boundary, args.grid_m)
    print(f"Lưới {args.grid_m:.0f} m: {len(grid)} điểm / {grid['district_idx'].nunique()} đơn vị.")

    lst_img = build_dataset._lst_image_for_date(roi, t_str, mask_kwargs).select("LST_C").rename("LST_t")
    s2 = prepare_features.get_sentinel2_features(roi, t_str, s2_back, 0)  # không nhìn sau t
    era5 = prepare_features.get_era5_features(roi, ee.Date(int(t_ms)))
    img = ee.Image.cat([lst_img, s2, era5])

    print("Đang lấy đặc trưng tại các điểm lưới từ GEE...")
    vals = prepare_features.reduce_at_points(img, grid, GRID_BANDS, label=t_str)
    df = grid.merge(vals, on="point_id", how="left")

    no_lst = df["LST_t"].isna()
    no_s2 = df[["NDVI", "NDBI", "NDWI"]].isna().any(axis=1)
    no_era5 = df[ERA5_VARS].isna().any(axis=1) | (df["era5_dt_min"] < 0) | (df["era5_dt_min"] > 60.0 + 1e-6)
    df["reason"] = np.select([no_lst, no_s2, no_era5], ["no_lst_t", "no_s2", "no_era5"], default="ok")
    df["has_data"] = df["reason"] == "ok"
    df["surface_type"] = "Không có dữ liệu"
    ok = df["has_data"]
    if ok.any():
        df.loc[ok, "surface_type"] = prepare_features.classify_surface(df.loc[ok])

    print(f"\nĐộ phủ dữ liệu ngày {t_str}: {ok.sum()}/{len(df)} điểm có dữ liệu ({ok.mean() * 100:.1f}%)")
    print("   Lý do không có dữ liệu:", df.loc[~ok, "reason"].value_counts().to_dict())

    # --- dự báo cho từng delta_t
    base_cols = ["point_id", "lon", "lat", "ADM1_NAME", "ADM2_NAME", "LST_t", "NDVI", "NDBI", "NDWI"] + ERA5_VARS + [
        "has_data", "reason", "surface_type"]
    t_date = datetime.strptime(t_str, "%Y-%m-%d")
    parts = []
    for dlt in deltas:
        part = df[base_cols].copy()
        part["date_t"] = t_str
        part["delta_t"] = dlt
        part["date_forecast"] = (t_date + timedelta(days=int(round(dlt)))).strftime("%Y-%m-%d")
        part["LST_pred"] = np.nan
        part["model"] = args.model
        if ok.any():
            X = df.loc[ok].copy()
            X["delta_t"] = dlt
            part.loc[ok.values, "LST_pred"] = model.predict(X[features])
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)

    out_path = os.path.join(args.data_dir, f"forecast_grid_{t_str}.csv")
    out.to_csv(out_path, index=False)

    print("\nLST dự báo trung bình theo delta_t (°C):")
    print(out[out["has_data"]].groupby("delta_t")["LST_pred"].agg(["mean", "min", "max", "count"]).round(2).to_string())
    print(f"\n✅ Đã lưu {out_path} ({len(out)} dòng). Đẩy vào CSDL: python 09_export_to_postgis.py --tables grid")
    print("   Frontend: vẽ điểm has_data=True theo LST_pred (phân cấp màu); has_data=False hiển thị 'không có dữ liệu' (kèm reason).")


if __name__ == "__main__":
    main()
