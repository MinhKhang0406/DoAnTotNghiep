"""
BƯỚC 16 — ĐẨY KẾT QUẢ NÂNG CẤP (11–15, 10b) VÀO POSTGIS — bổ sung cho 09, KHÔNG đụng vào các bảng của 09
==========================================================================================================
Dùng lại hàm của 09_export_to_postgis.py (chuỗi kết nối LST_DB_URL, chọn driver, ép kiểu DATE, tạo chỉ mục) nên cách chạy
giống hệt 09. Mọi bảng mới có hậu tố/tên riêng để KHÔNG ghi đè bảng cũ (lst_predictions, lst_forecast_grid, model_metrics...):

  --tables      bảng được tạo (bảng nào thiếu file nguồn thì bỏ qua, có cảnh báo)           file nguồn
  predictions   lst_predictions_v2 (điểm)   dự báo + khoảng tin cậy (conformal & cqr) +     interval_predictions.csv (13)
                                            error, delta_t_bin, delta_t_out_of_range
  grid          lst_forecast_grid_v2 (điểm) lưới dự báo v2 + LST_lo/LST_hi                  forecast_grid_v2_*.csv (10b)
  metrics       model_metrics_v2            baseline mạnh + 4 họ mô hình × variant × 3 kịch bản   metrics_improved.csv (11)
  tuning        model_tuning                XGBoost mặc định vs đã dò                        metrics_tuned.csv (12)
  ci            model_rmse_ci               RMSE/MAE/skill + CI bootstrap theo ngày          bootstrap_ci.csv (14)
  pairs         model_paired_tests         hiệu RMSE từng cặp + p Holm                      paired_tests.csv (14)
  delta         error_by_delta_t, delta_t_partial_dependence                               error_by_delta_t.csv, partial_dependence_delta_t.csv (13)
  intervals     interval_eval              độ phủ/độ rộng khoảng theo kịch bản × nhóm delta_t      interval_eval.csv (13)
  calib         interval_calibration       phân vị hiệu chuẩn theo nhóm delta_t              interval_calibration.json (13)
  diag          delta_t_diagnostic         kiểm tra delta_t có là proxy cho mùa không (1 dòng)   delta_t_diagnostic.json (13)
  spatialcv     spatial_cv_district, spatial_cv_summary   leave-district-out                  spatial_cv_*.csv (15)
  moran         moran_residuals            Moran's I phần dư                                moran_residuals.csv (14)
  residmap      residual_map_mean (điểm)   phần dư trung bình theo điểm                     residual_map_mean.csv (14)

Bảng điểm (predictions, grid, residmap) có cột hình học (EPSG:4326) + chỉ mục GiST; cột ngày được ép kiểu DATE + chỉ mục B-tree.
Nếu chuỗi kết nối KHÔNG phải PostgreSQL (vd sqlite:///thu.db — chỉ để thử nhanh), bảng điểm được ghi thành bảng thường (lon/lat, không hình học).

Cách chạy:
    set LST_DB_URL=postgresql://postgres:MATKHAU@localhost:5432/postgres      (Windows cmd)
    python 16_export_to_postgis_v2.py
    python 16_export_to_postgis_v2.py --tables predictions,grid,ci
    python 16_export_to_postgis_v2.py --dry_run            # không cần DB, chỉ kiểm tra dữ liệu sẽ đẩy
"""

import argparse
import glob
import importlib.util
import json
import os
import sys

import geopandas as gpd
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_THIS_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export09 = _load("export09", "09_export_to_postgis.py")

ALL_TABLES = ["predictions", "grid", "metrics", "tuning", "ci", "pairs", "delta", "intervals", "calib", "diag", "spatialcv", "moran", "residmap"]
DATE_INDEX = {"lst_predictions_v2": ("date_target", "date_t"), "lst_forecast_grid_v2": ("date_forecast", "date_t")}


def _csv(data_dir: str, fname: str, label: str):
    d = export09._read(os.path.join(data_dir, fname))
    if d is None:
        print(f"⚠️  Chưa có {fname} — bỏ qua {label}.")
    return d


def _json(data_dir: str, fname: str, label: str):
    p = os.path.join(data_dir, fname)
    if not os.path.exists(p):
        print(f"⚠️  Chưa có {fname} — bỏ qua {label}.")
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _geo(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")


def build_payload(data_dir: str, want: list) -> dict:
    """tên bảng -> (DataFrame/GeoDataFrame, là_không_gian)"""
    out = {}
    calib = None
    if {"predictions", "calib"} & set(want):
        calib = _json(data_dir, "interval_calibration.json", "model_name / calib")
    if "predictions" in want:
        d = _csv(data_dir, "interval_predictions.csv", "lst_predictions_v2 (chạy 13)")
        if d is not None:
            d["error"] = d["LST_pred"] - d["LST_target"]
            d["model_v2"] = f"{calib['model_name']}|{calib['variant']}" if calib else None
            d["interval_coverage"] = calib["coverage"] if calib else None
            out["lst_predictions_v2"] = (_geo(d), True)
    if "grid" in want:
        files = sorted(glob.glob(os.path.join(data_dir, "forecast_grid_v2_*.csv")))
        if not files:
            print("⚠️  Chưa có forecast_grid_v2_*.csv — bỏ qua lst_forecast_grid_v2 (chạy 10b_add_intervals_to_grid.py).")
        else:
            out["lst_forecast_grid_v2"] = (_geo(pd.concat([pd.read_csv(f) for f in files], ignore_index=True)), True)
    simple = [("metrics", "model_metrics_v2", "metrics_improved.csv", "11"), ("tuning", "model_tuning", "metrics_tuned.csv", "12"),
              ("ci", "model_rmse_ci", "bootstrap_ci.csv", "14"), ("pairs", "model_paired_tests", "paired_tests.csv", "14"),
              ("intervals", "interval_eval", "interval_eval.csv", "13"), ("moran", "moran_residuals", "moran_residuals.csv", "14")]
    for key, name, fname, step in simple:
        if key in want:
            d = _csv(data_dir, fname, f"{name} (chạy {step})")
            if d is not None:
                out[name] = (d, False)
    if "delta" in want:
        for name, fname in (("error_by_delta_t", "error_by_delta_t.csv"), ("delta_t_partial_dependence", "partial_dependence_delta_t.csv")):
            d = _csv(data_dir, fname, f"{name} (chạy 13)")
            if d is not None:
                out[name] = (d, False)
    if "spatialcv" in want:
        for name, fname in (("spatial_cv_district", "spatial_cv_by_district.csv"), ("spatial_cv_summary", "spatial_cv_summary.csv")):
            d = _csv(data_dir, fname, f"{name} (chạy 15)")
            if d is not None:
                out[name] = (d, False)
    if "residmap" in want:
        d = _csv(data_dir, "residual_map_mean.csv", "residual_map_mean (chạy 14)")
        if d is not None:
            out["residual_map_mean"] = (_geo(d), True)
    if "calib" in want and calib is not None:
        rows = []
        tr = calib.get("train_delta_t", {})
        for method in ("conformal", "cqr"):
            if method not in calib:
                continue
            cfg = calib[method]
            meaning = "nua_do_rong_C" if method == "conformal" else "hieu_chinh_C"
            rows.append({"method": method, "delta_bin": "global", "value_celsius": cfg["global"], "n_calib": sum(cfg["n"].values()), "value_meaning": meaning})
            for lab, q in cfg["bins"].items():
                rows.append({"method": method, "delta_bin": lab, "value_celsius": q, "n_calib": cfg["n"].get(lab), "value_meaning": meaning})
        d = pd.DataFrame(rows)
        d["coverage"], d["model_file"], d["variant"], d["created"] = calib["coverage"], calib["model_file"], calib["variant"], calib["created"]
        d["train_delta_min"], d["train_delta_p99"], d["train_delta_max"] = tr.get("min"), tr.get("max"), tr.get("max_observed")
        out["interval_calibration"] = (d, False)
    if "diag" in want:
        j = _json(data_dir, "delta_t_diagnostic.json", "delta_t_diagnostic (chạy 13)")
        if j is not None:
            j = dict(j)
            j["mean_delta_t_by_month_train"] = json.dumps(j.get("mean_delta_t_by_month_train", {}), ensure_ascii=False)
            out["delta_t_diagnostic"] = (pd.DataFrame([j]), False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--tables", default="all", help=f"Danh sách, phân tách bởi dấu phẩy, hoặc 'all'. Gồm: {ALL_TABLES}")
    ap.add_argument("--db_url", default=None, help="Chuỗi kết nối SQLAlchemy; mặc định lấy biến môi trường LST_DB_URL")
    ap.add_argument("--mode", choices=["replace", "append"], default="replace")
    ap.add_argument("--dry_run", action="store_true", help="Không kết nối DB; chỉ dựng dữ liệu và in số dòng")
    args = ap.parse_args()

    want = ALL_TABLES if args.tables == "all" else [t.strip() for t in args.tables.split(",") if t.strip()]
    bad = [t for t in want if t not in ALL_TABLES]
    if bad:
        raise SystemExit(f"❌ Tên bảng không hợp lệ: {bad}. Hợp lệ: {ALL_TABLES}")

    payload = build_payload(args.data_dir, want)
    print("\nBảng sẽ đẩy:")
    for name, (df, is_geo) in payload.items():
        print(f"   {name:<28} {len(df):>9} dòng{'  (điểm, EPSG:4326)' if is_geo else ''}")
    if args.dry_run:
        print("\n(--dry_run: không kết nối CSDL.)")
        return
    if not payload:
        raise SystemExit("Không có gì để đẩy — chạy 11–15 (và 10b cho lưới) trước.")

    db_url = args.db_url or os.environ.get("LST_DB_URL")
    if not db_url:
        raise SystemExit("❌ Chưa có chuỗi kết nối CSDL. Đặt biến môi trường LST_DB_URL hoặc dùng --db_url, ví dụ:\n"
                         "   postgresql://postgres:MATKHAU@localhost:5432/postgres")

    from sqlalchemy import create_engine

    print("\nĐang kết nối Database và đẩy dữ liệu...")
    try:
        engine = create_engine(export09._resolve_db_url(db_url))
        is_pg = engine.dialect.name == "postgresql"
        if not is_pg:
            print(f"   ℹ️  Dialect '{engine.dialect.name}' không phải PostgreSQL: bảng điểm được ghi thành bảng thường (không hình học, không chỉ mục).")
        geo_tables = []
        for name, (df, is_geo) in payload.items():
            df = export09._typed_dates(df)
            if is_geo and is_pg:
                df.to_postgis(name=name, con=engine, if_exists=args.mode, index=False)
                geo_tables.append(name)
            else:
                pd.DataFrame(df.drop(columns="geometry", errors="ignore")).to_sql(name=name, con=engine, if_exists=args.mode, index=False)
            print(f"✅ Đã đẩy {len(df)} dòng vào bảng '{name}'.")
        if is_pg and geo_tables:
            date_cols = {t: list(c) for t, c in DATE_INDEX.items() if t in payload}
            export09.create_indexes(engine, geo_tables, date_cols)
            print("✅ Đã tạo chỉ mục (GiST hình học + ngày).")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"❌ LỖI KẾT NỐI/GHI DATABASE ({type(e).__name__}):", str(e)[:600])
        print("Kiểm tra lại LST_DB_URL (hoặc --db_url), PostGIS đã bật chưa (CREATE EXTENSION postgis;), "
              "và user có quyền tạo bảng trong schema public.")
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())
