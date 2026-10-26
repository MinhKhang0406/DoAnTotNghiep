"""
BƯỚC 9 — ĐẨY DỰ BÁO + RANH GIỚI + SỐ LIỆU PHÂN TÍCH VÀO POSTGIS
=======================================================================
Đề cương Tuần 10: "đẩy kết quả dự báo LST và ranh giới hành chính vào CSDL"; Tuần 12:
hiển thị "không có dữ liệu"; Tuần 13: biểu đồ chuỗi thời gian, so sánh
RMSE/MAE/R² giữa các mô hình, SUHII. Bản cũ chỉ đẩy 2 bảng (lst_predictions,
admin_boundaries) nên các số liệu còn lại KHÔNG có đường vào CSDL.

CẬP NHẬT 2026-10-19 — các bảng được đẩy (bảng nào thiếu file nguồn thì bỏ qua, có cảnh báo):
  lst_predictions    (điểm)  dự báo của model đã tune trên val / test_temporal / test_spatial,
                             kèm date_target (NGÀY ĐƯỢC DỰ BÁO — bản cũ thiếu nên frontend
                             không biết dự báo cho ngày nào), delta_t, sai số, cột split
  lst_forecast_grid  (điểm)  lưới dự báo phủ toàn vùng từ 10_build_inference_grid.py, có
                             has_data/reason để frontend vẽ "không có dữ liệu"
  admin_boundaries   (vùng)  ranh giới (tự dùng bản nhẹ hcmc_boundary_web.geojson nếu có)
  suhii_by_date, suhii_monthly   từ 07
  model_metrics              RMSE/MAE/R² từ 05 + 06 (metrics_baseline.csv + metrics_ml.csv)
  ablation_results, shap_importance   từ 08
Tạo luôn chỉ mục (GiST cho hình học, B-tree cho ngày) — to_postgis không tự tạo
chỉ mục ngày và `if_exists="replace"` làm mất chỉ mục cũ.

BẢO MẬT: bỏ chuỗi kết nối mặc định có mật khẩu ghi cứng trong code (nếu mật khẩu đó từng
được dùng thật hoặc đã lên GitHub thì nên ĐỔI MẬT KHẨU). Truyền qua biến môi trường
LST_DB_URL hoặc --db_url. Với Supabase dùng
connection string dạng postgresql://USER:PASS@HOST:5432/postgres (thêm
?sslmode=require nếu server yêu cầu).

Cách chạy:
    set LST_DB_URL=postgresql://postgres:MATKHAU@localhost:5432/postgres      (Windows cmd)
    python 09_export_to_postgis.py
    python 09_export_to_postgis.py --model rf --tables predictions,metrics
    python 09_export_to_postgis.py --dry_run        # không cần DB, chỉ kiểm tra dữ liệu sẽ đẩy
"""

import argparse
import glob
import os
import pickle

import geopandas as gpd
import pandas as pd

ALL_TABLES = ["predictions", "grid", "boundary", "suhii", "metrics", "ablation", "shap"]


def _read(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return None


def build_predictions(data_dir: str, model_key: str) -> gpd.GeoDataFrame:
    model_path = os.path.join(data_dir, f"best_model_{model_key}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không thấy {model_path} — chạy 06_train_ml_models.py trước.")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model, features = saved["model"], saved["features"]
    print(f"   Model '{model_key}' dùng {len(features)} feature: {features}")

    parts = []
    for split, fname in [("val", "val.csv"), ("test_temporal", "test_temporal.csv"), ("test_spatial", "test_spatial.csv")]:
        df = _read(os.path.join(data_dir, fname))
        if df is None or len(df) == 0:
            print(f"   ⚠️  {fname} không có/rỗng — bỏ qua split '{split}'.")
            continue
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise SystemExit(f"❌ {fname} thiếu cột {missing} so với feature model đã train — dataset và model KHÔNG khớp nhau.")
        df = df.copy()
        df["LST_pred"] = model.predict(df[features])
        df["error"] = df["LST_pred"] - df["LST_target"]
        df["split"] = split
        df["model"] = model_key
        parts.append(df)
    if not parts:
        raise SystemExit("❌ Không có split nào để dự báo.")
    out = pd.concat(parts, ignore_index=True)
    # ADM1_NAME + ADM2_NAME = khóa nối với admin_boundaries (tên ADM2 một mình có thể trùng giữa các tỉnh cũ)
    cols = ["point_id", "lon", "lat", "date_t", "date_target", "delta_t", "LST_t", "LST_target", "LST_pred", "error",
            "ADM1_NAME", "ADM2_NAME", "split", "model"]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]
    return gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out.lon, out.lat), crs="EPSG:4326")


def build_grid(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "forecast_grid_*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")


def export_boundary(boundary_path: str):
    gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    return gdf


def build_metrics(data_dir: str):
    parts = []
    for fname, src in [("metrics_baseline.csv", "baseline"), ("metrics_ml.csv", "ml")]:
        d = _read(os.path.join(data_dir, fname))
        if d is not None:
            d["source"] = src
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


DATE_COLS = ("date_t", "date_target", "date_forecast")


def _typed_dates(df):
    """Cột ngày đọc từ CSV là chuỗi -> đổi sang kiểu DATE thật để frontend lọc theo ngày/khoảng ngày
    (WHERE date_target BETWEEN ...) và chỉ mục B-tree hoạt động đúng như với cột ngày."""
    df = df.copy()
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def _resolve_db_url(url: str) -> str:
    """Chọn driver PostgreSQL theo gói ĐANG CÀI. SQLAlchemy 2.1+ đổi driver mặc định của 'postgresql://' từ
    psycopg2 sang psycopg (v3): ai chỉ cài psycopg2-binary sẽ gặp "No module named 'psycopg'". Đồng thời nhận
    cả 'postgres://' (kiểu Heroku/một số nhà cung cấp hay ghi; SQLAlchemy không nhận). Chuỗi đã chỉ rõ driver
    (postgresql+xxx://) thì giữ nguyên."""
    import importlib.util

    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            rest = url[len(prefix):]
            for mod in ("psycopg2", "psycopg"):
                if importlib.util.find_spec(mod) is not None:
                    return f"postgresql+{mod}://{rest}"
            raise SystemExit("❌ Chưa cài driver PostgreSQL — chạy: pip install psycopg2-binary")
    return url


def create_indexes(engine, geo_tables: dict, date_cols: dict) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        for t in geo_tables:
            # Cùng tên với chỉ mục mà geopandas/GeoAlchemy2 tự tạo (idx_<bảng>_geometry) -> IF NOT EXISTS không tạo
            # trùng; vẫn đảm bảo có chỉ mục khi bảng được tạo bằng cách khác (vd append vào bảng cũ thiếu chỉ mục).
            conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{t}_geometry ON "{t}" USING GIST (geometry)'))
        for t, cols in date_cols.items():
            for c in cols:
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_{t}_{c} ON "{t}" ("{c}")'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--boundary", default=None,
                        help="Mặc định: ../data/hcmc_boundary_web.geojson nếu có, không thì hcmc_boundary.geojson")
    parser.add_argument("--model", choices=["xgb", "rf"], default="xgb", help="Model dùng cho lst_predictions")
    parser.add_argument("--tables", default="all", help=f"Danh sách bảng, phân tách bởi dấu phẩy, hoặc 'all'. Gồm: {ALL_TABLES}")
    parser.add_argument("--db_url", default=None, help="Chuỗi kết nối SQLAlchemy; mặc định lấy biến môi trường LST_DB_URL")
    parser.add_argument("--mode", choices=["replace", "append"], default="replace",
                        help="replace = ghi đè bảng (mặc định, như bản cũ); append = nối thêm")
    parser.add_argument("--dry_run", action="store_true", help="Không kết nối DB; chỉ dựng dữ liệu và in số dòng")
    args = parser.parse_args()

    want = ALL_TABLES if args.tables == "all" else [t.strip() for t in args.tables.split(",") if t.strip()]
    bad = [t for t in want if t not in ALL_TABLES]
    if bad:
        raise SystemExit(f"❌ Tên bảng không hợp lệ: {bad}. Hợp lệ: {ALL_TABLES}")

    data_dir = args.data_dir
    boundary = args.boundary
    if boundary is None:
        web = os.path.join(data_dir, "hcmc_boundary_web.geojson")
        boundary = web if os.path.exists(web) else os.path.join(data_dir, "hcmc_boundary.geojson")

    payload = {}  # tên bảng -> (DataFrame/GeoDataFrame, là_không_gian)
    if "predictions" in want:
        print("Đang dựng lst_predictions...")
        payload["lst_predictions"] = (build_predictions(data_dir, args.model), True)
    if "grid" in want:
        g = build_grid(data_dir)
        if g is None:
            print("⚠️  Chưa có forecast_grid_*.csv — bỏ qua lst_forecast_grid (chạy 10_build_inference_grid.py).")
        else:
            payload["lst_forecast_grid"] = (g, True)
    if "boundary" in want:
        if os.path.exists(boundary):
            payload["admin_boundaries"] = (export_boundary(boundary), True)
        else:
            print(f"⚠️  Không thấy {boundary} — BỎ QUA admin_boundaries (frontend sẽ không vẽ được layer ranh giới).")
    if "suhii" in want:
        for name, fname in [("suhii_by_date", "suhii_by_date.csv"), ("suhii_monthly", "suhii_monthly.csv")]:
            d = _read(os.path.join(data_dir, fname))
            if d is None:
                print(f"⚠️  Chưa có {fname} — bỏ qua {name} (chạy 07_compute_suhii.py).")
            else:
                payload[name] = (d, False)
    if "metrics" in want:
        m = build_metrics(data_dir)
        if m is None:
            print("⚠️  Chưa có metrics_baseline.csv/metrics_ml.csv — bỏ qua model_metrics (chạy 05, 06).")
        else:
            payload["model_metrics"] = (m, False)
    if "ablation" in want:
        d = _read(os.path.join(data_dir, "ablation_results.csv"))
        if d is None:
            print("⚠️  Chưa có ablation_results.csv — bỏ qua (chạy 08).")
        else:
            payload["ablation_results"] = (d, False)
    if "shap" in want:
        d = _read(os.path.join(data_dir, "shap_importance.csv"))
        if d is None:
            print("⚠️  Chưa có shap_importance.csv — bỏ qua (chạy 08).")
        else:
            payload["shap_importance"] = (d, False)

    print("\nBảng sẽ đẩy:")
    for name, (df, _) in payload.items():
        print(f"   {name:<20} {len(df):>9} dòng")
    if args.dry_run:
        print("\n(--dry_run: không kết nối CSDL.)")
        return
    if not payload:
        raise SystemExit("Không có gì để đẩy.")

    db_url = args.db_url or os.environ.get("LST_DB_URL")
    if not db_url:
        raise SystemExit(
            "❌ Chưa có chuỗi kết nối CSDL. Đặt biến môi trường LST_DB_URL hoặc dùng --db_url, ví dụ:\n"
            "   postgresql://postgres:MATKHAU@localhost:5432/postgres"
        )

    from sqlalchemy import create_engine

    print("\nĐang kết nối Database và đẩy dữ liệu...")
    try:
        engine = create_engine(_resolve_db_url(db_url))
        geo_tables, date_cols = [], {}
        for name, (df, is_geo) in payload.items():
            df = _typed_dates(df)
            if is_geo:
                df.to_postgis(name=name, con=engine, if_exists=args.mode, index=False)
                geo_tables.append(name)
            else:
                df.to_sql(name=name, con=engine, if_exists=args.mode, index=False)
            print(f"✅ Đã đẩy {len(df)} dòng vào bảng '{name}'.")
        date_cols = {
            "lst_predictions": [c for c in ("date_target", "date_t") if "lst_predictions" in payload],
            "lst_forecast_grid": [c for c in ("date_forecast", "date_t") if "lst_forecast_grid" in payload],
            "suhii_by_date": ["date_t"] if "suhii_by_date" in payload else [],
        }
        date_cols = {t: c for t, c in date_cols.items() if c}
        create_indexes(engine, [t for t in geo_tables], date_cols)
        print("✅ Đã tạo chỉ mục (GiST hình học + ngày).")
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"❌ LỖI KẾT NỐI/GHI DATABASE ({type(e).__name__}):", str(e)[:600])
        print("Kiểm tra lại LST_DB_URL (hoặc --db_url), PostGIS đã bật chưa (CREATE EXTENSION postgis;), "
              "và user có quyền tạo bảng trong schema public.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
