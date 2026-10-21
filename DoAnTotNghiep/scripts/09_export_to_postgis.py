"""
BƯỚC 9 — XUẤT DỰ BÁO + RANH GIỚI HÀNH CHÍNH VÀO POSTGIS
=======================================================================
Giữ nguyên từ bản trước (đã đúng): nạp lại ĐÚNG model đã tune ở
06_train_ml_models.py (không train lại 1 bản khác với tham số mặc định),
để kết quả báo cáo RMSE/MAE/R² khớp với kết quả thực tế chạy trên WebGIS.

CẬP NHẬT 2026-09-09: BỔ SUNG đẩy ranh giới hành chính (hcmc_boundary.geojson)
vào bảng `admin_boundaries` — Tuần 10 đề cương yêu cầu "đẩy kết quả dự báo
LST VÀ ranh giới hành chính vào CSDL", bản trước chỉ mới đẩy lst_predictions,
thiếu hẳn phần ranh giới (nên frontend không vẽ được layer boundary / không
hiển thị được trạng thái "không có dữ liệu" cho vùng bị mây che hoàn toàn).
"""

import argparse
import os
import pickle

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine


def export_boundary(boundary_path: str, engine) -> None:
    """Đẩy ranh giới hành chính (district_idx/ADM2_NAME) vào bảng
    admin_boundaries — dùng chung geometry type "Geometry" (không ép cứng
    POLYGON/MULTIPOLYGON) vì union nhiều quận/huyện dễ ra lẫn cả 2 loại."""
    print("Đang nạp ranh giới hành chính...")
    boundary_gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    boundary_gdf.to_postgis(
        name="admin_boundaries",
        con=engine,
        if_exists="replace",
        index=False,
        dtype={"geometry": "Geometry"},
    )
    print(f"✅ Đã đẩy {len(boundary_gdf)} đơn vị ranh giới vào bảng 'admin_boundaries'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    args = parser.parse_args()

    data_dir = "../data"
    model_path = os.path.join(data_dir, "best_model_xgb.pkl")

    if not os.path.exists(model_path):
        raise SystemExit(
            f"❌ Không thấy {model_path} — chạy 06_train_ml_models.py trước để tạo model đã tune."
        )

    print("Đang nạp model đã tune từ Bước 6...")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model, features = saved["model"], saved["features"]
    print(f"   Model dùng {len(features)} feature: {features}")

    test_df = pd.read_csv(os.path.join(data_dir, "test_temporal.csv"))
    missing = [c for c in features if c not in test_df.columns]
    if missing:
        raise SystemExit(
            f"❌ test_temporal.csv thiếu cột {missing} so với feature model đã train — "
            f"dataset và model đang KHÔNG khớp nhau, kiểm tra lại đã chạy đúng 04 bản mới chưa."
        )

    test_df["LST_pred"] = model.predict(test_df[features])

    export_df = test_df[["lon", "lat", "date_t", "LST_t", "LST_target", "LST_pred", "ADM2_NAME"]].copy()

    print("Đang chuyển đổi sang GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        export_df, geometry=gpd.points_from_xy(export_df.lon, export_df.lat), crs="EPSG:4326"
    )

    # THAY CHUỖI KẾT NỐI CỦA BẠN VÀO ĐÂY (đừng commit mật khẩu thật lên Git):
    DB_URL = os.environ.get("LST_DB_URL", "postgresql://postgres:123456@localhost:5432/postgres")

    print("Đang kết nối Database và đẩy dữ liệu...")
    try:
        engine = create_engine(DB_URL)
        gdf.to_postgis(
            name="lst_predictions",
            con=engine,
            if_exists="replace",
            index=False,
            dtype={"geometry": "Geometry(POINT, 4326)"},
        )
        print("✅ Đã đẩy dữ liệu dự báo vào bảng 'lst_predictions'.")

        if os.path.exists(args.boundary):
            export_boundary(args.boundary, engine)
        else:
            print(
                f"⚠️ Không thấy {args.boundary} — BỎ QUA đẩy ranh giới hành chính. "
                f"Tuần 10 đề cương yêu cầu đẩy CẢ dự báo LẪN ranh giới, thiếu bảng "
                f"admin_boundaries thì frontend sẽ không vẽ được layer boundary / "
                f"trạng thái 'không có dữ liệu'."
            )
    except Exception as e:
        print("❌ LỖI KẾT NỐI DATABASE:", e)
        print("Kiểm tra lại LST_DB_URL (biến môi trường) hoặc DB_URL mặc định trong file.")


if __name__ == "__main__":
    main()
