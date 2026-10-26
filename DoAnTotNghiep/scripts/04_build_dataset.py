"""
BƯỚC 4 — Dataset dự báo "kỳ quan sát khả dụng tiếp theo" (Train/Val/Test/Spatial)
==================================================================================
VIẾT LẠI 2026-10-19 sau review toàn bộ luồng lấy dữ liệu. Thay stratifiedSample()
(lấy ngẫu nhiên cả ROI x 30 m từng cặp ngày, bỏ pixel mây SAU khi bốc) bằng
KHUNG ĐIỂM CỐ ĐỊNH + CACHE THEO NGÀY + GHÉP CẶP OFFLINE. Lý do (đã tìm ra khi
review):

  (1) stratifiedSample bốc điểm TRƯỚC rồi mới loại điểm dính mask (dropNulls) ->
      quận nhỏ/hay mây (Quận 1,3,4,5... ~4 km²) bốc 15 điểm mà phần lớn rơi vào
      pixel bị mask => ra 0 mẫu qua hàng trăm ngày. Cách mới: khung điểm cố định
      (mỗi quận n điểm ứng viên), mỗi ngày CHỈ ĐỌC giá trị tại các điểm đó rồi
      giữ điểm hợp lệ -> không còn "bốc trượt".
  (2) .paint() nền 0 trùng với district_idx = 0 (pixel ngoài mọi polygon bị tính
      là quận 0) và polygon chồng lấn bị đè theo thứ tự file; sjoin sau đó lại
      nhân đôi điểm nằm vùng chồng lấn (có thể rơi 1 bản vào train, 1 bản vào
      test = rò rỉ). Cách mới: gán quận bằng hình học ngay lúc tạo khung điểm,
      loại điểm mơ hồ (nằm vùng chồng lấn), KHÔNG sjoin lại.
  (3) Bản cũ tính 'ngày hợp lệ' = có >= 1 pixel sạch ở BẤT KỲ đâu trong ROI. Ngày
      chỉ phủ 1 dải mỏng ở rìa (path/row khác) vẫn được tính -> sinh delta_t = 1
      ngày "ảo", mẫu chỉ ở 1 góc ROI, delta_t lẫn với vị trí. Cách mới: ngày hợp
      lệ = ĐỘ PHỦ (có trọng số diện tích) >= --min_valid_frac. Tính offline từ
      cache nên đổi ngưỡng KHÔNG phải gọi lại GEE (xem bảng quét ngưỡng ở stage pair
      hoặc check_t7_pairing.py).
  (4) Chia tập theo năm của date_t làm mẫu (t=2023-12, nhãn=2024-01) nằm ở Train
      nhưng nhãn thuộc năm Val = rò rỉ nhẹ qua mốc chia. Cách mới: lưu date_target;
      cặp nào có t và nhãn khác giai đoạn bị loại (purge) và báo số lượng.
  (5) Resume theo "ngày có dòng trong raw_dataset.csv" làm ngày 0 mẫu bị gọi lại
      mãi và dễ lẫn dữ liệu bản cũ. Cách mới: cache có trạng thái từng ngày
      (ok/error) + manifest tham số; đổi tham số quan trọng mà không --reset_cache
      thì DỪNG, không âm thầm trộn dữ liệu.
  (6) Mỗi ngày bản cũ tính LST 2 lần (làm t rồi làm nhãn). Cách mới: mỗi ngày
      Landsat chỉ lấy 1 lần (stage lst), dùng cho cả vai trò t lẫn nhãn.

LUỒNG (mỗi stage chạy lại được, tự bỏ qua phần đã xong):
    frame     (không cần GEE) khung điểm ứng viên -> sampling_frame.csv
    lst       (GEE) LST_C + ST_UNC_K tại các điểm cho MỌI ngày Landsat -> cache_lst*.csv
    features  (GEE) NDVI/NDBI/NDWI (S2) + ERA5 tại t cho các ngày hợp lệ -> cache_feat*.csv
    pair      (offline) ghép (t, ngày hợp lệ liền kề) -> raw_dataset.csv (+ cờ Moran)
    (QA)      (offline) qa_overlay.py: kiểm tra chồng lớp VỊ TRÍ / THỜI ĐIỂM / HỆ TỌA ĐỘ trên raw_dataset.csv
              — đề cương: "overlay QA được thực hiện trước khi xuất dataset". Có FAIL thì DỪNG,
              không chia tập (bỏ qua bằng --skip_qa). Kết quả: qa_report.csv
    split     (offline) Train/Val/Test/Spatial -> train.csv, val.csv, test_*.csv (rồi QA chia tập: mốc năm,
              rò rỉ nhãn, Test không gian tách hẳn)

Cách chạy:
    # Thử nhanh 3 ngày để chắc code GEE chạy được trên máy bạn (vài phút):
    python 04_build_dataset.py --project YOUR_ID --rebuild_raw --limit_dates 3

    # Chạy thật (lâu; đứt giữa chừng chạy lại đúng lệnh này sẽ tự tiếp tục):
    python 04_build_dataset.py --project YOUR_ID --rebuild_raw
    python 04_build_dataset.py --project YOUR_ID --rebuild_raw --workers 4   # nhanh hơn ~3-4x (thử trước)

    # Đổi ngưỡng ngày hợp lệ / chia lại tập — KHÔNG cần GEE:
    python 04_build_dataset.py --min_valid_frac 0.15

    # Chỉ chạy 1 stage:
    python 04_build_dataset.py --stage pair
    python 04_build_dataset.py --project YOUR_ID --stage features

    # Bỏ cache cũ khi cố ý đổi tham số (zscore, ST_QA, cửa sổ S2, khung điểm...):
    python 04_build_dataset.py --project YOUR_ID --rebuild_raw --reset_cache
"""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prepare_features", os.path.join(_THIS_DIR, "03_prepare_features.py")
)
prepare_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_features)

FEATURE_COLS = prepare_features.FEATURE_COLS
QA_COLS = prepare_features.QA_COLS
ERA5_VARS = prepare_features.ERA5_VARS
CRS_UTM = prepare_features.CRS_UTM

FRAME_COLS = ["point_id", "lon", "lat", "district_idx", "ADM1_NAME", "ADM2_NAME", "district_area_km2"]
RAW_HEADER_COLS = (
    ["point_id", "lon", "lat", "date_t", "date_target", "district_idx", "ADM1_NAME", "ADM2_NAME", "district_area_km2"]
    + FEATURE_COLS
    + ["LST_target"]
    + QA_COLS
    + ["moran_outlier"]
)
SPLIT_KEEP_COLS = (
    ["point_id", "lon", "lat", "date_t", "date_target", "ADM1_NAME", "ADM2_NAME", "district_area_km2"]
    + FEATURE_COLS
    + ["LST_target"]
    + QA_COLS
)

# 2-3 quận/huyện giữ riêng làm Test không gian (đề cương: "2–3 quận/huyện").
# Cần Giờ (rừng ngập mặn/ven biển), Củ Chi (ven đô/nông nghiệp), Dầu Tiếng (Bình
# Dương cũ — kiểm tra khái quát sang vùng mới sáp nhập, đúng TODO cũ của bạn).
# Ghi tên KHÔNG dấu, so khớp bằng normalize_vn_name nên GADM ghi "CầnGiờ" vẫn khớp.
# Lưu ý: 3 quận này đều nông thôn/ven đô -> Test không gian KHÔNG đo được khả năng
# khái quát sang vùng đô thị lõi. Cân nhắc thêm 1 quận ven đô thị (vd "Go Vap").
SPATIAL_TEST_DISTRICTS = ["Can Gio", "Cu Chi", "Dau Tieng"]

TRAIN_END_YEAR = 2023
VAL_YEAR = 2024
TEST_START_YEAR = 2025


# ---------------------------------------------------------------------
# TIỆN ÍCH: đường dẫn, CSV append, manifest
# ---------------------------------------------------------------------

def _p(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, name)


def _read_csv_if_exists(path: str, **kw):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path, **kw)
    return None


def _append_csv(path: str, df: pd.DataFrame, columns: list) -> None:
    if df is None or len(df) == 0:
        return
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    df[columns].to_csv(path, mode="a", header=not exists, index=False)


def _eta(done: int, total: int, t0: float) -> str:
    if done == 0:
        return "?"
    rem = (time.time() - t0) / done * (total - done)
    return f"{rem / 60:.1f} phút"


def ensure_manifest(data_dir: str, section: str, params: dict, cache_files: list, reset: bool) -> None:
    """So tham số của lần chạy này với tham số đã dùng để tạo cache. Khác nhau mà
    không --reset_cache -> DỪNG (tránh âm thầm trộn dữ liệu 2 cấu hình khác nhau —
    đúng sự cố 'raw_dataset.csv lẫn dữ liệu trước bản sửa' từng gặp)."""
    mp = _p(data_dir, "cache_manifest.json")
    manifest = {}
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    old = manifest.get(section)
    if old is not None and old != params:
        diff = {k: (old.get(k), params.get(k)) for k in sorted(set(old) | set(params)) if old.get(k) != params.get(k)}
        if not reset:
            raise SystemExit(
                f"❌ Tham số cache '{section}' KHÁC lần chạy trước (cũ -> mới): {diff}\n"
                f"   Chạy lại với --reset_cache (xóa cache '{section}' rồi làm lại) hoặc đưa tham số về như cũ."
            )
        for fpath in cache_files:
            if os.path.exists(fpath):
                os.remove(fpath)
        print(f"   🧹 --reset_cache: đã xóa cache '{section}' do tham số đổi: {diff}")
    manifest[section] = params
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------
# STAGE 1 — KHUNG ĐIỂM (không cần GEE)
# ---------------------------------------------------------------------

def _random_points_in_polygon(poly, n: int, rng, max_batches: int = 400) -> np.ndarray:
    """n điểm ngẫu nhiên đều trong poly (rejection sampling, tọa độ mét UTM)."""
    minx, miny, maxx, maxy = poly.bounds
    got, parts = 0, []
    batch = max(4000, n * 30)
    try:
        import shapely

        contains_xy = shapely.contains_xy
    except (ImportError, AttributeError):
        from shapely.geometry import Point
        from shapely.prepared import prep

        pp = prep(poly)

        def contains_xy(g, xs, ys):  # noqa: ARG001 - chữ ký giống shapely 2.0
            return np.array([pp.contains(Point(x, y)) for x, y in zip(xs, ys)])

    for _ in range(max_batches):
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        m = contains_xy(poly, xs, ys)
        sel = np.column_stack([xs[m], ys[m]])
        parts.append(sel)
        got += len(sel)
        if got >= n:
            break
    arr = np.vstack(parts) if parts else np.empty((0, 2))
    return arr[:n]


def build_sampling_frame(boundary_path: str, n_per_district: int, seed: int, edge_buffer_m: float) -> pd.DataFrame:
    """Khung điểm ỨNG VIÊN cố định: mỗi đơn vị phân vùng n_per_district điểm ngẫu
    nhiên (seed cố định), cách ranh giới >= edge_buffer_m (tránh pixel lẫn ranh
    giới), loại điểm nằm vùng polygon chồng lấn. Cùng 1 điểm được đọc ở MỌI ngày
    -> mỗi điểm là 1 chuỗi thời gian ("các vị trí/pixel xuất hiện xuyên suốt cả
    ba tập" — đúng kịch bản Temporal trong đề cương)."""
    import geopandas as gpd

    districts = prepare_features.load_districts(boundary_path)
    dist_m = districts.to_crs(CRS_UTM)
    rng = np.random.default_rng(seed)

    rows = []
    for _, r in dist_m.iterrows():
        geom = r.geometry
        inner = geom.buffer(-edge_buffer_m) if edge_buffer_m > 0 else geom
        use = inner if (not inner.is_empty and inner.area >= 0.3 * geom.area) else geom
        xy = _random_points_in_polygon(use, n_per_district, rng)
        for x, y in xy:
            rows.append((x, y, int(r.district_idx)))
    pts = pd.DataFrame(rows, columns=["x", "y", "district_idx"])

    pts_gdf = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.x, pts.y), crs=CRS_UTM)
    polys = dist_m[["district_idx", "geometry"]].rename(columns={"district_idx": "hit_idx"})
    joined = gpd.sjoin(pts_gdf, polys, how="left", predicate="within")
    n_hit = joined.groupby(level=0)["hit_idx"].nunique()
    own_ok = joined[joined["hit_idx"] == joined["district_idx"]].index.unique()
    keep = np.asarray(pts_gdf.index.isin(own_ok)) & np.asarray([n_hit.get(i, 0) == 1 for i in pts_gdf.index])
    n_dropped = int((~keep).sum())
    pts_gdf = pts_gdf[keep].copy()

    ll = pts_gdf.to_crs("EPSG:4326")
    pts_gdf["lon"] = ll.geometry.x.values
    pts_gdf["lat"] = ll.geometry.y.values
    frame = pts_gdf.drop(columns="geometry").merge(
        districts[["district_idx", "ADM1_NAME", "ADM2_NAME", "district_area_km2"]], on="district_idx"
    )
    frame = frame.sort_values(["district_idx"]).reset_index(drop=True)
    frame["point_id"] = np.arange(len(frame))
    frame = frame[FRAME_COLS]
    frame[["lon", "lat"]] = frame[["lon", "lat"]].round(7)

    print(f"   Khung điểm: {len(frame)} điểm / {frame['district_idx'].nunique()} đơn vị "
          f"(loại {n_dropped} điểm mơ hồ nằm vùng polygon chồng lấn).")
    thin = frame.groupby("district_idx").size()
    print(f"   Điểm/đơn vị: min={thin.min()} median={int(thin.median())} max={thin.max()}")
    return frame


def frame_hash(frame: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(frame[["point_id", "lon", "lat"]], index=False).values.tobytes()
    return hashlib.md5(h).hexdigest()[:12]


def frame_weights(frame: pd.DataFrame) -> pd.Series:
    """Trọng số diện tích của từng điểm khung = diện tích quận / số điểm của quận
    đó. Dùng để tính độ phủ theo DIỆN TÍCH (không phải theo số điểm, vì mỗi quận
    có số điểm bằng nhau dù diện tích chênh ~180 lần)."""
    n = frame.groupby("district_idx")["point_id"].transform("count")
    return pd.Series((frame["district_area_km2"] / n).values, index=frame["point_id"].values)


# ---------------------------------------------------------------------
# STAGE 2 — LST TẠI ĐIỂM CHO MỌI NGÀY LANDSAT (GEE)
# ---------------------------------------------------------------------

LST_CACHE = "cache_lst.csv"
LST_DATES = "cache_lst_dates.csv"
FEAT_CACHE = "cache_feat.csv"
FEAT_DATES = "cache_feat_dates.csv"

LST_COLS = ["date", "point_id", "LST_C", "ST_UNC_K"]
LST_DATE_COLS = ["date", "t_utc_ms", "n_valid", "n_frame", "status", "msg"]
FEAT_BANDS = ["NDVI", "NDBI", "NDWI", "s2_n_img"] + ERA5_VARS + ["era5_dt_min"]
FEAT_COLS_CACHE = ["date", "point_id"] + FEAT_BANDS
FEAT_DATE_COLS = ["date", "status", "fwd_eff", "n_points", "n_complete", "era5_dt_min", "s2_n_img", "msg"]


def _lst_image_for_date(roi, d: str, mask_kwargs: dict):
    import ee

    d1 = (datetime.strptime(d, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(d, d1)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(d, d1)
    col = l8.merge(l9).map(lambda im: prepare_features.mask_landsat_c2l2(im, **mask_kwargs))
    return col.mosaic().select(["LST_C", "ST_UNC_K"])


def _lst_worker(roi, frame, d, t_ms, mask_kwargs):
    img = _lst_image_for_date(roi, d, mask_kwargs)
    df = prepare_features.reduce_at_points(img, frame, ["LST_C", "ST_UNC_K"], label=d)
    return df.dropna(subset=["LST_C"])


def run_stage_lst(args, frame: pd.DataFrame) -> None:
    data_dir = args.outdir
    mask_kwargs = dict(
        zscore_threshold=args.zscore, focal_radius_px=args.focal_radius_px,
        apply_zscore=True, max_st_unc_k=args.max_st_unc_k,
    )
    params = {"frame_hash": frame_hash(frame), **{k: v for k, v in mask_kwargs.items()},
              "scale": prepare_features.GRID_SCALE_M, "crs": CRS_UTM}
    ensure_manifest(data_dir, "lst", params, [_p(data_dir, LST_CACHE), _p(data_dir, LST_DATES)], args.reset_cache)

    prepare_features.init_ee(args.project)
    roi = prepare_features.load_boundary_roi(args.boundary)

    today = datetime.now(timezone.utc).date()
    end_eff = min(datetime.strptime(args.end, "%Y-%m-%d").date(), today + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\n[lst] Liệt kê ngày Landsat 8/9 chạm ROI ({args.start} -> {end_eff}) — chỉ đọc metadata...")
    scene_times = prepare_features.list_landsat_scene_times(
        roi, args.start, end_eff, max_scene_cloud=args.max_scene_cloud
    )
    print(f"   Tổng {len(scene_times)} ngày ứng viên.")

    done = _read_csv_if_exists(_p(data_dir, LST_DATES))
    done_ok = set(done.loc[done["status"] == "ok", "date"]) if done is not None else set()
    pending = [(d, ms) for d, ms in scene_times.items() if d not in done_ok]
    if args.limit_dates:
        pending = pending[: args.limit_dates]
    print(f"   Đã xong {len(done_ok)} | cần chạy {len(pending)} ngày (workers={args.workers}).")
    if not pending:
        return

    n_frame = len(frame)
    t0, n_done, errors = time.time(), 0, []

    def handle(d, ms, df, err):
        nonlocal n_done
        n_done += 1
        if err is not None:
            errors.append(d)
            _append_csv(_p(data_dir, LST_DATES), pd.DataFrame(
                [{"date": d, "t_utc_ms": ms, "n_valid": 0, "n_frame": n_frame, "status": "error", "msg": str(err)[:200]}]
            ), LST_DATE_COLS)
            print(f"[{n_done}/{len(pending)}] {d}: ❌ {str(err)[:100]} (sẽ thử lại ở lần chạy sau)")
            return
        out = df.copy()
        out["date"] = d
        _append_csv(_p(data_dir, LST_CACHE), out, LST_COLS)  # điểm TRƯỚC, trạng thái ngày SAU
        _append_csv(_p(data_dir, LST_DATES), pd.DataFrame(
            [{"date": d, "t_utc_ms": ms, "n_valid": len(out), "n_frame": n_frame, "status": "ok", "msg": ""}]
        ), LST_DATE_COLS)
        print(f"[{n_done}/{len(pending)}] {d}: {len(out)}/{n_frame} điểm hợp lệ ({len(out) / n_frame * 100:.1f}%)"
              f" | còn ~{_eta(n_done, len(pending), t0)}")

    if args.workers <= 1:
        for d, ms in pending:
            try:
                handle(d, ms, _lst_worker(roi, frame, d, ms, mask_kwargs), None)
            except Exception as e:  # noqa: BLE001
                handle(d, ms, None, e)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_lst_worker, roi, frame, d, ms, mask_kwargs): (d, ms) for d, ms in pending}
            for fut in as_completed(futs):
                d, ms = futs[fut]
                try:
                    handle(d, ms, fut.result(), None)
                except Exception as e:  # noqa: BLE001
                    handle(d, ms, None, e)

    print(f"\n[lst] Xong: {n_done - len(errors)} ngày ok, {len(errors)} lỗi. Cache: {_p(data_dir, LST_CACHE)}")
    if errors:
        print(f"   Ngày lỗi (chạy lại cùng lệnh để thử lại): {errors[:15]}{' ...' if len(errors) > 15 else ''}")


# ---------------------------------------------------------------------
# ĐỘ PHỦ / NGÀY HỢP LỆ (offline, từ cache LST)
# ---------------------------------------------------------------------

def load_lst_cache(data_dir: str):
    dates = _read_csv_if_exists(_p(data_dir, LST_DATES))
    lst = _read_csv_if_exists(_p(data_dir, LST_CACHE))
    if dates is None or lst is None:
        raise SystemExit(f"❌ Chưa có {LST_CACHE}/{LST_DATES} trong {data_dir} — chạy stage lst (cần GEE) trước.")
    dates = dates.drop_duplicates("date", keep="last")
    dates = dates[dates["status"] == "ok"].copy()
    lst = lst.drop_duplicates(["date", "point_id"], keep="last")
    lst = lst[lst["date"].isin(dates["date"])]
    return dates, lst


def compute_date_coverage(lst_long: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """Độ phủ LST hợp lệ theo DIỆN TÍCH (0..1) của từng ngày."""
    w = frame_weights(frame)
    tmp = lst_long[["date", "point_id"]].copy()
    tmp["w"] = tmp["point_id"].map(w)
    return (tmp.groupby("date")["w"].sum() / w.sum()).rename("coverage")


def valid_dates_table(dates: pd.DataFrame, lst_long: pd.DataFrame, frame: pd.DataFrame, min_valid_frac: float) -> pd.DataFrame:
    cov = compute_date_coverage(lst_long, frame)
    tbl = dates.merge(cov, left_on="date", right_index=True, how="left")
    tbl["coverage"] = tbl["coverage"].fillna(0.0)
    tbl = tbl.sort_values("date").reset_index(drop=True)
    tbl["valid"] = tbl["coverage"] >= min_valid_frac
    return tbl


def pairs_from_valid(valid_dates: list) -> list:
    return list(zip(valid_dates[:-1], valid_dates[1:]))


def print_threshold_sweep(tbl: pd.DataFrame, chosen: float) -> None:
    print("\n   Quét ngưỡng độ phủ (ngày hợp lệ -> phân phối delta_t). Chọn ngưỡng ở đoạn delta_t")
    print("   ổn định, tránh cả 'quá thấp' (nhiều delta_t=1 ảo từ dải mỏng) lẫn 'quá cao' (mất nhiều ngày):")
    print(f"   {'ngưỡng':>7} {'n_ngày':>7} {'n_cặp':>6} {'dt_min':>6} {'dt_med':>6} {'dt_mean':>7} {'dt_max':>6} {'%dt=1':>6}")
    for thr in sorted({0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, chosen}):
        vd = tbl.loc[tbl["coverage"] >= thr, "date"].tolist()
        if len(vd) < 2:
            print(f"   {thr:>7.2f} {len(vd):>7} {'-':>6}")
            continue
        dts = np.array([(datetime.strptime(b, '%Y-%m-%d') - datetime.strptime(a, '%Y-%m-%d')).days
                        for a, b in zip(vd[:-1], vd[1:])])
        mark = "  <== đang dùng" if abs(thr - chosen) < 1e-9 else ""
        print(f"   {thr:>7.2f} {len(vd):>7} {len(dts):>6} {dts.min():>6} {int(np.median(dts)):>6} "
              f"{dts.mean():>7.1f} {dts.max():>6} {(dts == 1).mean() * 100:>5.1f}%{mark}")


# ---------------------------------------------------------------------
# STAGE 3 — ĐẶC TRƯNG TẠI t (S2 + ERA5) (GEE)
# ---------------------------------------------------------------------

def _feat_worker(roi, t, t_ms, pts, back_days, fwd_eff):
    import ee

    s2 = prepare_features.get_sentinel2_features(roi, t, back_days, fwd_eff)
    era5 = prepare_features.get_era5_features(roi, ee.Date(int(t_ms)))
    img = ee.Image.cat([s2, era5])
    return prepare_features.reduce_at_points(img, pts, FEAT_BANDS, label=t)


def run_stage_features(args, frame: pd.DataFrame) -> None:
    import ee

    data_dir = args.outdir
    dates, lst = load_lst_cache(data_dir)
    lst_params = {"frame_hash": frame_hash(frame), "zscore_threshold": args.zscore,
                  "focal_radius_px": args.focal_radius_px, "apply_zscore": True, "max_st_unc_k": args.max_st_unc_k,
                  "scale": prepare_features.GRID_SCALE_M, "crs": CRS_UTM}
    params = {"frame_hash": frame_hash(frame), "s2_back_days": args.s2_back_days,
              "s2_fwd_days_requested": args.s2_fwd_days, "era5_window_min": 60, "s2_prob_thr": 40.0,
              "s2_scl_mask": True, "lst": lst_params}
    ensure_manifest(data_dir, "feat", params, [_p(data_dir, FEAT_CACHE), _p(data_dir, FEAT_DATES)], args.reset_cache)

    tbl = valid_dates_table(dates, lst, frame, args.min_valid_frac)
    vd = tbl.loc[tbl["valid"], "date"].tolist()
    print(f"\n[features] {len(vd)} ngày hợp lệ (độ phủ >= {args.min_valid_frac:.2f}) trong {len(tbl)} ngày có cache LST.")
    if len(vd) < 2:
        raise SystemExit("❌ Dưới 2 ngày hợp lệ — hạ --min_valid_frac hoặc chạy thêm stage lst.")
    needed = vd[:-1]  # ngày cuối chuỗi không có "kỳ ảnh tiếp theo"
    nxt = {a: (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days for a, b in zip(vd[:-1], vd[1:])}
    t_ms_map = tbl.set_index("date")["t_utc_ms"].to_dict()

    def fwd_for(t):
        # Chặn cứng: ảnh S2 không bao giờ chụp sau ngày nhãn (t + delta_t)
        return int(min(args.s2_fwd_days, max(nxt[t] - 1, 0)))

    prepare_features.init_ee(args.project)
    roi = prepare_features.load_boundary_roi(args.boundary)
    era5_max = prepare_features.get_collection_max_date(prepare_features.ERA5_COLLECTION)
    print(f"   ERA5-Land trên GEE có dữ liệu tới: {era5_max}")

    done = _read_csv_if_exists(_p(data_dir, FEAT_DATES))
    done_ok = set(done.loc[done["status"] == "ok", "date"]) if done is not None else set()

    # Cache cũ có thể được tính với cửa sổ S2 nhìn sau t RỘNG hơn mức giờ được phép: hạ --min_valid_frac làm
    # xuất hiện thêm ngày hợp lệ -> ngày kế tiếp gần hơn -> delta_t-1 nhỏ hơn -> ảnh S2 có thể chụp SAU ngày nhãn.
    # Chỉ xảy ra khi --s2_fwd_days > 0 (mặc định 0 thì không bao giờ). Xóa dòng cũ của các ngày đó rồi tính lại,
    # để không trộn hai cấu hình cửa sổ trong cùng một cache.
    if done is not None and args.s2_fwd_days > 0:
        last_ok = done[done["status"] == "ok"].drop_duplicates("date", keep="last").set_index("date")["fwd_eff"]
        stale = [t for t in needed if t in last_ok.index and pd.notna(last_ok[t]) and int(last_ok[t]) > fwd_for(t)]
        if stale:
            print(f"   🔁 {len(stale)} ngày có cache S2 với cửa sổ tiến rộng hơn mức hiện cho phép (ngày hợp lệ kế tiếp "
                  f"gần hơn) -> xóa dòng cũ và tính lại: {stale[:5]}{' ...' if len(stale) > 5 else ''}")
            fc = _read_csv_if_exists(_p(data_dir, FEAT_CACHE))
            if fc is not None:
                fc[~fc["date"].isin(stale)].to_csv(_p(data_dir, FEAT_CACHE), index=False)
            done[~done["date"].isin(stale)].to_csv(_p(data_dir, FEAT_DATES), index=False)
            done_ok -= set(stale)
    no_era5 = [t for t in needed if era5_max is not None and t > era5_max]
    if no_era5:
        print(f"   ⚠️  {len(no_era5)} ngày t > {era5_max} CHƯA có ERA5 -> tạm bỏ qua (khi ERA5 cập nhật, chạy lại stage này): "
              f"{no_era5[:5]}{' ...' if len(no_era5) > 5 else ''}")
    pending = [t for t in needed if t not in done_ok and t not in set(no_era5)]
    if args.limit_dates:
        pending = pending[: args.limit_dates]
    print(f"   Đã xong {len(done_ok)} | cần chạy {len(pending)} ngày (workers={args.workers}).")
    if not pending:
        return

    lst_idx = lst.groupby("date")["point_id"].apply(list).to_dict()
    frame_i = frame.set_index("point_id")
    t0, n_done, errors = time.time(), 0, []

    def points_for(t):
        ids = lst_idx.get(t, [])
        p = frame_i.loc[ids, ["lon", "lat"]].reset_index()
        return p

    def handle(t, df, fwd_eff, err):
        nonlocal n_done
        n_done += 1
        if err is not None:
            errors.append(t)
            _append_csv(_p(data_dir, FEAT_DATES), pd.DataFrame([{
                "date": t, "status": "error", "fwd_eff": fwd_eff, "n_points": 0, "n_complete": 0,
                "era5_dt_min": np.nan, "s2_n_img": np.nan, "msg": str(err)[:200]}]), FEAT_DATE_COLS)
            print(f"[{n_done}/{len(pending)}] {t}: ❌ {str(err)[:100]} (sẽ thử lại ở lần chạy sau)")
            return
        need_cols = ["NDVI", "NDBI", "NDWI"] + ERA5_VARS
        comp = df.dropna(subset=need_cols).copy()
        comp["date"] = t
        _append_csv(_p(data_dir, FEAT_CACHE), comp, FEAT_COLS_CACHE)
        _append_csv(_p(data_dir, FEAT_DATES), pd.DataFrame([{
            "date": t, "status": "ok", "fwd_eff": fwd_eff, "n_points": len(df), "n_complete": len(comp),
            "era5_dt_min": float(df["era5_dt_min"].median()) if len(df) else np.nan,
            "s2_n_img": float(df["s2_n_img"].median()) if len(df) else np.nan, "msg": ""}]), FEAT_DATE_COLS)
        print(f"[{n_done}/{len(pending)}] {t}: {len(comp)}/{len(df)} điểm đủ S2+ERA5 "
              f"(s2_n_img={df['s2_n_img'].median() if len(df) else float('nan'):.0f}) | còn ~{_eta(n_done, len(pending), t0)}")

    if args.workers <= 1:
        for t in pending:
            f_eff = fwd_for(t)
            try:
                handle(t, _feat_worker(roi, t, t_ms_map[t], points_for(t), args.s2_back_days, f_eff), f_eff, None)
            except Exception as e:  # noqa: BLE001
                handle(t, None, f_eff, e)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {}
            for t in pending:
                f_eff = fwd_for(t)
                futs[ex.submit(_feat_worker, roi, t, t_ms_map[t], points_for(t), args.s2_back_days, f_eff)] = (t, f_eff)
            for fut in as_completed(futs):
                t, f_eff = futs[fut]
                try:
                    handle(t, fut.result(), f_eff, None)
                except Exception as e:  # noqa: BLE001
                    handle(t, None, f_eff, e)

    print(f"\n[features] Xong: {n_done - len(errors)} ngày ok, {len(errors)} lỗi. Cache: {_p(data_dir, FEAT_CACHE)}")
    if errors:
        print(f"   Ngày lỗi (chạy lại cùng lệnh để thử lại): {errors[:15]}{' ...' if len(errors) > 15 else ''}")


# ---------------------------------------------------------------------
# STAGE 4 — GHÉP CẶP (offline) + kiểm tra Moran cục bộ
# ---------------------------------------------------------------------

def flag_local_spatial_outliers(
    df: pd.DataFrame, value_col: str = "LST_t", k_neighbors: int = 8, sig_level: float = 0.05
) -> np.ndarray:
    """Local Moran's I (LISA) tính RIÊNG cho từng ngày date_t (k hàng xóm gần nhất
    trong CÙNG 1 ngày). Trả về mảng bool: True = High-Low/Low-High với p_sim <=
    sig_level. CHỈ ĐÁNH DẤU, không tự xóa.

    ⚠️ Giới hạn cần ghi vào báo cáo: các điểm lấy mẫu thưa (cách nhau hàng km) nên
    'hàng xóm' không phải pixel liền kề — High-Low/Low-High ở đây phần lớn là
    tương phản thật (công viên giữa khu xây dựng, mặt nước...), KHÔNG nhất thiết là
    nhiễu. Kiểm tra bước nhảy giữa pixel LÂN CẬN thật sự đã được làm bằng z-score
    cục bộ trong mask_landsat_c2l2 (cửa sổ 5x5 pixel). Vì vậy mặc định KHÔNG loại
    điểm theo cờ này."""
    flags = np.zeros(len(df), dtype=bool)
    try:
        from esda.moran import Moran_Local
        from libpysal.weights import KNN
    except ImportError:
        print("   ⚠️  Chưa cài libpysal/esda (pip install libpysal esda) -> BỎ QUA bước Moran's I.")
        return flags

    df = df.reset_index(drop=True)
    for _, group in df.groupby("date_t"):
        if len(group) < k_neighbors + 1:
            continue
        try:
            w = KNN.from_array(group[["lon", "lat"]].to_numpy(), k=k_neighbors)
            w.transform = "r"
            lisa = Moran_Local(group[value_col].to_numpy(), w, permutations=199, seed=42)
        except Exception:
            continue
        is_out = ((lisa.q == 2) | (lisa.q == 3)) & (lisa.p_sim <= sig_level)
        flags[group.index.to_numpy()] = is_out
    return flags


def run_stage_pair(args, frame: pd.DataFrame) -> None:
    data_dir = args.outdir
    dates, lst = load_lst_cache(data_dir)
    feat = _read_csv_if_exists(_p(data_dir, FEAT_CACHE))
    feat_dates = _read_csv_if_exists(_p(data_dir, FEAT_DATES))
    if feat is None or feat_dates is None:
        raise SystemExit(f"❌ Chưa có {FEAT_CACHE} — chạy stage features (cần GEE) trước.")
    feat = feat.drop_duplicates(["date", "point_id"], keep="last")
    feat_dates = feat_dates.drop_duplicates("date", keep="last")
    fwd_eff = feat_dates.set_index("date")["fwd_eff"].to_dict()

    print("\n[pair] Ghép mẫu (t, ngày hợp lệ liền kề tiếp theo)...")
    tbl = valid_dates_table(dates, lst, frame, args.min_valid_frac)
    print_threshold_sweep(tbl, args.min_valid_frac)
    vd = tbl.loc[tbl["valid"], "date"].tolist()
    pairs = pairs_from_valid(vd)
    print(f"\n   Dùng ngưỡng {args.min_valid_frac:.2f}: {len(vd)} ngày hợp lệ -> {len(pairs)} cặp (t, t_next); "
          f"ngày cuối chuỗi ({vd[-1] if vd else '-'}) không có nhãn nên bị bỏ.")

    lst_by = {d: g.set_index("point_id") for d, g in lst.groupby("date")}
    feat_by = {d: g.set_index("point_id") for d, g in feat.groupby("date")}

    chunks, n_skip_nofeat, n_skip_fwd = [], 0, 0
    for t, tn in pairs:
        delta_t = (datetime.strptime(tn, "%Y-%m-%d") - datetime.strptime(t, "%Y-%m-%d")).days
        if args.max_delta_t and delta_t > args.max_delta_t:
            continue
        if t not in feat_by or tn not in lst_by or t not in lst_by:
            n_skip_nofeat += 1
            continue
        if fwd_eff.get(t, 0) > max(delta_t - 1, 0):
            n_skip_fwd += 1  # feature S2 của t được tính với cửa sổ nhìn sau t quá xa so với delta_t hiện tại
            continue
        ft, lt, ln = feat_by[t], lst_by[t], lst_by[tn]
        ids = ft.index.intersection(lt.index).intersection(ln.index)
        if len(ids) == 0:
            continue
        d = pd.DataFrame(index=ids)
        d.index.name = "point_id"
        d["LST_t"] = lt.loc[ids, "LST_C"].to_numpy()
        d["st_unc_t"] = lt.loc[ids, "ST_UNC_K"].to_numpy()
        d["LST_target"] = ln.loc[ids, "LST_C"].to_numpy()
        d["st_unc_target"] = ln.loc[ids, "ST_UNC_K"].to_numpy()
        for c in ["NDVI", "NDBI", "NDWI"] + ERA5_VARS + ["era5_dt_min", "s2_n_img"]:
            d[c] = ft.loc[ids, c].to_numpy()
        d["date_t"], d["date_target"], d["delta_t"] = t, tn, float(delta_t)
        chunks.append(d.reset_index())

    if n_skip_nofeat:
        print(f"   ⚠️  {n_skip_nofeat} cặp bị bỏ vì thiếu cache đặc trưng/LST (chạy stage features cho đủ).")
    if n_skip_fwd:
        print(f"   ⚠️  {n_skip_fwd} cặp bị bỏ vì cửa sổ S2 của t nhìn sau t xa hơn delta_t-1 -> chạy lại "
              f"`--stage features` (cần --s2_fwd_days như cũ): các ngày đó tự được xóa cache và tính lại.")
    if not chunks:
        raise SystemExit("❌ Không ghép được cặp nào — kiểm tra cache LST/features hoặc hạ --min_valid_frac.")

    raw = pd.concat(chunks, ignore_index=True)
    n0 = len(raw)
    raw = raw.merge(frame[FRAME_COLS], on="point_id", how="left")

    # Lọc kiểm soát chất lượng cuối (đề cương: loại dòng NaN; ERA5 phải khớp giờ ±60 phút)
    raw = raw.dropna(subset=FEATURE_COLS + ["LST_target"])
    bad_era5 = (raw["era5_dt_min"] < 0) | (raw["era5_dt_min"] > 60.0 + 1e-6)
    if bad_era5.any():
        print(f"   ⚠️  Loại {int(bad_era5.sum())} dòng có ERA5 lệch giờ chụp > 60 phút (hoặc thiếu ERA5).")
        raw = raw[~bad_era5]
    dup = raw.duplicated(["point_id", "date_t"]).sum()
    if dup:
        print(f"   ⚠️  Loại {int(dup)} dòng trùng (point_id, date_t).")
        raw = raw.drop_duplicates(["point_id", "date_t"])
    print(f"   {n0} dòng ghép được -> {len(raw)} dòng sau lọc NaN/ERA5/trùng.")

    if args.no_moran:
        raw["moran_outlier"] = False
    else:
        print("   Đang kiểm tra Local Moran's I theo từng ngày (có thể mất vài phút)...")
        raw = raw.reset_index(drop=True)
        raw["moran_outlier"] = flag_local_spatial_outliers(raw, value_col="LST_t")
        n_fl = int(raw["moran_outlier"].sum())
        print(f"   Moran's I: {n_fl}/{len(raw)} dòng ({n_fl / max(len(raw), 1) * 100:.2f}%) bị đánh dấu "
              f"High-Low/Low-High (p<=0.05) — chỉ BÁO CÁO, xem lưu ý trong flag_local_spatial_outliers().")

    raw = raw.sort_values(["date_t", "point_id"]).reset_index(drop=True)[RAW_HEADER_COLS]
    raw_path = _p(data_dir, "raw_dataset.csv")
    raw.to_csv(raw_path, index=False)

    print(f"\n✅ Đã ghi {raw_path}: {len(raw)} dòng, {raw['date_t'].nunique()} ngày t, "
          f"{raw['point_id'].nunique()}/{len(frame)} điểm khung.")
    print("   Phân phối delta_t (theo dòng): " + ", ".join(
        f"{k}={v:.1f}" for k, v in raw["delta_t"].describe()[["min", "50%", "mean", "max"]].items()))
    per_d = raw.groupby("ADM2_NAME").size().reindex(sorted(frame["ADM2_NAME"].unique()), fill_value=0)
    zero = per_d[per_d == 0].index.tolist()
    print(f"   Dòng/đơn vị: min={per_d.min()} median={int(per_d.median())} max={per_d.max()} | đơn vị 0 dòng: {len(zero)}")
    if zero:
        print(f"   ⚠️  Đơn vị KHÔNG có dòng nào: {zero}\n"
              f"      -> chạy diagnose_mask_components.py để xem bước mask nào loại hết pixel ở đó "
              f"(nghi ST_QA/z-score/mây), hoặc tăng --n_points_per_district.")


# ---------------------------------------------------------------------
# STAGE 5 — CHIA TẬP (offline)
# ---------------------------------------------------------------------

def split_dataset(raw_csv_path: str, out_dir: str, spatial_districts, drop_moran_outliers: bool = False) -> None:
    print("\n[split] Đang phân chia tập dữ liệu (Data Splitting)...")
    if not os.path.exists(raw_csv_path):
        raise SystemExit(f"❌ Không thấy {raw_csv_path} — chạy stage pair trước.")
    df = pd.read_csv(raw_csv_path)
    if len(df) == 0:
        raise SystemExit(f"❌ {raw_csv_path} RỖNG — xem lại log các stage GEE/pair trước khi chia dữ liệu.")
    missing = [c for c in RAW_HEADER_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ {raw_csv_path} thiếu cột {missing} — file từ bản 04 CŨ, hãy xóa và build lại bằng --rebuild_raw.")

    if drop_moran_outliers:
        n = int(df["moran_outlier"].sum())
        df = df[~df["moran_outlier"].astype(bool)]
        print(f"   Đã LOẠI {n} dòng Moran outlier (--drop_moran_outliers).")

    d_t = pd.to_datetime(df["date_t"])
    d_n = pd.to_datetime(df["date_target"])

    # Quận/huyện spatial-test: so khớp bằng tên đã chuẩn hóa (bỏ dấu/khoảng trắng)
    keys = {prepare_features.normalize_vn_name(d) for d in spatial_districts}
    is_spatial = df["ADM2_NAME"].apply(prepare_features.normalize_vn_name).isin(keys)
    if not is_spatial.any():
        print(f"\n⚠️  CẢNH BÁO: test_spatial sẽ RỖNG — không đơn vị nào khớp {list(spatial_districts)}.\n"
              f"   Tên thật trong dữ liệu: {sorted(df['ADM2_NAME'].unique().tolist())}")
    else:
        matched = sorted(df.loc[is_spatial, "ADM2_NAME"].unique())
        unmatched = [d for d in spatial_districts if prepare_features.normalize_vn_name(d)
                     not in {prepare_features.normalize_vn_name(m) for m in matched}]
        print(f"   Test không gian khớp: {matched}" + (f" | KHÔNG khớp: {unmatched}" if unmatched else ""))

    spatial = df[is_spatial]
    temporal = df[~is_spatial]
    dt_t, dt_n = d_t[~is_spatial], d_n[~is_spatial]

    # Chia theo NĂM, xét CẢ ngày t LẪN ngày nhãn: cặp có t và nhãn khác giai đoạn bị loại (purge)
    m_train = dt_n.dt.year <= TRAIN_END_YEAR
    m_val = (dt_t.dt.year == VAL_YEAR) & (dt_n.dt.year == VAL_YEAR)
    m_test = dt_t.dt.year >= TEST_START_YEAR
    m_gap = ~(m_train | m_val | m_test)
    df_train, df_val, df_test = temporal[m_train], temporal[m_val], temporal[m_test]
    n_gap = int(m_gap.sum())

    os.makedirs(out_dir, exist_ok=True)
    df_train[SPLIT_KEEP_COLS].to_csv(_p(out_dir, "train.csv"), index=False)
    df_val[SPLIT_KEEP_COLS].to_csv(_p(out_dir, "val.csv"), index=False)
    df_test[SPLIT_KEEP_COLS].to_csv(_p(out_dir, "test_temporal.csv"), index=False)
    spatial[SPLIT_KEEP_COLS].to_csv(_p(out_dir, "test_spatial.csv"), index=False)

    n_main = len(df_train) + len(df_val) + len(df_test)
    summary = []
    for name, part in [("Train", df_train), ("Validation", df_val), ("Test thời gian", df_test), ("Test không gian", spatial)]:
        summary.append({
            "tap": name, "n_dong": len(part),
            "ty_le_pct_trong_3_tap_thoi_gian": round(len(part) / n_main * 100, 2) if name != "Test không gian" and n_main else None,
            "n_ngay_t": part["date_t"].nunique(), "n_don_vi": part["ADM2_NAME"].nunique(),
            "t_min": part["date_t"].min() if len(part) else None, "t_max": part["date_t"].max() if len(part) else None,
            "delta_t_trung_binh": round(part["delta_t"].mean(), 2) if len(part) else None,
        })
    pd.DataFrame(summary).to_csv(_p(out_dir, "split_summary.csv"), index=False)

    print("\n✅ HOÀN TẤT! Đã xuất 4 file CSV + split_summary.csv:")
    for s in summary:
        pct = f" ({s['ty_le_pct_trong_3_tap_thoi_gian']}%)" if s["ty_le_pct_trong_3_tap_thoi_gian"] is not None else ""
        print(f" - {s['tap']:<16}: {s['n_dong']:>8} dòng{pct} | {s['n_ngay_t']} ngày t | {s['n_don_vi']} đơn vị")
    print(f" - Bị loại (purge) vì t và nhãn khác giai đoạn: {n_gap} dòng "
          f"({n_gap / max(len(temporal), 1) * 100:.2f}% phần thời gian)")
    print("   (Tỷ lệ % ở trên chính là số liệu 'tỷ lệ mẫu thực tế' đề cương yêu cầu báo cáo.)")

    tr_units = df_train["ADM2_NAME"].value_counts()
    print(f"\nTrain: {len(tr_units)} đơn vị có dữ liệu; dòng/đơn vị min={tr_units.min() if len(tr_units) else 0} "
          f"median={int(tr_units.median()) if len(tr_units) else 0} max={tr_units.max() if len(tr_units) else 0}")
    print("\nPhân phối delta_t (toàn bộ dữ liệu):")
    print(df["delta_t"].value_counts().sort_index().to_string())


# ---------------------------------------------------------------------
# OVERLAY QA (đề cương: kiểm tra chồng lớp TRƯỚC khi xuất dataset)
# ---------------------------------------------------------------------

def _qa_gate(args, scope: str, spatial, min_valid_frac) -> None:
    """Chạy qa_overlay.py (cùng thư mục). Có kiểm tra FAIL -> DỪNG, không cho pipeline đi tiếp
    với dữ liệu nghi ghép sai vị trí/thời điểm/hệ tọa độ."""
    path = os.path.join(_THIS_DIR, "qa_overlay.py")
    if not os.path.exists(path):
        print("   ⚠️  Không thấy qa_overlay.py cạnh 04_build_dataset.py — BỎ QUA bước overlay QA.")
        return
    spec = importlib.util.spec_from_file_location("qa_overlay", path)
    qa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qa)
    # max_st_unc_k=None: QA đọc ngưỡng từ cache_manifest.json (đúng cấu hình đã dùng khi trích dữ liệu)
    rep = qa.run_qa(args.outdir, args.boundary, scope=scope, min_valid_frac=min_valid_frac,
                    max_st_unc_k=None, spatial_districts=spatial)
    n_fail = int((rep["status"] == "FAIL").sum())
    if n_fail:
        raise SystemExit(
            f"\n❌ Overlay QA ({scope}): {n_fail} kiểm tra FAIL — DỪNG (xem các dòng ❌ ở trên và {_p(args.outdir, 'qa_report.csv')}).\n"
            f"   Sửa nguyên nhân rồi chạy lại. Nếu chắc đó là báo động giả: thêm --skip_qa (và ghi lý do vào báo cáo)."
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", help="GEE Cloud Project ID (bắt buộc cho stage lst/features)")
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--outdir", default="../data/")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--stage", choices=["frame", "lst", "features", "pair", "split", "gee", "offline", "all"], default=None,
                        help="gee = frame+lst+features | offline = pair+split (mặc định khi không có --rebuild_raw) | all = tất cả")
    parser.add_argument("--rebuild_raw", action="store_true",
                        help="Tương đương --stage all: chạy phần GEE (tự tiếp tục từ cache) rồi ghép cặp + chia tập.")
    parser.add_argument("--n_points_per_district", type=int, default=100,
                        help="Số điểm ỨNG VIÊN cố định mỗi đơn vị trong khung (ý nghĩa KHÁC bản cũ 15 điểm/ngày: "
                             "mỗi ngày chỉ giữ điểm hợp lệ trong số này). Tổng điểm ~ 100 x số đơn vị.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge_buffer_m", type=float, default=45.0, help="Cách ranh giới quận tối thiểu (m)")
    parser.add_argument("--rebuild_frame", action="store_true", help="Tạo lại khung điểm (sẽ vô hiệu cache cũ)")
    parser.add_argument("--min_valid_frac", type=float, default=0.10,
                        help="Độ phủ LST hợp lệ tối thiểu (theo diện tích, 0..1) để 1 ngày được tính là 'ngày hợp lệ'. "
                             "Xem bảng quét ngưỡng ở stage pair để chọn.")
    parser.add_argument("--max_delta_t", type=int, default=None, help="Bỏ cặp có delta_t lớn hơn (mặc định: không giới hạn)")
    parser.add_argument("--zscore", type=float, default=3.0)
    parser.add_argument("--focal_radius_px", type=int, default=2)
    parser.add_argument("--max_st_unc_k", type=float, default=prepare_features.DEFAULT_MAX_ST_UNC_K,
                        help="Ngưỡng độ bất định ST_QA (K). Bản cũ 2.0; mặc định mới 3.0 (đo bằng diagnose_mask_components.py)")
    parser.add_argument("--s2_back_days", type=int, default=prepare_features.S2_BACK_DAYS)
    parser.add_argument("--s2_fwd_days", type=int, default=prepare_features.S2_FWD_DAYS,
                        help="Số ngày nhìn SAU t cho ảnh S2 (mặc định 0 = không nhìn tương lai). Nếu >0 luôn bị chặn <= delta_t-1.")
    parser.add_argument("--max_scene_cloud", type=float, default=100.0, help="Bỏ cảnh có CLOUD_COVER >= giá trị này (100 = tắt)")
    parser.add_argument("--workers", type=int, default=1, help="Số luồng gọi GEE song song (thử 3-4 nếu máy/mạng ổn)")
    parser.add_argument("--limit_dates", type=int, default=None, help="Chỉ chạy N ngày đầu cần xử lý (thử nhanh)")
    parser.add_argument("--reset_cache", action="store_true", help="Xóa cache khi tham số quan trọng đã đổi")
    parser.add_argument("--spatial_districts", default=",".join(SPATIAL_TEST_DISTRICTS),
                        help="Quận/huyện giữ làm Test không gian, phân tách bằng dấu phẩy (không cần dấu)")
    parser.add_argument("--no_moran", action="store_true", help="Bỏ qua bước Local Moran's I (nhanh hơn)")
    parser.add_argument("--drop_moran_outliers", action="store_true",
                        help="Loại điểm bị cờ Moran khi chia tập (mặc định chỉ báo cáo — xem lưu ý trong code)")
    parser.add_argument("--skip_qa", action="store_true",
                        help="Bỏ qua overlay QA (qa_overlay.py) trước/sau khi chia tập. Chỉ dùng khi chắc FAIL là báo động giả.")
    args = parser.parse_args()

    stage = "all" if args.rebuild_raw else (args.stage or "offline")
    stages = {
        "frame": ["frame"], "lst": ["lst"], "features": ["features"], "pair": ["pair"], "split": ["split"],
        "gee": ["frame", "lst", "features"], "offline": ["pair", "split"],
        "all": ["frame", "lst", "features", "pair", "split"],
    }[stage]

    os.makedirs(args.outdir, exist_ok=True)
    frame_path = _p(args.outdir, "sampling_frame.csv")

    if "frame" in stages or not os.path.exists(frame_path):
        if os.path.exists(frame_path) and not args.rebuild_frame:
            print(f"[frame] Dùng khung điểm có sẵn {frame_path} (thêm --rebuild_frame nếu muốn tạo lại).")
        else:
            print("[frame] Đang tạo khung điểm ứng viên...")
            fr = build_sampling_frame(args.boundary, args.n_points_per_district, args.seed, args.edge_buffer_m)
            fr.to_csv(frame_path, index=False)
            print(f"   ✅ Đã lưu {frame_path}")
    frame = pd.read_csv(frame_path)

    if "lst" in stages:
        if not args.project:
            raise SystemExit("Thiếu --project (GEE Cloud Project ID) cho stage lst")
        run_stage_lst(args, frame)
    if "features" in stages:
        if not args.project:
            raise SystemExit("Thiếu --project (GEE Cloud Project ID) cho stage features")
        run_stage_features(args, frame)
    spatial = [s.strip() for s in args.spatial_districts.split(",") if s.strip()]
    if "pair" in stages:
        run_stage_pair(args, frame)
    if ("pair" in stages or "split" in stages) and not args.skip_qa:
        # Chỉ truyền --min_valid_frac khi vừa ghép cặp trong lần chạy này; nếu chỉ chia tập thì để QA tự suy ra
        # ngưỡng từ dữ liệu (raw_dataset.csv có thể đã ghép với ngưỡng khác).
        _qa_gate(args, "raw", spatial, args.min_valid_frac if "pair" in stages else None)
    if "split" in stages:
        split_dataset(_p(args.outdir, "raw_dataset.csv"), args.outdir, spatial, drop_moran_outliers=args.drop_moran_outliers)
        if args.limit_dates:
            print("\n   (--limit_dates: chạy thử vài ngày nên bỏ qua QA chia tập — Val/Test sẽ rỗng là bình thường.)")
        elif not args.skip_qa:
            _qa_gate(args, "split", spatial, None)
        print("\nGợi ý: python qa_overlay.py --plot  (ảnh chồng lớp để nhìn bằng mắt)  |  --gee --project ID  (đối chiếu với GEE)")


if __name__ == "__main__":
    main()
