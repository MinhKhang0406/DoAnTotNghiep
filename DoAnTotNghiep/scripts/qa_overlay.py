"""
KIỂM TRA CHỒNG LỚP (OVERLAY QA) — chạy TRƯỚC khi huấn luyện
=============================================================
Đề cương mục 4 (Data Fusion): "Việc ghép sai vị trí/thời điểm/hệ tọa độ sẽ khiến mô
hình vẫn chạy được nhưng học sai quan hệ, do đó bước kiểm tra chồng lớp (overlay QA)
được thực hiện trước khi xuất dataset." — bản cũ KHÔNG có bước này (mô hình chạy
được dù ghép sai thì không ai biết). Script này là bước đó. Nó trả lời 3 câu hỏi của
đề cương, mỗi câu bằng kiểm tra cụ thể, kết quả PASS / WARN / FAIL:

  VỊ TRÍ      điểm nằm đúng trong quận/huyện được gán (trong hệ UTM 48N), tọa độ khớp
              khung điểm gốc, không bị đảo lon/lat, tên đơn vị khớp ranh giới hiện tại.
  THỜI ĐIỂM   delta_t = hiệu 2 ngày; nhãn nằm ĐÚNG ngày hợp lệ liền kề tiếp theo (không
              bỏ sót ngày ở giữa); ERA5 lệch <= 60 phút; S2 không nhìn quá ngày nhãn;
              dòng ghép ra khớp từng giá trị với cache (LST_t, LST_target, NDVI, ERA5...);
              chuỗi t -> t' -> t'' nhất quán (LST_target của cặp trước == LST_t của cặp sau).
  HỆ TỌA ĐỘ   lon/lat nằm trong khung ranh giới; đối chiếu UTM; (tùy chọn --gee) đọc
              projection/độ phân giải gốc của từng nguồn và TRÍCH LẠI vài điểm từ GEE so
              với cache để chắc cách lấy mẫu (crs=EPSG:32648, scale=30) không lệch.
  KHOẢNG GIÁ TRỊ  LST -10..60 °C, chỉ số [-1,1], ERA5 trong giới hạn vật lý, ST_QA <= ngưỡng.
  BẰNG CHỨNG THỐNG KÊ  (cảnh báo sớm khi ghép lệch mà vẫn "hợp lệ" về hình thức):
              NDVI–LST_t tương quan âm; LST_t–LST_target hơn hẳn ghép ngẫu nhiên giữa các
              điểm cùng ngày (kiểm định hoán vị); Temp_2m ERA5 đồng pha với LST trung bình
              ngày; vùng xây dựng nóng hơn vùng thực vật.
  CHIA TẬP    (--scope split) Train/Val/Test đúng mốc năm, không rò rỉ nhãn qua mốc chia,
              Test không gian không lọt vào Train/Val/Test thời gian, không trùng dòng.

Kết quả: in ra màn hình + ../data/qa_report.csv (bảng nhóm / kiểm tra / trạng thái /
chi tiết — chính là "table view" để dán vào báo cáo). Có bất kỳ FAIL nào thì thoát với
mã 1. 04_build_dataset.py tự gọi script này trước khi chia tập (tắt bằng --skip_qa).

Cách chạy:
    python qa_overlay.py                          # kiểm tra raw_dataset.csv + (nếu có) các tập đã chia
    python qa_overlay.py --scope raw --plot       # thêm ảnh chồng lớp qa_overlay_<ngày>.png
    python qa_overlay.py --min_valid_frac 0.10    # đúng ngưỡng đã dùng khi ghép cặp (mặc định: tự suy ra)
    python qa_overlay.py --gee --project YOUR_ID --qa_date 2025-02-11   # thêm kiểm tra với GEE (cần mạng)

LƯU Ý ĐỌC KẾT QUẢ: các kiểm tra thống kê chỉ là bằng chứng gián tiếp. FAIL ở nhóm
"hình thức" (schema, thời gian, vị trí, khớp cache) là lỗi thật cần sửa; WARN ở nhóm
thống kê nghĩa là "đáng nhìn lại" chứ chưa chắc là lỗi.
"""

import argparse
import importlib.util
import json
import os
import sys

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

FEATURE_COLS = prepare_features.FEATURE_COLS
QA_COLS = prepare_features.QA_COLS
ERA5_VARS = prepare_features.ERA5_VARS
CRS_UTM = prepare_features.CRS_UTM
RAW_HEADER_COLS = build_dataset.RAW_HEADER_COLS
SPLIT_KEEP_COLS = build_dataset.SPLIT_KEEP_COLS

LST_RANGE = (-10.0, 60.0)          # đề cương: loại LST ngoài -10..60 °C
INDEX_RANGE = (-1.0, 1.0)
ERA5_HARD = {                      # giới hạn vật lý CỨNG (vượt = sai đơn vị/công thức, không phải thời tiết lạ)
    "Temp_2m": (0.0, 55.0),        # °C
    "Precip": (0.0, 300.0),        # mm/giờ
    "Humidity": (0.0, 101.0),      # %
    "Wind": (0.0, 60.0),           # m/s
    "Solar": (0.0, 1500.0),        # W/m²
}

# Màu cho ảnh QA: chrome + dải tuần tự 1 màu (xanh, bước 250 -> 700) của palette tham chiếu dataviz
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, BLUE = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#2a78d6"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]


# ---------------------------------------------------------------------
# BÁO CÁO
# ---------------------------------------------------------------------

class Report:
    ICON = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌", "INFO": "ℹ️ ", "SKIP": "⏭️ "}

    def __init__(self, verbose: bool = True):
        self.rows, self.verbose = [], verbose

    def add(self, group: str, check: str, status: str, detail: str = "") -> None:
        self.rows.append({"group": group, "check": check, "status": status, "detail": detail})
        if self.verbose:
            print(f"   {self.ICON[status]} [{status}] {check}: {detail}")

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["group", "check", "status", "detail"])

    def n(self, status: str) -> int:
        return sum(1 for r in self.rows if r["status"] == status)


def _p(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, name)


def _read_if_exists(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return pd.read_csv(path)
    return None


def _manifest(data_dir: str) -> dict:
    mp = _p(data_dir, "cache_manifest.json")
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _fmt_list(items, k: int = 5) -> str:
    items = list(items)
    return str(items[:k]) + (f" ... (+{len(items) - k})" if len(items) > k else "")


# ---------------------------------------------------------------------
# KIỂM TRA TRÊN raw_dataset.csv
# ---------------------------------------------------------------------

def check_raw(rep: Report, data_dir: str, boundary: str, min_valid_frac, max_st_unc_k, n_perm: int = 50):
    G = "raw"
    path = _p(data_dir, "raw_dataset.csv")
    if not os.path.exists(path):
        rep.add(G, "Có raw_dataset.csv", "FAIL", f"không thấy {path} — chạy 04 --stage pair trước")
        return None
    raw = pd.read_csv(path)
    missing = [c for c in RAW_HEADER_COLS if c not in raw.columns]
    if missing:
        rep.add(G, "Schema", "FAIL", f"thiếu cột {missing} — file từ bản 04 CŨ, build lại bằng --rebuild_raw")
        return None
    if len(raw) == 0:
        rep.add(G, "Schema", "FAIL", "raw_dataset.csv RỖNG")
        return None
    rep.add(G, "Schema", "PASS", f"{len(raw):,} dòng, {raw['date_t'].nunique()} ngày t, "
                                 f"{raw['point_id'].nunique():,} điểm, đủ {len(RAW_HEADER_COLS)} cột")

    manifest = _manifest(data_dir)
    if max_st_unc_k is None:
        max_st_unc_k = manifest.get("lst", {}).get("max_st_unc_k", prepare_features.DEFAULT_MAX_ST_UNC_K)

    # ---- NaN
    must = FEATURE_COLS + ["LST_target"]
    na = raw[must].isna().sum()
    na = na[na > 0]
    if len(na):
        rep.add(G, "Không NaN ở feature/nhãn", "FAIL", f"{na.to_dict()} — đề cương: loại dòng NaN trước khi xuất")
    else:
        rep.add(G, "Không NaN ở feature/nhãn", "PASS", f"{len(must)} cột sạch")
    qa_na = raw[QA_COLS].isna().sum()
    qa_na = qa_na[qa_na > 0]
    if len(qa_na):
        rep.add(G, "Cột QA không NaN", "WARN", str(qa_na.to_dict()))

    # ---- trùng dòng
    dup = int(raw.duplicated(["point_id", "date_t"]).sum())
    rep.add(G, "Không trùng (point_id, date_t)", "FAIL" if dup else "PASS",
            f"{dup} dòng trùng" if dup else "mỗi điểm-ngày đúng 1 dòng")

    # ---- THỜI ĐIỂM: delta_t, thứ tự ngày
    d_t = pd.to_datetime(raw["date_t"])
    d_n = pd.to_datetime(raw["date_target"])
    diff_days = (d_n - d_t).dt.days.to_numpy()
    bad_dt = int((np.abs(diff_days - raw["delta_t"].to_numpy()) > 1e-6).sum())
    bad_order = int((diff_days <= 0).sum())
    rep.add(G, "delta_t = date_target − date_t", "FAIL" if bad_dt else "PASS",
            f"{bad_dt} dòng lệch" if bad_dt else f"khớp {len(raw):,}/{len(raw):,} dòng")
    rep.add(G, "Nhãn SAU ngày t (không rò rỉ thời gian)", "FAIL" if bad_order else "PASS",
            f"{bad_order} dòng có date_target <= date_t" if bad_order else "date_target > date_t ở mọi dòng")

    # ---- THỜI ĐIỂM: nhãn là ngày hợp lệ LIỀN KỀ (đối chiếu cache LST)
    frame = _read_if_exists(_p(data_dir, "sampling_frame.csv"))
    lst_cache = feat_cache = None
    try:
        dates_tbl, lst_cache = build_dataset.load_lst_cache(data_dir)
    except SystemExit:
        dates_tbl = None
    if dates_tbl is None or frame is None:
        rep.add(G, "Nhãn là ngày hợp lệ liền kề", "SKIP", "thiếu cache LST/khung điểm — chỉ kiểm tra được khi còn cache")
    else:
        full = build_dataset.valid_dates_table(dates_tbl, lst_cache, frame, 0.0).set_index("date")
        used = sorted(set(raw["date_t"]) | set(raw["date_target"]))
        not_in_cache = [d for d in used if d not in full.index]
        if not_in_cache:
            rep.add(G, "Nhãn là ngày hợp lệ liền kề", "FAIL",
                    f"{len(not_in_cache)} ngày trong raw KHÔNG có trong cache LST: {_fmt_list(not_in_cache)}")
        else:
            cov_used = full.loc[used, "coverage"]
            if min_valid_frac is None:
                thr, how = float(cov_used.min()), "suy ra từ ngày có độ phủ thấp nhất được dùng"
            else:
                thr, how = float(min_valid_frac), "theo --min_valid_frac"
            under = [d for d in used if full.loc[d, "coverage"] < thr - 1e-9]
            vd = full.index[full["coverage"] >= thr - 1e-9].tolist()
            pos = {d: i for i, d in enumerate(vd)}
            pairs = raw[["date_t", "date_target"]].drop_duplicates()
            skip_valid = [(a, b) for a, b in zip(pairs["date_t"], pairs["date_target"])
                          if a in pos and b in pos and pos[b] - pos[a] != 1]
            if under or skip_valid:
                msg = []
                if under:
                    msg.append(f"{len(under)} ngày có độ phủ < {thr:.3f}: {_fmt_list(under, 3)}")
                if skip_valid:
                    msg.append(f"{len(skip_valid)} cặp BỎ SÓT ngày hợp lệ ở giữa: {_fmt_list(skip_valid, 3)}")
                rep.add(G, "Nhãn là ngày hợp lệ liền kề", "FAIL",
                        "; ".join(msg) + f" (ngưỡng {how}) — nếu ghép cặp bằng --min_valid_frac khác, chạy lại QA với đúng giá trị đó")
            else:
                rep.add(G, "Nhãn là ngày hợp lệ liền kề", "PASS",
                        f"{len(pairs)} cặp (t, t') đều liền kề trong {len(vd)} ngày hợp lệ (ngưỡng độ phủ {thr:.3f}, {how})")

    # ---- ERA5 khớp giờ
    bad_era5 = int(((raw["era5_dt_min"] < 0) | (raw["era5_dt_min"] > 60.0 + 1e-6)).sum())
    rep.add(G, "ERA5 lệch giờ chụp <= 60 phút", "FAIL" if bad_era5 else "PASS",
            f"{bad_era5} dòng vượt" if bad_era5 else f"max = {raw['era5_dt_min'].max():.1f} phút, trung vị {raw['era5_dt_min'].median():.1f}")

    # ---- S2 không nhìn quá ngày nhãn
    fdates = _read_if_exists(_p(data_dir, build_dataset.FEAT_DATES))
    if fdates is None:
        rep.add(G, "S2 không nhìn quá ngày nhãn", "SKIP", "thiếu cache_feat_dates.csv")
    else:
        fwd = raw["date_t"].map(fdates.drop_duplicates("date", keep="last").set_index("date")["fwd_eff"])
        n_unknown = int(fwd.isna().sum())
        bad_fwd = int((fwd > np.maximum(raw["delta_t"] - 1, 0)).sum())
        if bad_fwd:
            rep.add(G, "S2 không nhìn quá ngày nhãn", "FAIL", f"{bad_fwd} dòng có cửa sổ S2 vươn tới/qua ngày nhãn (rò rỉ tương lai)")
        elif n_unknown:
            rep.add(G, "S2 không nhìn quá ngày nhãn", "WARN", f"{n_unknown} dòng không có bản ghi fwd_eff")
        else:
            fm = manifest.get("feat", {})
            rep.add(G, "S2 không nhìn quá ngày nhãn", "PASS",
                    f"fwd_eff <= delta_t−1 ở mọi dòng (cửa sổ S2: lùi {fm.get('s2_back_days', '?')} ngày, "
                    f"tiến yêu cầu {fm.get('s2_fwd_days_requested', '?')})")
    if (raw["s2_n_img"] < 1).any():
        rep.add(G, "Có ảnh S2 ở mọi dòng", "FAIL", f"{int((raw['s2_n_img'] < 1).sum())} dòng s2_n_img < 1")

    # ---- KHOẢNG GIÁ TRỊ
    problems = []
    for c in ["LST_t", "LST_target"]:
        n = int(((raw[c] < LST_RANGE[0]) | (raw[c] > LST_RANGE[1])).sum())
        if n:
            problems.append(f"{c} ngoài {LST_RANGE}: {n}")
    for c in ["NDVI", "NDBI", "NDWI"]:
        n = int(((raw[c] < INDEX_RANGE[0]) | (raw[c] > INDEX_RANGE[1])).sum())
        if n:
            problems.append(f"{c} ngoài [-1,1]: {n}")
    for c, (lo, hi) in ERA5_HARD.items():
        n = int(((raw[c] < lo) | (raw[c] > hi)).sum())
        if n:
            problems.append(f"{c} ngoài [{lo},{hi}]: {n} (nghi sai đơn vị/công thức)")
    for c in ["st_unc_t", "st_unc_target"]:
        n = int((raw[c] > max_st_unc_k + 1e-6).sum())
        if n:
            problems.append(f"{c} > {max_st_unc_k} K: {n}")
    rep.add(G, "Khoảng giá trị vật lý", "FAIL" if problems else "PASS",
            "; ".join(problems) if problems else
            f"LST {raw['LST_t'].min():.1f}..{raw['LST_t'].max():.1f} °C | NDVI {raw['NDVI'].min():.2f}..{raw['NDVI'].max():.2f} | "
            f"Temp_2m {raw['Temp_2m'].min():.1f}..{raw['Temp_2m'].max():.1f} °C | ST_QA <= {max_st_unc_k} K")

    # ---- VỊ TRÍ + HỆ TỌA ĐỘ
    _check_positions(rep, raw, frame, boundary)

    # ---- KHỚP CACHE (không ghép sai khi build)
    _check_cache_consistency(rep, raw, lst_cache, data_dir)

    # ---- CHUỖI t -> t' nhất quán
    a = raw[["point_id", "date_t", "LST_t"]].rename(columns={"date_t": "d", "LST_t": "v_as_t"})
    b = raw[["point_id", "date_target", "LST_target"]].rename(columns={"date_target": "d", "LST_target": "v_as_target"})
    chain = a.merge(b.drop_duplicates(["point_id", "d"]), on=["point_id", "d"])
    if len(chain) == 0:
        rep.add(G, "Chuỗi t→t'→t'' nhất quán", "SKIP", "không có điểm nào xuất hiện ở 2 cặp liên tiếp")
    else:
        bad = int(((chain["v_as_t"] - chain["v_as_target"]).abs() > 1e-4).sum())
        rep.add(G, "Chuỗi t→t'→t'' nhất quán", "FAIL" if bad else "PASS",
                f"{bad}/{len(chain):,} (điểm, ngày) có LST_target của cặp trước ≠ LST_t của cặp sau" if bad
                else f"{len(chain):,} (điểm, ngày) LST_target(cặp trước) == LST_t(cặp sau)")

    # ---- phân phối delta_t + phủ theo đơn vị
    vc = raw["delta_t"].round().astype(int).value_counts().sort_index()
    share1 = float((raw["delta_t"] == 1).mean())
    rep.add(G, "Phân phối delta_t", "WARN" if share1 > 0.20 else "INFO",
            f"min={raw['delta_t'].min():.0f} trung vị={raw['delta_t'].median():.0f} max={raw['delta_t'].max():.0f}; "
            f"delta_t=1 chiếm {share1 * 100:.1f}% (cặp 2 path liền kề)" + (" — nhiều bất thường, xem bảng quét ngưỡng ở stage pair" if share1 > 0.2 else "")
            + f" | đếm: {vc.head(8).to_dict()}")
    if frame is not None:
        units = set(frame["ADM2_NAME"].unique())
        per_d = raw.groupby("ADM2_NAME").size().reindex(sorted(units), fill_value=0)
        zero = per_d[per_d == 0].index.tolist()
        rep.add(G, "Mỗi đơn vị có dữ liệu", "WARN" if zero else "PASS",
                (f"{len(zero)}/{len(units)} đơn vị 0 dòng: {_fmt_list(zero)}" if zero else
                 f"{len(units)}/{len(units)} đơn vị có dòng (min={per_d.min():,} trung vị={int(per_d.median()):,} max={per_d.max():,})"))

    # ---- BẰNG CHỨNG THỐNG KÊ
    _check_statistics(rep, raw, n_perm)
    return raw


def _check_positions(rep: Report, raw: pd.DataFrame, frame, boundary: str) -> None:
    G = "vị trí / hệ tọa độ"
    try:
        import geopandas as gpd
    except ImportError:
        rep.add(G, "Vị trí", "SKIP", "chưa cài geopandas")
        return
    if not os.path.exists(boundary):
        rep.add(G, "Vị trí", "SKIP", f"không thấy ranh giới {boundary}")
        return

    d4326 = prepare_features.load_districts(boundary)
    minx, miny, maxx, maxy = d4326.total_bounds
    m = 0.05
    pts = raw.drop_duplicates("point_id")[["point_id", "lon", "lat", "district_idx", "ADM1_NAME", "ADM2_NAME"]].reset_index(drop=True)

    out_box = ~(pts["lon"].between(minx - m, maxx + m) & pts["lat"].between(miny - m, maxy + m))
    swapped = pts["lat"].between(minx - m, maxx + m) & pts["lon"].between(miny - m, maxy + m)
    if out_box.any():
        rep.add(G, "lon/lat trong khung ranh giới", "FAIL",
                f"{int(out_box.sum())} điểm ngoài khung lon[{minx:.2f},{maxx:.2f}] lat[{miny:.2f},{maxy:.2f}]"
                + (" — nghi ĐẢO lon/lat" if swapped.any() else ""))
    else:
        rep.add(G, "lon/lat trong khung ranh giới", "PASS",
                f"{len(pts):,} điểm trong lon[{minx:.2f},{maxx:.2f}] lat[{miny:.2f},{maxy:.2f}]")

    # tọa độ khớp khung điểm gốc
    if frame is not None:
        mm = pts.merge(frame[["point_id", "lon", "lat", "district_idx"]], on="point_id", how="left", suffixes=("", "_f"))
        n_unk = int(mm["lon_f"].isna().sum())
        drift = (mm["lon"] - mm["lon_f"]).abs().gt(1e-6) | (mm["lat"] - mm["lat_f"]).abs().gt(1e-6) | (mm["district_idx"] != mm["district_idx_f"])
        n_dr = int(drift.sum()) - n_unk
        rep.add(G, "Tọa độ khớp sampling_frame.csv", "FAIL" if (n_unk or n_dr > 0) else "PASS",
                f"{n_unk} point_id không có trong khung, {max(n_dr, 0)} điểm lệch tọa độ/đơn vị" if (n_unk or n_dr > 0)
                else f"{len(pts):,}/{len(pts):,} điểm trùng khung (sai số < 1e-6°)")

    # điểm nằm trong đúng polygon (UTM 48N)
    du = d4326.to_crs(CRS_UTM)
    poly = du.set_index("district_idx")["geometry"]
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326").to_crs(CRS_UTM)
    unknown = ~g["district_idx"].isin(poly.index)
    n_out, max_out = int(unknown.sum()), 0.0
    try:
        import shapely

        def contains(pg, xs, ys):
            return shapely.contains_xy(pg, xs, ys)
    except (ImportError, AttributeError):
        def contains(pg, xs, ys):
            return gpd.GeoSeries(gpd.points_from_xy(xs, ys)).within(pg).to_numpy()
    for idx, grp in g[~unknown].groupby("district_idx"):
        pg = poly.loc[idx]
        inside = np.asarray(contains(pg, grp.geometry.x.to_numpy(), grp.geometry.y.to_numpy()))
        if (~inside).any():
            dist = grp.geometry[~inside].distance(pg)
            far = dist[dist > 1.0]
            n_out += len(far)
            max_out = max(max_out, float(far.max()) if len(far) else 0.0)
    rep.add(G, "Điểm nằm trong đúng quận/huyện được gán (UTM 48N)", "FAIL" if n_out else "PASS",
            f"{n_out} điểm nằm ngoài polygon của đơn vị được gán (xa nhất {max_out:.0f} m) — ranh giới đã đổi sau khi tạo khung? "
            f"chạy lại 04 --rebuild_frame --reset_cache" if n_out else f"{len(pts):,}/{len(pts):,} điểm nằm trong polygon của đơn vị mình")
    nm = d4326.set_index("district_idx")
    bad_name = int(((nm["ADM2_NAME"].reindex(pts["district_idx"]).to_numpy() != pts["ADM2_NAME"].to_numpy())
                    | (nm["ADM1_NAME"].reindex(pts["district_idx"]).to_numpy() != pts["ADM1_NAME"].to_numpy())).sum())
    rep.add(G, "Tên đơn vị khớp ranh giới hiện tại", "FAIL" if bad_name else "PASS",
            f"{bad_name} điểm có (ADM1, ADM2) khác ranh giới hiện tại — file ranh giới đã thay đổi" if bad_name
            else "district_idx ↔ (ADM1_NAME, ADM2_NAME) khớp")


def _check_cache_consistency(rep: Report, raw: pd.DataFrame, lst_cache, data_dir: str, n_sample: int = 3000) -> None:
    G = "khớp cache"
    feat_cache = _read_if_exists(_p(data_dir, build_dataset.FEAT_CACHE))
    if lst_cache is None or feat_cache is None:
        rep.add(G, "Dòng ghép khớp cache", "SKIP", "thiếu cache LST/đặc trưng")
        return
    s = raw.sample(min(n_sample, len(raw)), random_state=0)
    tol = 1e-4
    msgs, tot = [], 0

    ct = lst_cache.drop_duplicates(["date", "point_id"], keep="last").set_index(["date", "point_id"])
    for date_col, val_col, cache_col, label in [("date_t", "LST_t", "LST_C", "LST_t"), ("date_target", "LST_target", "LST_C", "LST_target"),
                                                  ("date_t", "st_unc_t", "ST_UNC_K", "st_unc_t"), ("date_target", "st_unc_target", "ST_UNC_K", "st_unc_target")]:
        key = pd.MultiIndex.from_arrays([s[date_col], s["point_id"]])
        got = ct[cache_col].reindex(key).to_numpy()
        bad = int((np.isnan(got) | (np.abs(got - s[val_col].to_numpy()) > tol)).sum())
        tot += len(s)
        if bad:
            msgs.append(f"{label}: {bad}/{len(s)} lệch/không có trong cache")

    cf = feat_cache.drop_duplicates(["date", "point_id"], keep="last").set_index(["date", "point_id"])
    key = pd.MultiIndex.from_arrays([s["date_t"], s["point_id"]])
    for c in ["NDVI", "NDBI", "NDWI", "s2_n_img"] + ERA5_VARS + ["era5_dt_min"]:
        got = cf[c].reindex(key).to_numpy()
        bad = int((np.isnan(got) | (np.abs(got - s[c].to_numpy()) > tol)).sum())
        tot += len(s)
        if bad:
            msgs.append(f"{c}: {bad}/{len(s)} lệch/không có trong cache")
    rep.add(G, "Dòng ghép khớp cache (mẫu ngẫu nhiên)", "FAIL" if msgs else "PASS",
            "; ".join(msgs) + " — ghép sai ngày/điểm khi build" if msgs
            else f"{len(s):,} dòng × (LST_t, LST_target, ST_QA, NDVI, NDBI, NDWI, ERA5…) khớp cache từng giá trị")


def _pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _check_statistics(rep: Report, raw: pd.DataFrame, n_perm: int) -> None:
    G = "bằng chứng thống kê"
    rng = np.random.default_rng(0)

    # (a) NDVI–LST_t: ngược chiều (thực vật mát hơn) — kiểm tra S2 chồng đúng lên Landsat
    r_nv, r_obs, r_p99 = [], [], []
    for _, g in raw.groupby("date_t"):
        if len(g) < 30:
            continue
        r_nv.append(_pearson(g["NDVI"], g["LST_t"]))
        x, y = g["LST_t"].to_numpy(float), g["LST_target"].to_numpy(float)
        if x.std() == 0 or y.std() == 0:
            continue
        xs, ys = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
        r_obs.append(float((xs * ys).mean()))
        perms = [float((xs * rng.permutation(ys)).mean()) for _ in range(n_perm)]
        r_p99.append(float(np.percentile(perms, 99)))
    r_nv = np.array([r for r in r_nv if not np.isnan(r)])
    if len(r_nv) == 0:
        rep.add(G, "NDVI–LST_t tương quan âm", "SKIP", "không ngày nào đủ >= 30 dòng")
    else:
        med = float(np.median(r_nv))
        neg = float((r_nv < 0).mean())
        rep.add(G, "NDVI–LST_t tương quan âm (mỗi ngày)", "PASS" if med < -0.1 else "WARN",
                f"trung vị r = {med:.2f}, {neg * 100:.0f}% số ngày r<0 ({len(r_nv)} ngày)"
                + ("" if med < -0.1 else " — kỳ vọng ÂM rõ (thực vật mát hơn); nếu ~0: nghi S2 và Landsat chồng lệch vị trí/ngày"))

    # (b) LST_t–LST_target vs ghép ngẫu nhiên giữa các điểm CÙNG ngày
    if r_obs:
        med_obs, med_p99 = float(np.median(r_obs)), float(np.median(r_p99))
        if med_obs <= med_p99 and med_obs < 0.1:
            st, why = "FAIL", " — không hơn ghép ngẫu nhiên: nghi nhãn bị ghép lệch VỊ TRÍ so với đặc trưng"
        elif med_obs < 0.3:
            st, why = "WARN", " — khá thấp; kiểm tra thêm (dữ liệu thưa/nhiều mây?)"
        else:
            st, why = "PASS", ""
        rep.add(G, "LST_t–LST_target hơn hẳn ghép ngẫu nhiên", st,
                f"trung vị r thật = {med_obs:.2f} so với r ghép ngẫu nhiên (bách phân vị 99) = {med_p99:.2f} ({len(r_obs)} ngày){why}")

    # (c) ERA5 đồng pha với LST trung bình ngày. Chỉ xét tương quan thô thì chu kỳ mùa làm r luôn cao dù
    #     lệch vài ngày -> báo thêm tương quan của PHẦN DƯ sau khi bỏ chu kỳ mùa (thời tiết bất thường
    #     của từng ngày: ngày nóng ở ERA5 phải là ngày nóng ở LST) và so với lệch 1 kỳ ảnh.
    by_d = raw.groupby("date_t").agg(temp=("Temp_2m", "mean"), lst=("LST_t", "mean")).sort_index()
    if len(by_d) >= 24 and by_d["temp"].std() > 0 and by_d["lst"].std() > 0:
        doy = pd.to_datetime(by_d.index).dayofyear.to_numpy() / 365.25 * 2 * np.pi
        X = np.column_stack([np.ones(len(doy)), np.sin(doy), np.cos(doy), np.sin(2 * doy), np.cos(2 * doy)])

        def _resid(v):
            v = v.to_numpy(float)
            return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]

        r0 = _pearson(by_d["temp"], by_d["lst"])
        ra, rb = _resid(by_d["temp"]), _resid(by_d["lst"])
        r_an = _pearson(ra, rb)
        r_an_shift = _pearson(ra[1:], rb[:-1])
        ok = r0 >= 0.3 and r_an >= 0.1
        rep.add(G, "ERA5 Temp_2m đồng pha với LST trung bình ngày", "PASS" if ok else "WARN",
                f"r thô = {r0:.2f}; r phần dư sau khi bỏ chu kỳ mùa = {r_an:.2f} (lệch 1 kỳ ảnh: {r_an_shift:.2f}; {len(by_d)} ngày)"
                + ("" if ok else " — kỳ vọng DƯƠNG ở cả hai; phần dư ~0 gợi ý ERA5 ghép sai ngày/giờ (hoặc dữ liệu quá nhiễu — "
                                 "đối chiếu thêm bằng 'qa_overlay.py --gee')"))
    else:
        rep.add(G, "ERA5 Temp_2m đồng pha với LST trung bình ngày", "SKIP", "ERA5 không đổi theo ngày hoặc < 24 ngày")

    # (d) xây dựng nóng hơn thực vật
    surf = pd.Series(prepare_features.classify_surface(raw), index=raw.index)
    n_b, n_v = int((surf == "Xây dựng").sum()), int((surf == "Thực vật").sum())
    if n_b >= 50 and n_v >= 50:
        diff = float(raw.loc[surf == "Xây dựng", "LST_t"].median() - raw.loc[surf == "Thực vật", "LST_t"].median())
        rep.add(G, "Vùng xây dựng nóng hơn vùng thực vật", "PASS" if diff > 0.3 else "WARN",
                f"LST_t trung vị: xây dựng − thực vật = {diff:+.2f} °C (n={n_b:,} / {n_v:,})"
                + ("" if diff > 0.3 else " — kỳ vọng DƯƠNG; nghi chỉ số S2 không khớp vị trí với LST"))
    else:
        rep.add(G, "Vùng xây dựng nóng hơn vùng thực vật", "SKIP", f"quá ít mẫu (xây dựng={n_b}, thực vật={n_v})")

    # (e) bước nhảy bất thường giữa t và nhãn
    jump = (raw["LST_target"] - raw["LST_t"]).abs()
    share = float((jump > 15).mean())
    rep.add(G, "Bước nhảy |LST_target − LST_t| > 15 °C", "WARN" if share > 0.02 else "PASS",
            f"{share * 100:.2f}% dòng (trung vị |Δ| = {jump.median():.1f} °C)"
            + (" — nhiều: nghi mây/bóng mây lọt mặt nạ ở 1 trong 2 ngày" if share > 0.02 else ""))
    if "moran_outlier" in raw.columns:
        rep.add(G, "Cờ Moran cục bộ", "INFO", f"{float(raw['moran_outlier'].astype(bool).mean()) * 100:.2f}% dòng bị cờ (chỉ báo cáo)")


# ---------------------------------------------------------------------
# KIỂM TRA CÁC TẬP ĐÃ CHIA
# ---------------------------------------------------------------------

def check_split(rep: Report, data_dir: str, spatial_districts, raw_len=None) -> None:
    G = "chia tập"
    names = {"train": "train.csv", "val": "val.csv", "test_temporal": "test_temporal.csv", "test_spatial": "test_spatial.csv"}
    sets = {}
    for k, fn in names.items():
        path = _p(data_dir, fn)
        if not os.path.exists(path):
            rep.add(G, f"Có {fn}", "FAIL" if k in ("train", "val") else "WARN", f"không thấy {path} — chạy 04 --stage split")
            continue
        try:
            sets[k] = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            sets[k] = pd.DataFrame(columns=SPLIT_KEEP_COLS)
    if "train" not in sets or "val" not in sets:
        return

    for k, d in sets.items():
        miss = [c for c in FEATURE_COLS + ["LST_target", "date_t", "date_target", "point_id", "ADM2_NAME"] if c not in d.columns]
        if miss:
            rep.add(G, f"{names[k]} đủ cột", "FAIL", f"thiếu {miss}")
            return
    sizes = {k: len(d) for k, d in sets.items()}
    for k in ("train", "val"):
        if sizes[k] == 0:
            rep.add(G, f"{names[k]} không rỗng", "FAIL", "RỖNG — không huấn luyện được")
    if sizes.get("test_temporal", 0) == 0:
        rep.add(G, "test_temporal.csv không rỗng", "WARN", "RỖNG — chưa có dữ liệu ngày >= 2025?")
    if sizes.get("test_spatial", 0) == 0:
        rep.add(G, "test_spatial.csv không rỗng", "WARN", "RỖNG — không đơn vị nào khớp --spatial_districts?")
    rep.add(G, "Kích thước các tập", "INFO", ", ".join(f"{k}={v:,}" for k, v in sizes.items()))
    if sizes["train"] == 0 or sizes["val"] == 0:
        return

    yr = lambda s: pd.to_datetime(s).dt.year  # noqa: E731
    tr, va = sets["train"], sets["val"]
    te = sets.get("test_temporal", pd.DataFrame(columns=tr.columns))
    sp = sets.get("test_spatial", pd.DataFrame(columns=tr.columns))
    TE, VY, TS = build_dataset.TRAIN_END_YEAR, build_dataset.VAL_YEAR, build_dataset.TEST_START_YEAR

    bad = []
    if (yr(tr["date_target"]) > TE).any():
        bad.append(f"Train có nhãn sau {TE}: {int((yr(tr['date_target']) > TE).sum())} dòng")
    if ((yr(va["date_t"]) != VY) | (yr(va["date_target"]) != VY)).any():
        bad.append(f"Val có ngày ngoài {VY}: {int(((yr(va['date_t']) != VY) | (yr(va['date_target']) != VY)).sum())} dòng")
    if len(te) and (yr(te["date_t"]) < TS).any():
        bad.append(f"Test có ngày t trước {TS}: {int((yr(te['date_t']) < TS).sum())} dòng")
    rep.add(G, f"Mốc năm: Train nhãn <= {TE}, Val = {VY}, Test >= {TS}", "FAIL" if bad else "PASS",
            "; ".join(bad) if bad else "đúng mốc ở cả 3 tập (xét cả ngày t lẫn ngày nhãn)")

    leak = []
    if pd.to_datetime(tr["date_target"]).max() >= pd.to_datetime(va["date_t"]).min():
        leak.append("nhãn Train chạm tới ngày t của Val")
    if len(te) and pd.to_datetime(va["date_target"]).max() >= pd.to_datetime(te["date_t"]).min():
        leak.append("nhãn Val chạm tới ngày t của Test")
    rep.add(G, "Không rò rỉ nhãn qua mốc chia", "FAIL" if leak else "PASS",
            "; ".join(leak) if leak else
            f"Train nhãn tối đa {pd.to_datetime(tr['date_target']).max().date()} < Val t tối thiểu {pd.to_datetime(va['date_t']).min().date()}")

    # Test không gian
    if len(sp):
        others = set(tr["ADM2_NAME"]) | set(va["ADM2_NAME"]) | set(te["ADM2_NAME"])
        inter = sorted(set(sp["ADM2_NAME"]) & others)
        pid_inter = set(sp["point_id"]) & (set(tr["point_id"]) | set(va["point_id"]) | set(te["point_id"]))
        rep.add(G, "Test không gian tách hẳn khỏi Train/Val/Test thời gian", "FAIL" if (inter or pid_inter) else "PASS",
                f"đơn vị lọt cả 2 phía: {_fmt_list(inter)}; điểm trùng: {len(pid_inter)}" if (inter or pid_inter)
                else f"{sp['ADM2_NAME'].nunique()} đơn vị giữ riêng: {_fmt_list(sorted(sp['ADM2_NAME'].unique()), 6)}")
        if spatial_districts:
            keys = {prepare_features.normalize_vn_name(x) for x in spatial_districts}
            found = {prepare_features.normalize_vn_name(x) for x in sp["ADM2_NAME"].unique()}
            miss = [x for x in spatial_districts if prepare_features.normalize_vn_name(x) not in found]
            if miss:
                rep.add(G, "Đơn vị Test không gian khớp danh sách yêu cầu", "WARN", f"không có dữ liệu cho: {miss}")
            extra = found - keys
            if extra:
                rep.add(G, "Test không gian không chứa đơn vị ngoài danh sách", "FAIL", f"thừa: {sorted(extra)}")

    # trùng dòng giữa các tập
    keyset = {}
    dup_pairs = []
    for k, d in sets.items():
        ks = set(zip(d["point_id"], d["date_t"]))
        for k2, s2 in keyset.items():
            n = len(ks & s2)
            if n:
                dup_pairs.append(f"{k}∩{k2}={n}")
        keyset[k] = ks
    rep.add(G, "Không trùng (point_id, date_t) giữa các tập", "FAIL" if dup_pairs else "PASS",
            ", ".join(dup_pairs) if dup_pairs else "mỗi dòng thuộc đúng 1 tập")

    n_split = sum(sizes.values())
    if raw_len is not None:
        if n_split > raw_len:
            rep.add(G, "Cân đối số dòng", "FAIL", f"tổng các tập {n_split:,} > raw {raw_len:,}")
        else:
            rep.add(G, "Cân đối số dòng", "INFO",
                    f"raw {raw_len:,} → các tập {n_split:,}; bị loại (purge qua mốc chia / cờ Moran nếu bật) {raw_len - n_split:,} "
                    f"({(raw_len - n_split) / max(raw_len, 1) * 100:.2f}%)")
    # lệch phân phối nhãn
    m_tr = float(tr["LST_target"].mean())
    notes = []
    for k, d in (("val", va), ("test_temporal", te), ("test_spatial", sp)):
        if len(d):
            notes.append(f"{k}: {d['LST_target'].mean():.1f}°C / delta_t {d['delta_t'].mean():.1f}")
    shift = [k for k, d in (("val", va), ("test_temporal", te)) if len(d) and abs(float(d["LST_target"].mean()) - m_tr) > 3.0]
    rep.add(G, "Lệch phân phối nhãn giữa các tập", "WARN" if shift else "INFO",
            f"train {m_tr:.1f}°C / delta_t {tr['delta_t'].mean():.1f} | " + " | ".join(notes)
            + (f" — {shift} lệch > 3 °C so với Train (do mùa/năm ít mẫu? ghi vào báo cáo)" if shift else ""))


# ---------------------------------------------------------------------
# ẢNH CHỒNG LỚP (offline) — để NHÌN bằng mắt
# ---------------------------------------------------------------------

def make_overlay_plot(raw: pd.DataFrame, frame, boundary: str, date: str, out_png: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    districts = prepare_features.load_districts(boundary)
    day = raw[raw["date_t"] == date]
    if len(day) == 0:
        raise SystemExit(f"❌ Không có dòng nào của ngày {date} trong raw_dataset.csv")

    plt.rcParams.update({"axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor=SURFACE)
    lat0 = float(np.deg2rad(day["lat"].mean()))

    def _map(ax, title):
        districts.boundary.plot(ax=ax, color=AXIS, linewidth=0.5)
        ax.set_aspect(1 / np.cos(lat0))
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Kinh độ (°E)")
        ax.set_ylabel("Vĩ độ (°N)")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    # A. vị trí: có dữ liệu (xanh) / bị loại (xám)
    ax = axes[0, 0]
    _map(ax, f"Vị trí điểm ngày t = {date}")
    if frame is not None:
        miss = frame[~frame["point_id"].isin(day["point_id"])]
        ax.scatter(miss["lon"], miss["lat"], s=5, color=MUTED, alpha=0.55, linewidths=0, label=f"Bị loại/thiếu (n={len(miss):,})")
    ax.scatter(day["lon"], day["lat"], s=7, color=BLUE, alpha=0.85, linewidths=0, label=f"Có đủ dữ liệu (n={len(day):,})")
    leg = ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=AXIS, markerscale=2.2)
    for t in leg.get_texts():
        t.set_color(INK2)

    # B. LST_t
    ax = axes[0, 1]
    _map(ax, "LST_t (°C) tại các điểm có dữ liệu — đậm = nóng hơn")
    lo, hi = np.percentile(day["LST_t"], [2, 98])
    sc = ax.scatter(day["lon"], day["lat"], c=day["LST_t"], s=9, cmap=cmap, vmin=lo, vmax=hi, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("LST_t (°C)", color=INK2)
    cb.outline.set_edgecolor(AXIS)

    # C/D. phân tán
    def _scatter(ax, x, y, xl, yl, title, one2one=False):
        ax.scatter(x, y, s=8, color=BLUE, alpha=0.45, linewidths=0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        r = _pearson(x, y)
        if one2one:
            lim = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
            ax.plot(lim, lim, color=INK2, linewidth=1.0)
        elif len(x) > 2 and np.std(x) > 0:
            k, b = np.polyfit(x, y, 1)
            xs = np.array([np.min(x), np.max(x)])
            ax.plot(xs, k * xs + b, color=INK, linewidth=1.2)
        ax.set_title(f"{title}  (r = {r:.2f}, n = {len(x):,})", loc="left", fontsize=11, color=INK)

    _scatter(axes[1, 0], day["NDVI"].to_numpy(), day["LST_t"].to_numpy(), "NDVI (S2)", "LST_t (°C)",
             "NDVI – LST_t: kỳ vọng ÂM")
    _scatter(axes[1, 1], day["LST_t"].to_numpy(), day["LST_target"].to_numpy(), "LST_t (°C)", "LST_target (°C)",
             "LST_t – LST_target: kỳ vọng DƯƠNG, bám đường 1:1", one2one=True)
    fig.suptitle("Overlay QA — kiểm tra bằng mắt: vị trí, chồng lớp S2/Landsat, nhãn", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------
# KIỂM TRA VỚI GEE (tùy chọn) — projection gốc + trích lại điểm so với cache
# ---------------------------------------------------------------------

def _proj_info(image, band: str) -> dict:
    p = image.select(band).projection()
    info = prepare_features.ee_retry(lambda: p.getInfo(), label="projection")
    scale = prepare_features.ee_retry(lambda: p.nominalScale().getInfo(), label="nominalScale")
    return {"crs": info.get("crs"), "scale_m": float(scale)}


def run_gee_qa(rep: Report, data_dir: str, boundary: str, project: str, qa_date: str, n_check: int, thumbs: bool = True) -> None:
    import ee
    from datetime import datetime, timedelta

    G = "GEE"
    raw = _read_if_exists(_p(data_dir, "raw_dataset.csv"))
    if raw is None:
        rep.add(G, "Kiểm tra GEE", "SKIP", "cần raw_dataset.csv")
        return
    day = raw[raw["date_t"] == qa_date]
    if len(day) == 0:
        rep.add(G, "Kiểm tra GEE", "SKIP", f"ngày {qa_date} không có dòng trong raw_dataset.csv")
        return
    prepare_features.init_ee(project)
    roi = prepare_features.load_boundary_roi(boundary)
    manifest = _manifest(data_dir)
    lp = manifest.get("lst", {})
    mask_kwargs = dict(zscore_threshold=lp.get("zscore_threshold", 3.0), focal_radius_px=lp.get("focal_radius_px", 2),
                       apply_zscore=True, max_st_unc_k=lp.get("max_st_unc_k", prepare_features.DEFAULT_MAX_ST_UNC_K))
    s2_back = manifest.get("feat", {}).get("s2_back_days", prepare_features.S2_BACK_DAYS)

    # 1) projection / độ phân giải GỐC của từng nguồn
    d1 = (datetime.strptime(qa_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    d0 = (datetime.strptime(qa_date, "%Y-%m-%d") - timedelta(days=s2_back)).strftime("%Y-%m-%d")
    srcs = [
        ("Landsat 8/9 ST_B10", lambda: ee.Image(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
                                                  .filterBounds(roi).filterDate(qa_date, d1).first()), "ST_B10", 30.0),
        ("Sentinel-2 B8 (10 m)", lambda: ee.Image(ee.ImageCollection(prepare_features.S2_SR_COLLECTION).filterBounds(roi)
                                                    .filterDate(d0, d1).first()), "B8", 10.0),
        ("Sentinel-2 B11 (20 m)", lambda: ee.Image(ee.ImageCollection(prepare_features.S2_SR_COLLECTION).filterBounds(roi)
                                                     .filterDate(d0, d1).first()), "B11", 20.0),
        ("ERA5-Land temperature_2m", lambda: ee.Image(ee.ImageCollection(prepare_features.ERA5_COLLECTION).filterBounds(roi)
                                                        .filterDate(qa_date, d1).first()), "temperature_2m", 11132.0),
    ]
    for name, getter, band, expect in srcs:
        try:
            info = _proj_info(getter(), band)
            ok = abs(info["scale_m"] - expect) / expect < 0.15
            rep.add(G, f"Nguồn gốc: {name}", "PASS" if ok else "WARN",
                    f"crs={info['crs']}, độ phân giải ≈ {info['scale_m']:.0f} m (kỳ vọng ≈ {expect:.0f} m). "
                    f"GEE tự chiếu lại về EPSG:32648/30 m khi lấy mẫu — chênh crs gốc không phải lỗi")
        except Exception as e:  # noqa: BLE001
            rep.add(G, f"Nguồn gốc: {name}", "WARN", f"không đọc được ({str(e)[:90]})")

    # 2) TRÍCH LẠI các điểm của ngày qa_date và so với cache/raw
    sample = day.sample(min(n_check, len(day)), random_state=1)
    pts = sample[["point_id", "lon", "lat"]].reset_index(drop=True)
    try:
        lst_img = build_dataset._lst_image_for_date(roi, qa_date, mask_kwargs)
        re_lst = prepare_features.reduce_at_points(lst_img, pts, ["LST_C", "ST_UNC_K"], label=qa_date)
        m = sample.merge(re_lst, on="point_id", how="left")
        d = (m["LST_t"] - m["LST_C"]).abs()
        n_nan = int(d.isna().sum())
        mx = float(d.max()) if d.notna().any() else float("nan")
        st = "PASS" if (n_nan == 0 and mx <= 0.05) else ("WARN" if n_nan < len(m) else "FAIL")
        rep.add(G, f"Trích lại LST_t ngày {qa_date} (n={len(m)}) so với raw", st,
                f"|Δ| lớn nhất = {mx:.3f} °C, {n_nan} điểm không trích lại được"
                + ("" if st == "PASS" else " — lệch: kiểm tra tham số mặt nạ trong cache_manifest.json, hoặc GEE đã xử lý lại ảnh gốc"))
    except Exception as e:  # noqa: BLE001
        rep.add(G, "Trích lại LST_t", "WARN", f"lỗi gọi GEE: {str(e)[:120]}")

    try:
        dates = _read_if_exists(_p(data_dir, build_dataset.LST_DATES))
        t_ms = int(dates.drop_duplicates("date", keep="last").set_index("date").loc[qa_date, "t_utc_ms"])
        fd = _read_if_exists(_p(data_dir, build_dataset.FEAT_DATES))
        fwd = int(fd.drop_duplicates("date", keep="last").set_index("date").loc[qa_date, "fwd_eff"]) if fd is not None else 0
        s2 = prepare_features.get_sentinel2_features(roi, qa_date, s2_back, fwd)
        era5 = prepare_features.get_era5_features(roi, ee.Date(t_ms))
        bands = ["NDVI", "NDBI", "NDWI"] + ERA5_VARS + ["era5_dt_min"]
        re_f = prepare_features.reduce_at_points(ee.Image.cat([s2, era5]), pts, bands, label=qa_date)
        m = sample.merge(re_f, on="point_id", how="left", suffixes=("", "_re"))
        worst = {}
        for c in bands:
            dd = (m[c] - m[c + "_re"]).abs()
            worst[c] = float(dd.max()) if dd.notna().any() else float("nan")
        bad = {c: v for c, v in worst.items() if np.isnan(v) or v > (0.01 if c in ("NDVI", "NDBI", "NDWI") else 0.05)}
        rep.add(G, f"Trích lại S2 + ERA5 ngày {qa_date} (n={len(m)}) so với raw", "PASS" if not bad else "WARN",
                "|Δ| lớn nhất: " + ", ".join(f"{c}={v:.3g}" for c, v in worst.items())
                + ("" if not bad else f" — vượt ngưỡng ở {list(bad)}: kiểm tra cửa sổ S2/ERA5 hoặc ảnh gốc đã được xử lý lại"))
    except Exception as e:  # noqa: BLE001
        rep.add(G, "Trích lại S2 + ERA5", "WARN", f"lỗi gọi GEE: {str(e)[:120]}")

    # 3) ảnh thumbnail để nhìn bằng mắt (cùng dải màu tuần tự 1 màu với ảnh offline)
    if thumbs:
        try:
            import urllib.request

            palette = [c.lstrip("#") for c in BLUE_RAMP]
            lst_vis = build_dataset._lst_image_for_date(roi, qa_date, mask_kwargs).select("LST_C")
            ndvi = prepare_features.get_sentinel2_features(roi, qa_date, s2_back, 0).select("NDVI")
            for nm, im, lo, hi in (("LST", lst_vis, 20, 50), ("NDVI", ndvi, -0.2, 0.9)):
                url = im.getThumbURL({"min": lo, "max": hi, "palette": palette, "region": roi, "dimensions": 900, "format": "png"})
                out = _p(data_dir, f"qa_thumb_{nm}_{qa_date}.png")
                try:
                    urllib.request.urlretrieve(url, out)
                    rep.add(G, f"Thumbnail {nm}", "INFO", f"đã lưu {out} — mở cạnh qa_overlay_{qa_date}.png để so vị trí")
                except Exception:  # noqa: BLE001
                    rep.add(G, f"Thumbnail {nm}", "INFO", f"tải ảnh bằng URL này (mở trình duyệt): {url}")
        except Exception as e:  # noqa: BLE001
            rep.add(G, "Thumbnail", "SKIP", f"không tạo được ({str(e)[:100]})")


# ---------------------------------------------------------------------
# GIAO DIỆN GỌI TỪ 04 VÀ CLI
# ---------------------------------------------------------------------

def _busiest_date(raw: pd.DataFrame) -> str:
    return raw["date_t"].value_counts().idxmax()


def run_qa(data_dir: str, boundary: str, scope: str = "all", min_valid_frac=None, max_st_unc_k=None,
           spatial_districts=None, plot: bool = False, qa_date=None, gee_project=None, n_check: int = 60,
           n_perm: int = 50, verbose: bool = True) -> pd.DataFrame:
    """Chạy các kiểm tra, ghi qa_report.csv, trả về bảng báo cáo (cột status có PASS/WARN/FAIL/INFO/SKIP)."""
    rep = Report(verbose)
    raw = None
    if scope in ("raw", "all"):
        if verbose:
            print("\n[QA] Kiểm tra raw_dataset.csv (vị trí / thời điểm / hệ tọa độ / giá trị)...")
        raw = check_raw(rep, data_dir, boundary, min_valid_frac, max_st_unc_k, n_perm=n_perm)
    if scope in ("split", "all"):
        has_split = os.path.exists(_p(data_dir, "train.csv"))
        if has_split or scope == "split":
            if verbose:
                print("\n[QA] Kiểm tra các tập đã chia...")
            raw_len = None
            rp = _p(data_dir, "raw_dataset.csv")
            if os.path.exists(rp):
                raw_len = int(sum(1 for _ in open(rp, "rb")) - 1)  # đếm dòng, không nạp lại cả file
            check_split(rep, data_dir, spatial_districts, raw_len=raw_len)
        elif verbose:
            print("\n[QA] Chưa có train.csv — bỏ qua phần kiểm tra chia tập.")

    if plot or gee_project:
        if raw is None:
            raw = _read_if_exists(_p(data_dir, "raw_dataset.csv"))
        if raw is not None and len(raw):
            qd = qa_date or _busiest_date(raw)
            if plot:
                frame = _read_if_exists(_p(data_dir, "sampling_frame.csv"))
                out_png = _p(data_dir, f"qa_overlay_{qd}.png")
                try:
                    make_overlay_plot(raw, frame, boundary, qd, out_png)
                    rep.add("ảnh", "Ảnh chồng lớp", "INFO", f"đã lưu {out_png} — nhìn: điểm xanh phải phủ đúng trong ranh giới, "
                            f"LST cao ở lõi đô thị/thấp ở rừng-mặt nước, NDVI–LST dốc xuống")
                except Exception as e:  # noqa: BLE001
                    rep.add("ảnh", "Ảnh chồng lớp", "WARN", f"không vẽ được: {str(e)[:120]}")
            if gee_project:
                if verbose:
                    print(f"\n[QA] Kiểm tra với GEE (ngày {qd})...")
                run_gee_qa(rep, data_dir, boundary, gee_project, qd, n_check)

    out = rep.df()
    os.makedirs(data_dir, exist_ok=True)
    out.to_csv(_p(data_dir, "qa_report.csv"), index=False)
    if verbose:
        print(f"\n[QA] Tổng kết: {rep.n('PASS')} PASS | {rep.n('WARN')} WARN | {rep.n('FAIL')} FAIL | "
              f"{rep.n('INFO')} INFO | {rep.n('SKIP')} SKIP  → {_p(data_dir, 'qa_report.csv')}")
        if rep.n("FAIL"):
            print("   ❌ Có FAIL — KHÔNG nên huấn luyện trên dataset này cho tới khi sửa (xem các dòng ❌ ở trên).")
        elif rep.n("WARN"):
            print("   ⚠️  Có WARN — đọc chi tiết; ghi nhận vào báo cáo nếu là đặc điểm dữ liệu (không nhất thiết là lỗi).")
        else:
            print("   ✅ Không có cảnh báo.")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--boundary", default=None, help="Mặc định: <data_dir>/hcmc_boundary.geojson")
    parser.add_argument("--scope", choices=["raw", "split", "all"], default="all")
    parser.add_argument("--min_valid_frac", type=float, default=None,
                        help="Ngưỡng độ phủ đã dùng khi ghép cặp (04 --min_valid_frac). Mặc định: tự suy ra từ dữ liệu.")
    parser.add_argument("--max_st_unc_k", type=float, default=None, help="Mặc định: đọc từ cache_manifest.json")
    parser.add_argument("--spatial_districts", default=",".join(build_dataset.SPATIAL_TEST_DISTRICTS))
    parser.add_argument("--plot", action="store_true", help="Vẽ ảnh chồng lớp qa_overlay_<ngày>.png (offline)")
    parser.add_argument("--qa_date", default=None, help="Ngày t để vẽ/đối chiếu GEE (mặc định: ngày nhiều dòng nhất)")
    parser.add_argument("--gee", action="store_true", help="Thêm kiểm tra với GEE (cần --project, cần mạng)")
    parser.add_argument("--project", default=None, help="GEE Cloud Project ID (cho --gee)")
    parser.add_argument("--n_check", type=int, default=60, help="Số điểm trích lại từ GEE để so với cache")
    parser.add_argument("--strict", action="store_true", help="Coi WARN như FAIL (mã thoát 1)")
    args = parser.parse_args()
    if args.gee and not args.project:
        raise SystemExit("--gee cần --project YOUR_GEE_PROJECT_ID")

    boundary = args.boundary or _p(args.data_dir, "hcmc_boundary.geojson")
    spatial = [s.strip() for s in args.spatial_districts.split(",") if s.strip()]
    rep = run_qa(args.data_dir, boundary, scope=args.scope, min_valid_frac=args.min_valid_frac, max_st_unc_k=args.max_st_unc_k,
                 spatial_districts=spatial, plot=args.plot, qa_date=args.qa_date,
                 gee_project=args.project if args.gee else None, n_check=args.n_check)
    n_fail = int((rep["status"] == "FAIL").sum())
    n_warn = int((rep["status"] == "WARN").sum())
    sys.exit(1 if (n_fail or (args.strict and n_warn)) else 0)


if __name__ == "__main__":
    main()
