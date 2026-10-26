"""
CHẨN ĐOÁN: bước mặt nạ nào loại mất pixel? (thay diagnose_district_pixel_count.py)
===================================================================================
Bản cũ chỉ đếm pixel SAU toàn bộ mặt nạ -> thấy "quận X còn ít pixel" nhưng không biết
do bước nào (mây? bão hòa? ngoài khoảng vật lý? ST_QA? z-score cục bộ?). Script này đếm
pixel còn lại SAU TỪNG BƯỚC, cho từng quận/huyện, bằng đúng cùng mặt nạ mà 03/04 dùng:

  LANDSAT (mặc định)      scene -> +mây/bóng mây/mây mỏng -> +QA_RADSAT -> +LST -10..60 °C
                          -> +ST_QA <= K (so sánh nhiều K: 2, 3, 5 K) -> +z-score cục bộ
  SENTINEL-2 (--s2)       footprint -> +QA60 -> +s2cloudless (<40%) -> +SCL (bóng mây/lỗi)
                          trong cửa sổ [t - s2_back_days, t] — tỷ lệ pixel có NDVI/NDBI/NDWI hợp lệ

Chỉ cần 1 lần reduceRegions/chunk quận cho mọi bước (pixel đếm bằng Reducer.count() trên ảnh
mosaic đã mask — giống cách trích dữ liệu thật). Mỗi ngày mất vài phút (z-score nặng nhất).

CÁCH ĐỌC KẾT QUẢ
  * "giữ sau mây" = pixel còn lại cuối cùng / pixel không mây. ĐÂY là chỉ số quan trọng: mây là
    chuyện của từng ngày, không sửa được; còn các bước SAU mây là do ta chọn. Nếu "giữ sau mây" thấp
    (< ~60%) ở quận đô thị -> ST_QA quá chặt (thử --max_st_unc_k 5 ở 04) hoặc z-score quá nhạy
    (tăng --zscore / giảm --focal_radius_px).
  * Cột unc_3 phải == final_noz khi --max_st_unc_k 3 (kiểm tra chéo 2 cách tính mặt nạ).
  * Bảng S2: nếu 'scl' (hoặc 'prob') còn rất thấp -> dataset sẽ thiếu NDVI/NDBI/NDWI nhiều (mùa mưa).
    Cân nhắc nới --s2_back_days (04), hoặc phương án dự phòng chỉ số từ Landsat OLI SR (phải sửa đề cương).
    Nếu 'qa60' luôn 100% ở ảnh gần đây: QA60 không còn được ghi sau đầu 2022 (cần xác nhận) — khi đó
    s2cloudless + SCL mới là bộ lọc mây thật.

Cách chạy:
    python diagnose_mask_components.py --project YOUR_ID                  # tự chọn 2 ngày độ phủ cao nhất (khô + mưa) từ cache của 04
    python diagnose_mask_components.py --project YOUR_ID --dates 2025-02-11,2025-07-20
    python diagnose_mask_components.py --project YOUR_ID --districts "Quan 1,Quan 3,Tan Binh,Can Gio" --s2
Kết quả: bảng trên màn hình + ../data/diag_mask_components.csv (+ diag_s2_components.csv với --s2).
"""

import argparse
import importlib.util
import os
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

def _unc_name(k: float) -> str:
    """Tên băng/cột cho ngưỡng ST_QA k (K): 2 -> unc_2, 2.5 -> unc_2_5."""
    return "unc_" + f"{float(k):g}".replace(".", "_")


STEP_LABELS = {
    "scene": "trong cảnh", "cloud": "+không mây", "radsat": "+không bão hòa", "range": "+LST -10..60",
    "final_noz": "+ST_QA (không z)", "final_z": "+z-score = CUỐI",
}


# ---------------------------------------------------------------------
# Chọn ngày
# ---------------------------------------------------------------------

def pick_dates(data_dir: str, boundary: str, n_per_season: int = 1, min_valid_frac: float = 0.10) -> list:
    """Ngày độ phủ LST cao nhất trong mỗi mùa (khô/mưa), lấy từ cache của 04 — tránh phải đoán ngày."""
    frame_path = os.path.join(data_dir, "sampling_frame.csv")
    if not os.path.exists(frame_path):
        raise SystemExit("❌ Không tự chọn được ngày (thiếu sampling_frame.csv/cache của 04) — dùng --dates YYYY-MM-DD,...")
    dates, lst = build_dataset.load_lst_cache(data_dir)  # SystemExit nếu thiếu cache
    tbl = build_dataset.valid_dates_table(dates, lst, pd.read_csv(frame_path), min_valid_frac)
    tbl = tbl[tbl["valid"]].copy()
    if tbl.empty:
        raise SystemExit("❌ Cache không có ngày hợp lệ nào — dùng --dates.")
    tbl["season"] = prepare_features.classify_season(tbl["date"])
    picked = []
    for s in ("Mùa khô", "Mùa mưa"):
        sub = tbl[tbl["season"] == s].sort_values("coverage", ascending=False).head(n_per_season)
        picked += sub["date"].tolist()
    return sorted(picked)


# ---------------------------------------------------------------------
# LANDSAT: đếm pixel sau từng bước mặt nạ
# ---------------------------------------------------------------------

def _landsat_step_image(image, k_levels, k_final, zscore, radius):
    """Ảnh nhiều băng; băng = 1 ở pixel còn LẠI sau bước đó (masked ở pixel bị loại)."""
    import ee

    qa = image.select("QA_PIXEL")
    ones = ee.Image.constant(1).toFloat()
    scene = qa.bitwiseAnd(1).eq(0)  # bit 0 = fill
    cloud_ok = (qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0)).And(qa.bitwiseAnd(1 << 1).eq(0)))
    radsat_ok = image.select("QA_RADSAT").eq(0)
    lst_c = image.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)
    range_ok = lst_c.gte(-10).And(lst_c.lte(60))
    unc = image.select("ST_QA").multiply(0.01)

    s_scene = scene
    s_cloud = s_scene.And(cloud_ok)
    s_rad = s_cloud.And(radsat_ok)
    s_range = s_rad.And(range_ok)
    bands = [ones.updateMask(s_scene).rename("scene"), ones.updateMask(s_cloud).rename("cloud"),
             ones.updateMask(s_rad).rename("radsat"), ones.updateMask(s_range).rename("range")]
    for k in k_levels:
        bands.append(ones.updateMask(s_range.And(unc.lte(k))).rename(_unc_name(k)))
    # Đường ống đầy đủ qua đúng hàm dùng ở 03: không z-score và có z-score
    full_noz = prepare_features.mask_landsat_c2l2(image, zscore_threshold=zscore, focal_radius_px=radius,
                                                  apply_zscore=False, max_st_unc_k=k_final)
    full_z = prepare_features.mask_landsat_c2l2(image, zscore_threshold=zscore, focal_radius_px=radius,
                                                apply_zscore=True, max_st_unc_k=k_final)
    bands.append(ones.updateMask(full_noz.select("LST_C").mask()).rename("final_noz"))
    bands.append(ones.updateMask(full_z.select("LST_C").mask()).rename("final_z"))
    return ee.Image.cat(bands)


def _district_fc(districts, idxs, simplify_deg: float):
    import ee
    from shapely.geometry import mapping

    feats = []
    for _, r in districts[districts["district_idx"].isin(idxs)].iterrows():
        g = r.geometry.simplify(simplify_deg, preserve_topology=True)
        feats.append(ee.Feature(ee.Geometry(mapping(g)), {"district_idx": int(r["district_idx"])}))
    return ee.FeatureCollection(feats)


def _count_rows(img, fc, band_names, label):
    out = img.reduceRegions(collection=fc, reducer=prepare_features.ee.Reducer.count(), scale=prepare_features.GRID_SCALE_M,
                            crs=CRS_UTM, tileScale=8)
    info = prepare_features.ee_retry(lambda: out.getInfo(), label=label)
    rows = []
    for f in info.get("features", []):
        p = f.get("properties", {})
        rec = {"district_idx": p.get("district_idx")}
        for b in band_names:
            rec[b] = p.get(b) or 0
        rows.append(rec)
    return rows


def landsat_counts(roi, districts, date, k_levels, k_final, zscore, radius, chunk, simplify_deg) -> pd.DataFrame:
    import ee

    d1 = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(date, d1)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(date, d1)
    col = l8.merge(l9)
    n = prepare_features.ee_retry(lambda: col.size().getInfo(), label=f"size {date}")
    if not n:
        print(f"   ❌ Không có cảnh Landsat 8/9 nào chạm ROI ngày {date}")
        return pd.DataFrame()
    print(f"   {date}: {n} cảnh Landsat chạm ROI")
    img = col.map(lambda im: _landsat_step_image(im, k_levels, k_final, zscore, radius)).mosaic()
    band_names = ["scene", "cloud", "radsat", "range"] + [_unc_name(k) for k in k_levels] + ["final_noz", "final_z"]
    idxs = districts["district_idx"].tolist()
    rows = []
    for i in range(0, len(idxs), chunk):
        part = idxs[i:i + chunk]
        print(f"      chunk quận {i + 1}-{i + len(part)}/{len(idxs)} ...")
        rows += _count_rows(img, _district_fc(districts, part, simplify_deg), band_names, f"{date} chunk {i}")
    df = pd.DataFrame(rows)
    df.insert(0, "date", date)
    return df


# ---------------------------------------------------------------------
# SENTINEL-2: đếm pixel có NDVI hợp lệ sau từng bước mặt nạ (composite cửa sổ lùi)
# ---------------------------------------------------------------------

def s2_counts(roi, districts, date, back_days, chunk, simplify_deg) -> pd.DataFrame:
    import ee

    t = ee.Date(date)
    start, end = t.advance(-back_days, "day"), t.advance(1, "day")
    col = prepare_features._join_s2_with_cloud_prob(roi, start, end)
    n = prepare_features.ee_retry(lambda: col.size().getInfo(), label=f"s2 size {date}")
    print(f"   {date}: {n} ảnh S2 trong cửa sổ [t-{back_days}, t]")
    if not n:
        return pd.DataFrame()

    def steps(im):
        ones = ee.Image.constant(1).toFloat()
        qa = im.select("QA60")
        qa60_ok = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        raw_prob = im.get("cloud_prob")
        prob_ok = ee.Image(ee.Algorithms.If(raw_prob, ee.Image(raw_prob).select("probability").lt(40), ee.Image.constant(1)))
        scl = im.select("SCL")
        scl_ok = scl.neq(3).And(scl.neq(1))
        foot = im.select("B8").mask()
        s0 = foot
        s1 = s0.And(qa60_ok)
        s2 = s1.And(prob_ok)
        s3 = s2.And(scl_ok)
        return ee.Image.cat([ones.updateMask(s0).rename("s2_footprint"), ones.updateMask(s1).rename("s2_qa60"),
                             ones.updateMask(s2).rename("s2_prob"), ones.updateMask(s3).rename("s2_scl")])

    # pixel có >= 1 ảnh hợp lệ trong cửa sổ <=> composite median có giá trị
    img = col.map(steps).count().gt(0).selfMask()
    band_names = ["s2_footprint", "s2_qa60", "s2_prob", "s2_scl"]
    idxs = districts["district_idx"].tolist()
    rows = []
    for i in range(0, len(idxs), chunk):
        part = idxs[i:i + chunk]
        print(f"      chunk quận {i + 1}-{i + len(part)}/{len(idxs)} ...")
        rows += _count_rows(img, _district_fc(districts, part, simplify_deg), band_names, f"s2 {date} chunk {i}")
    df = pd.DataFrame(rows)
    df.insert(0, "date", date)
    df["s2_n_img"] = n
    return df


# ---------------------------------------------------------------------
# BÁO CÁO (thuần pandas — kiểm thử được không cần GEE)
# ---------------------------------------------------------------------

def report_landsat(counts: pd.DataFrame, districts: pd.DataFrame, k_final: float, k_levels, top: int, want_names) -> pd.DataFrame:
    unc_cols = {k: _unc_name(k) for k in k_levels}
    chain = ["scene", "cloud", "radsat", "range", unc_cols[k_final], "final_z"]
    chain_lab = ["trong cảnh", "+không mây", "+không bão hòa", "+LST hợp lý", f"+ST_QA<={k_final:g}K", "+z-score (CUỐI)"]
    df = counts.merge(districts[["district_idx", "ADM1_NAME", "ADM2_NAME"]], on="district_idx", how="left")

    for date, g in df.groupby("date"):
        tot = g[["scene", "cloud", "radsat", "range"] + list(unc_cols.values()) + ["final_noz", "final_z"]].sum()
        print(f"\n{'=' * 78}\n NGÀY {date} — TOÀN VÙNG (cộng mọi pixel, {int(tot['scene']):,} pixel trong cảnh)\n{'=' * 78}")
        print(f"   {'bước':<22}{'pixel còn':>14}{'% cảnh':>9}{'mất bước này':>14}{'% trên pixel không mây':>26}")
        prev = None
        for c, lab in zip(chain, chain_lab):
            v = float(tot[c])
            lost = "" if prev is None else f"{(prev - v) / max(tot['scene'], 1) * 100:>12.1f}pp"
            keep_cloud = "" if c in ("scene", "cloud") else f"{v / max(tot['cloud'], 1) * 100:>24.1f}%"
            print(f"   {lab:<22}{int(v):>14,}{v / max(tot['scene'], 1) * 100:>8.1f}%{lost:>14}{keep_cloud:>26}")
            prev = v
        print("   So sánh ngưỡng ST_QA (sau bước 'LST hợp lý'): " + " | ".join(
            f"<= {k:g} K giữ {float(tot[c]) / max(tot['range'], 1) * 100:.1f}%" for k, c in unc_cols.items()))
        chk = abs(float(tot[unc_cols[k_final]]) - float(tot["final_noz"]))
        print(f"   Kiểm tra chéo 2 cách tính mặt nạ (unc_{k_final:g} vs đường ống 03 không z): lệch {int(chk):,} pixel"
              + ("  ✅" if chk <= 0.001 * max(tot["scene"], 1) else "  ⚠️ lệch đáng kể — mặt nạ ở 03 khác mô tả trong script này"))

    # từng quận
    g = df.copy()
    g["keep_after_cloud_pct"] = np.where(g["cloud"] > 0, g["final_z"] / g["cloud"].clip(lower=1) * 100, np.nan)
    g["final_pct_scene"] = np.where(g["scene"] > 0, g["final_z"] / g["scene"].clip(lower=1) * 100, np.nan)
    losses = pd.DataFrame({
        "cloud": g["scene"] - g["cloud"], "radsat": g["cloud"] - g["radsat"], "range": g["radsat"] - g["range"],
        "ST_QA": g["range"] - g[unc_cols[k_final]], "z-score": g["final_noz"] - g["final_z"],
    })
    g["main_loss"] = losses.idxmax(axis=1)
    g["main_noncloud_loss"] = losses.drop(columns="cloud").idxmax(axis=1)

    print(f"\n{'=' * 78}\n TỪNG QUẬN — {top} quận 'giữ sau mây' THẤP nhất (mỗi ngày)\n{'=' * 78}")
    for date, gg in g.groupby("date"):
        gg = gg[gg["cloud"] > 0].sort_values("keep_after_cloud_pct").head(top)
        print(f"\n NGÀY {date}")
        print(f"   {'ADM2_NAME':<18}{'cảnh':>9}{'không mây':>11}{'CUỐI':>9}{'giữ sau mây':>13}  mất nhiều nhất (ngoài mây)")
        for _, r in gg.iterrows():
            print(f"   {str(r['ADM2_NAME'])[:17]:<18}{int(r['scene']):>9,}{int(r['cloud']):>11,}{int(r['final_z']):>9,}{r['keep_after_cloud_pct']:>12.1f}%  {r['main_noncloud_loss']}")
        low = g[(g["date"] == date) & (g["keep_after_cloud_pct"] < 60)]
        if len(low):
            print(f"   -> {len(low)} quận giữ < 60% pixel không mây; bước ngoài mây gây mất nhiều nhất: "
                  f"{low['main_noncloud_loss'].value_counts().to_dict()}")
        else:
            print("   -> Không quận nào giữ < 60% pixel không mây: mặt nạ ngoài mây (ST_QA/z-score/…) KHÔNG phải nút thắt.")

    if want_names:
        keys = {prepare_features.normalize_vn_name(x) for x in want_names}
        sel = g[g["ADM2_NAME"].apply(prepare_features.normalize_vn_name).isin(keys)]
        if len(sel):
            print(f"\n{'=' * 78}\n CÁC QUẬN BẠN CHỌN\n{'=' * 78}")
            print(sel[["date", "ADM2_NAME", "scene", "cloud", "radsat", "range", unc_cols[k_final], "final_z", "keep_after_cloud_pct"]]
                  .round(1).to_string(index=False))
        else:
            print(f"\n⚠️  Không quận nào khớp {list(want_names)} — xem tên thật trong cột ADM2_NAME của {len(districts)} đơn vị.")
    return g


def report_s2(counts: pd.DataFrame, districts: pd.DataFrame, top: int) -> pd.DataFrame:
    """Tỉ lệ DIỆN TÍCH quận có NDVI/NDBI/NDWI hợp lệ từ cửa sổ S2 [t-back, t] (đã tính cả phần không có ảnh S2 phủ)."""
    df = counts.merge(districts[["district_idx", "ADM1_NAME", "ADM2_NAME", "district_area_km2"]], on="district_idx", how="left")
    df["px_district"] = df["district_area_km2"] * 1e6 / (prepare_features.GRID_SCALE_M ** 2)
    print(f"\n{'=' * 78}\n SENTINEL-2 — % DIỆN TÍCH có NDVI/NDBI/NDWI hợp lệ trong cửa sổ lùi\n{'=' * 78}")
    for date, g in df.groupby("date"):
        tot = g[["px_district", "s2_footprint", "s2_qa60", "s2_prob", "s2_scl"]].sum()
        base = max(float(tot["px_district"]), 1.0)
        n_img = pd.to_numeric(g["s2_n_img"], errors="coerce").iloc[0]
        print(f"\n NGÀY {date} ({int(n_img) if pd.notna(n_img) else '?'} ảnh S2 trong cửa sổ)")
        prev = base
        for c, lab in [("s2_footprint", "có ảnh S2 phủ"), ("s2_qa60", "+QA60"), ("s2_prob", "+s2cloudless<40%"),
                       ("s2_scl", "+SCL (bóng/lỗi) = CUỐI")]:
            v = float(tot[c])
            print(f"   {lab:<26}{int(v):>14,} pixel  {v / base * 100:>6.1f}% diện tích  (mất {(prev - v) / base * 100:.1f}pp)")
            prev = v
        if abs(float(tot["s2_qa60"]) - float(tot["s2_footprint"])) < 1e-6 * base:
            print("   ⚠️  QA60 không loại pixel nào — có thể QA60 không còn được ghi ở ngày này (cần xác nhận); "
                  "bộ lọc mây thực sự là s2cloudless + SCL.")
        worst = g.assign(final_pct=lambda x: x["s2_scl"] / x["px_district"].clip(lower=1) * 100).sort_values("final_pct").head(top)
        print("   quận ít diện tích S2 hợp lệ nhất: " + ", ".join(f"{r['ADM2_NAME']} {r['final_pct']:.0f}%" for _, r in worst.iterrows()))
    print("\n   Hàng 'có ảnh S2 phủ' < 100% = phần quận KHÔNG có ảnh S2 nào trong cửa sổ (quỹ đạo/ngày chụp) — "
          "độc lập với mây; chỉ nới cửa sổ lùi (--s2_back_days ở 04) mới tăng được.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--boundary", default="../data/hcmc_boundary.geojson")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--dates", default=None, help="YYYY-MM-DD,YYYY-MM-DD... (mặc định: tự chọn ngày độ phủ cao nhất mỗi mùa từ cache 04)")
    parser.add_argument("--districts", default=None, help="Tên quận/huyện cần xem riêng (không cần dấu), phân tách bằng dấu phẩy")
    parser.add_argument("--k_levels", default="2,3,5", help="Các ngưỡng ST_QA (K) để so sánh")
    parser.add_argument("--max_st_unc_k", type=float, default=prepare_features.DEFAULT_MAX_ST_UNC_K, help="Ngưỡng dùng cho cột CUỐI")
    parser.add_argument("--zscore", type=float, default=3.0)
    parser.add_argument("--focal_radius_px", type=int, default=2)
    parser.add_argument("--s2", action="store_true", help="Thêm bảng mặt nạ Sentinel-2")
    parser.add_argument("--s2_back_days", type=int, default=prepare_features.S2_BACK_DAYS)
    parser.add_argument("--chunk", type=int, default=20, help="Số quận mỗi lần gọi GEE (giảm nếu bị quá tải)")
    parser.add_argument("--simplify_deg", type=float, default=0.0005, help="Đơn giản hóa polygon quận (độ; 0.0005 ≈ 55 m) để request nhẹ")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    k_levels = sorted({float(x) for x in args.k_levels.split(",") if x.strip()} | {float(args.max_st_unc_k)})
    k_final = float(args.max_st_unc_k)
    dates = [d.strip() for d in args.dates.split(",") if d.strip()] if args.dates else pick_dates(args.data_dir, args.boundary)
    print(f"Ngày kiểm tra: {dates}")

    prepare_features.init_ee(args.project)
    roi = prepare_features.load_boundary_roi(args.boundary)
    districts = prepare_features.load_districts(args.boundary)
    print(f"{len(districts)} đơn vị; ngưỡng ST_QA so sánh {k_levels}; z-score {args.zscore} (bán kính {args.focal_radius_px} px)")

    parts = []
    for d in dates:
        print(f"\n[Landsat] {d}")
        c = landsat_counts(roi, districts, d, k_levels, k_final, args.zscore, args.focal_radius_px, args.chunk, args.simplify_deg)
        if len(c):
            parts.append(c)
    if not parts:
        raise SystemExit("❌ Không có dữ liệu Landsat nào cho các ngày đã chọn.")
    counts = pd.concat(parts, ignore_index=True)
    want = [x.strip() for x in args.districts.split(",") if x.strip()] if args.districts else None
    res = report_landsat(counts, districts, k_final, k_levels, args.top, want)
    out = os.path.join(args.data_dir, "diag_mask_components.csv")
    os.makedirs(args.data_dir, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\n✅ Đã lưu {out}")

    if args.s2:
        sp = []
        for d in dates:
            print(f"\n[Sentinel-2] {d}")
            c = s2_counts(roi, districts, d, args.s2_back_days, args.chunk, args.simplify_deg)
            if len(c):
                sp.append(c)
        if sp:
            s2res = report_s2(pd.concat(sp, ignore_index=True), districts, args.top)
            out2 = os.path.join(args.data_dir, "diag_s2_components.csv")
            s2res.to_csv(out2, index=False)
            print(f"\n✅ Đã lưu {out2}")
        else:
            print("\n⚠️  Không có ảnh S2 nào trong cửa sổ ở các ngày đã chọn.")

    print("\n=> Dán các bảng 'TOÀN VÙNG' và 'giữ sau mây' lên chat nếu cần phân tích tiếp (quyết định ngưỡng ST_QA/z-score/cửa sổ S2).")


if __name__ == "__main__":
    main()
