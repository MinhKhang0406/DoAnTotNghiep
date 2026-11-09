"""
BƯỚC 10b — THÊM DỰ BÁO v2 + KHOẢNG TIN CẬY VÀO LƯỚI DỰ BÁO CỦA BƯỚC 10 (không cần GEE, không sửa 10)
=======================================================================================================
10_build_inference_grid.py ghi ../data/forecast_grid_<t>.csv (định dạng dài: mỗi (điểm, delta_t) một dòng, đã có LST_t,
NDVI/NDBI/NDWI, ERA5, date_t, delta_t, date_forecast, has_data, reason). Script này ĐỌC chính các file đó và:

  * dự báo lại bằng mô hình v2 (best_model_v2.pkl của 11, hoặc best_model_v2_tuned.pkl của 12 qua --model_file);
  * thêm khoảng tin cậy từ interval_calibration.json (13): LST_lo / LST_hi theo --method (conformal | cqr);
  * gắn cờ delta_t_out_of_range khi delta_t nằm ngoài khoảng đã học (mô hình phải ngoại suy -> khoảng kém tin cậy);
  * KHÔNG đụng vào cột LST_pred gốc (dự báo của 06) để còn so sánh; cột mới là LST_pred_v2.

Ghi ../data/forecast_grid_v2_<t>.csv (file gốc giữ nguyên). Điểm has_data=False giữ NaN (frontend vẫn vẽ 'không có dữ liệu').
16_export_to_postgis_v2.py đẩy các file này vào bảng lst_forecast_grid_v2.

Cách chạy (sau 10, 11, 13):
    python 10b_add_intervals_to_grid.py                       # mọi forecast_grid_*.csv chưa có bản v2
    python 10b_add_intervals_to_grid.py --date 2026-06-12     # chỉ một ngày
    python 10b_add_intervals_to_grid.py --method conformal --model_file best_model_v2_tuned.pkl --overwrite
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_utils as eu  # noqa: E402

NEED_COLS = ["ADM1_NAME", "ADM2_NAME", "date_t", "date_forecast", "delta_t", "has_data"] + eu.FEATURES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="../data")
    ap.add_argument("--date", default=None, help="Chỉ xử lý forecast_grid_<date>.csv")
    ap.add_argument("--model_file", default="best_model_v2.pkl")
    ap.add_argument("--method", choices=["auto", "conformal", "cqr"], default="auto",
                    help="auto = cqr nếu có quantile_model_v2.pkl và hiệu chuẩn cqr, ngược lại conformal")
    ap.add_argument("--overwrite", action="store_true", help="Ghi đè forecast_grid_v2_<t>.csv đã có")
    args = ap.parse_args()

    pattern = f"forecast_grid_{args.date}.csv" if args.date else "forecast_grid_*.csv"
    files = [f for f in sorted(glob.glob(os.path.join(args.data_dir, pattern))) if "forecast_grid_v2_" not in os.path.basename(f)]
    if not files:
        raise SystemExit(f"❌ Không thấy {pattern} trong {args.data_dir} — chạy 10_build_inference_grid.py trước.")

    bundle = eu.load_bundle(args.data_dir, args.model_file)
    calib = eu.load_json(os.path.join(args.data_dir, "interval_calibration.json"))
    if calib is None:
        raise SystemExit("❌ Không thấy interval_calibration.json — chạy 13_uncertainty_delta_t.py trước.")
    if calib.get("model_file") != args.model_file:
        print(f"⚠️  Hiệu chuẩn khoảng tin cậy được tạo cho '{calib.get('model_file')}' nhưng đang dùng '{args.model_file}' "
              f"— chạy lại 13 với --model_file {args.model_file} để khoảng tin cậy đúng.")
    qpath = os.path.join(args.data_dir, "quantile_model_v2.pkl")
    qbundle = None
    method = args.method
    if method in ("auto", "cqr"):
        if os.path.exists(qpath) and "cqr" in calib:
            import pickle

            with open(qpath, "rb") as f:
                qbundle = pickle.load(f)
            method = "cqr"
        elif method == "cqr":
            raise SystemExit("❌ --method cqr cần quantile_model_v2.pkl và mục 'cqr' trong interval_calibration.json (chạy 13).")
        else:
            method = "conformal"
    print(f"Mô hình: {bundle['model_name']} / {bundle['target_mode']}:{bundle['feature_set']} | khoảng {calib['coverage']:.0%} bằng {method}")
    rng = calib.get("train_delta_t", {})
    print(f"delta_t đã học: [{rng.get('min', float('nan')):.0f}, {rng.get('max', float('nan')):.0f}] ngày")

    done = 0
    for path in files:
        base = os.path.basename(path)
        out_path = os.path.join(args.data_dir, base.replace("forecast_grid_", "forecast_grid_v2_", 1))
        if os.path.exists(out_path) and not args.overwrite:
            print(f"   {base}: đã có bản v2 — bỏ qua (dùng --overwrite để làm lại).")
            continue
        g = pd.read_csv(path)
        miss = [c for c in NEED_COLS if c not in g.columns]
        if miss:
            print(f"   ⚠️  {base} thiếu cột {miss} — bỏ qua.")
            continue
        has = g["has_data"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
        for c in ("LST_pred_v2", "LST_lo", "LST_hi", "LST_lo_conformal", "LST_hi_conformal", "LST_lo_cqr", "LST_hi_cqr"):
            g[c] = np.nan
        g["delta_t_bin"] = None
        g["delta_t_out_of_range"] = False
        g["interval_method"] = method
        g["interval_coverage"] = calib["coverage"]
        g["model_v2"] = f"{bundle['model_name']}|{bundle['target_mode']}:{bundle['feature_set']}"
        if has.any():
            X = g.loc[has].copy()
            X["date_target"] = X["date_forecast"]  # cột mà Variant.prepare / climatology cần
            oc = eu.predict_with_interval(bundle, calib, X, "conformal")
            g.loc[has, "LST_pred_v2"] = oc["LST_pred"].to_numpy()
            g.loc[has, "LST_lo_conformal"], g.loc[has, "LST_hi_conformal"] = oc["LST_lo"].to_numpy(), oc["LST_hi"].to_numpy()
            if qbundle is not None:
                oq = eu.predict_with_interval(bundle, calib, X, "cqr", qbundle)
                g.loc[has, "LST_lo_cqr"], g.loc[has, "LST_hi_cqr"] = oq["LST_lo"].to_numpy(), oq["LST_hi"].to_numpy()
            chosen = oq if method == "cqr" else oc
            g.loc[has, "LST_lo"], g.loc[has, "LST_hi"] = chosen["LST_lo"].to_numpy(), chosen["LST_hi"].to_numpy()
            g.loc[has, "delta_t_bin"] = oc["delta_t_bin"].to_numpy()
            g.loc[has, "delta_t_out_of_range"] = oc["delta_t_out_of_range"].to_numpy()
        g.to_csv(out_path, index=False)
        done += 1
        print(f"\n   {base} -> {os.path.basename(out_path)} ({len(g):,} dòng, {int(has.sum()):,} có dữ liệu)")
        if has.any():
            h = g[has]
            summ = h.groupby("delta_t").agg(LST_v2=("LST_pred_v2", "mean"), rong_TB=("LST_hi", lambda s: float(np.mean(s - h.loc[s.index, "LST_lo"]))),
                                            ngoai_khoang=("delta_t_out_of_range", "mean")).round(2)
            print("   Theo delta_t (LST_v2 TB °C | độ rộng khoảng TB °C | tỉ lệ delta_t ngoài khoảng đã học):")
            print("   " + summ.to_string().replace("\n", "\n   "))
            if "LST_pred" in h.columns and h["LST_pred"].notna().any():
                d = (h["LST_pred_v2"] - h["LST_pred"]).dropna()
                print(f"   Chênh v2 − v1 (06): TB {d.mean():+.2f} °C, |chênh| TB {d.abs().mean():.2f} °C")
            if h["delta_t_out_of_range"].any():
                print("   ⚠️  Có delta_t ngoài khoảng đã học — khoảng tin cậy ở đó KHÔNG đáng tin; frontend nên cảnh báo (cột delta_t_out_of_range).")
    print(f"\n✅ Đã tạo {done} file forecast_grid_v2_*.csv. Đẩy vào CSDL: python 16_export_to_postgis_v2.py --tables grid")


if __name__ == "__main__":
    main()
