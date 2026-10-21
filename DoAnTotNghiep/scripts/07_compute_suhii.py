"""
BƯỚC 7 — Tính chỉ số Đảo nhiệt đô thị bề mặt (SUHII)
=========================================================
File này còn THIẾU HẲN ở bản trước (đề cương yêu cầu SUHII như 1 "phân tích
mô tả bổ sung" ở mục 1.2/3.6, và Tuần 13 yêu cầu "bổ sung biểu đồ... chỉ số
SUHII mô tả" tích hợp WebGIS — chưa có script nào tính cả).

Định nghĩa dùng ở đây (Voogt & Oke 2003, trích trong đề cương):
    SUHII = LST trung bình khu vực đô thị − LST trung bình khu vực nền/
            nông thôn tham chiếu, TRONG CÙNG 1 thời điểm.

Cách xác định "đô thị" / "nền tham chiếu": dùng phân loại bề mặt NDVI/NDBI/
NDWI (classify_surface — DÙNG CHUNG với 08_ablation_shap.py, xem
03_prepare_features.py) — "Xây dựng" = đô thị, "Thực vật"/"Mặt nước" = nền
tham chiếu. KHÔNG cố định theo tên quận/huyện, vì ranh giới quận/huyện cũ
không còn phản ánh đúng "trong đô thị hay ngoài đô thị" sau khi TP.HCM sáp
nhập thêm Bình Dương/BR-VT (nhiều xã cũ thuộc Bình Dương cũng đã đô thị hóa
cao, không còn "nông thôn" theo nghĩa cũ).

Tính RIÊNG cho từng ngày (date_t) trong test_temporal.csv (dùng LST_target
— giá trị THỰC ĐO, không dùng LST dự báo, vì SUHII là chỉ số mô tả hiện
trạng, không phải đánh giá mô hình) — ra được chuỗi SUHII theo thời gian để
vẽ biểu đồ trên WebGIS.

⚠️ LƯU Ý (đúng đề cương): SUHII là PHÂN TÍCH MÔ TẢ bổ sung, KHÔNG dùng để
kết luận về "rủi ro nhiệt" — đề cương yêu cầu tách bạch rõ 3 khái niệm
LST / UHI (qua SUHII) / rủi ro nhiệt, không được đánh đồng.

Cách chạy:
    python 07_compute_suhii.py
"""

import importlib.util
import os

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "prepare_features", os.path.join(_THIS_DIR, "03_prepare_features.py")
)
prepare_features = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare_features)


def compute_suhii_by_date(df: pd.DataFrame, lst_col: str = "LST_target") -> pd.DataFrame:
    """Trả về DataFrame: date_t, lst_urban_mean, lst_rural_mean, suhii, n_urban, n_rural.
    Ngày nào thiếu hẳn 1 trong 2 nhóm (đô thị hoặc nền tham chiếu) sẽ bị bỏ
    qua (không tự nội suy — nhất quán với nguyên tắc "không nội suy" của
    toàn bộ đồ án)."""
    df = df.copy()
    df["surface_type"] = prepare_features.classify_surface(df)

    rows = []
    for date_t, group in df.groupby("date_t"):
        urban = group.loc[group["surface_type"] == "Xây dựng", lst_col]
        rural = group.loc[group["surface_type"].isin(["Thực vật", "Mặt nước"]), lst_col]
        if len(urban) == 0 or len(rural) == 0:
            continue
        rows.append(
            {
                "date_t": date_t,
                "lst_urban_mean": urban.mean(),
                "lst_rural_mean": rural.mean(),
                "suhii": urban.mean() - rural.mean(),
                "n_urban": len(urban),
                "n_rural": len(rural),
            }
        )
    cols = ["date_t", "lst_urban_mean", "lst_rural_mean", "suhii", "n_urban", "n_rural"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("date_t").reset_index(drop=True)


def main():
    data_dir = "../data"
    test_path = os.path.join(data_dir, "test_temporal.csv")
    if not os.path.exists(test_path):
        print(f"❌ Không tìm thấy {test_path} — chạy 04_build_dataset.py trước.")
        return

    test_df = pd.read_csv(test_path)
    missing = [c for c in ["NDVI", "NDBI", "NDWI", "LST_target", "date_t"] if c not in test_df.columns]
    if missing:
        print(f"❌ test_temporal.csv thiếu cột {missing} — kiểm tra lại đã chạy đúng bản 04 mới chưa.")
        return

    print("Đang tính SUHII theo từng ngày (dựa trên LST_target thực đo, KHÔNG dùng LST dự báo)...")
    suhii_df = compute_suhii_by_date(test_df, lst_col="LST_target")

    if suhii_df.empty:
        surface_counts = pd.Series(prepare_features.classify_surface(test_df)).value_counts().to_dict()
        print("❌ Không đủ dữ liệu — thiếu hẳn nhóm 'Xây dựng' (đô thị) HOẶC nhóm")
        print("   'Thực vật'/'Mặt nước' (nền tham chiếu) ở MỌI ngày.")
        print(f"   Phân bố loại bề mặt hiện có trong test_temporal.csv: {surface_counts}")
        if surface_counts.get("Xây dựng", 0) == 0:
            print(
                "   -> 0 điểm 'Xây dựng' trong toàn bộ tập test — rất có thể do các quận/huyện\n"
                "      đô thị (Quận 1,3,4,5,6,7,8,10,11...) đang bị thiếu mẫu gần như hoàn toàn\n"
                "      trong dataset. Chạy diagnose_boundary.py và xem lại độ cân bằng dữ liệu ở\n"
                "      Bước 4 trước khi tin vào kết quả SUHII/ablation theo bề mặt."
            )
        return

    out_path = os.path.join(data_dir, "suhii_by_date.csv")
    suhii_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(" KẾT QUẢ SUHII (Surface Urban Heat Island Intensity)")
    print("=" * 70)
    print(suhii_df.to_string(index=False))
    print("-" * 70)
    print(f"SUHII trung bình toàn giai đoạn: {suhii_df['suhii'].mean():.2f} °C")
    idx_max = suhii_df["suhii"].idxmax()
    idx_min = suhii_df["suhii"].idxmin()
    print(f"SUHII lớn nhất:  {suhii_df.loc[idx_max, 'suhii']:.2f} °C (ngày {suhii_df.loc[idx_max, 'date_t']})")
    print(f"SUHII nhỏ nhất:  {suhii_df.loc[idx_min, 'suhii']:.2f} °C (ngày {suhii_df.loc[idx_min, 'date_t']})")
    print(f"\n✅ Đã lưu chi tiết vào: {out_path}")
    print("   -> Dùng file này để vẽ biểu đồ SUHII theo thời gian trên WebGIS (Tuần 13).")
    print("\n⚠️  SUHII là PHÂN TÍCH MÔ TẢ bổ sung — KHÔNG dùng để kết luận về 'rủi ro")
    print("   nhiệt' (đề cương yêu cầu tách bạch rõ LST / UHI / rủi ro nhiệt).")


if __name__ == "__main__":
    main()
