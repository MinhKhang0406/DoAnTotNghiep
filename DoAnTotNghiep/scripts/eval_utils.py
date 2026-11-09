"""
eval_utils.py — MODULE DÙNG CHUNG cho nhóm script nâng cấp khoa học (11 → 16)
==============================================================================
Không chạy trực tiếp. Gồm:

  * Nạp dữ liệu chia tập (train/val/test_temporal/test_spatial) + kiểm tra cột.
  * Climatology (trung bình LST theo đơn vị × tháng, chỉ học từ TRAIN) có đường lùi
    (đơn vị×tháng -> loại bề mặt×tháng -> tháng -> toàn cục) để đơn vị CHƯA thấy (Test không gian)
    vẫn có giá trị tham chiếu.
  * Baseline mạnh: Persistence, Climatology, Anomaly persistence, Damped anomaly (hệ số giảm dần theo delta_t).
  * Variant = (kiểu target × bộ feature):
        absolute : y = LST_target                       (như 06)
        delta    : y = LST_target − LST_t
        anomaly  : y = LST_target − climatology(tháng của ngày nhãn)
        feature "base" = FEATURE_COLS | "aug" = FEATURE_COLS + [clim_t, anom_t, clim_target]
    (anomaly gần như BẮT BUỘC đi với "aug": mô hình không có tọa độ/quận nên muốn biết dị thường tại t
     phải được cho clim_t.) Climatology luôn học trên Train (hoặc trên phần train của từng fold khi CV).
  * Fit 4 họ mô hình (XGBoost, LightGBM, CatBoost, Random Forest) với early stopping trên Validation.
  * Khoảng tin cậy: split conformal theo từng nhóm delta_t + CQR (XGBoost quantile) — dùng ở 13, 10, 16.
  * Walk-forward theo NGÀY có purge (cùng logic với 06) cho dò siêu tham số / CV.

Quy ước giữ nguyên: Validation dùng để early stopping + CHỌN mô hình; Test chỉ để BÁO CÁO (không chọn theo Test).
"""

import importlib.util
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _constants_from_source(path: str) -> dict:
    """Đọc FEATURE_COLS / ERA5_VARS thẳng từ MÃ NGUỒN 03 (ast, không chạy file) — dùng khi `import ee` trong 03 không được
    (vd Windows Application Control chặn DLL). Chỉ tính các phép gán danh sách/nối danh sách chuỗi, không thực thi mã khác."""
    import ast

    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("ERA5_VARS", "FEATURE_COLS"):
            env[node.targets[0].id] = eval(compile(ast.Expression(node.value), path, "eval"), {"__builtins__": {}}, dict(env))
    missing = [k for k in ("ERA5_VARS", "FEATURE_COLS") if k not in env]
    if missing:
        raise SystemExit(f"❌ Không đọc được {missing} từ 03_prepare_features.py — file 03 khác bản mà nhóm script này giả định.")
    return env


class _ConstOnly:
    """Thay thế tối thiểu cho module 03 khi không import được (chỉ có hằng số danh sách feature)."""

    def __init__(self, env: dict):
        self.FEATURE_COLS, self.ERA5_VARS = env["FEATURE_COLS"], env["ERA5_VARS"]


def _load_prepare_features():
    path = os.path.join(_THIS_DIR, "03_prepare_features.py")
    try:
        spec = importlib.util.spec_from_file_location("prepare_features", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except (ImportError, OSError) as e:  # vd: "DLL load failed ... Application Control policy has blocked this file"
        print(f"⚠️  Không import được 03_prepare_features.py ({type(e).__name__}: {str(e)[:90]}) — đọc FEATURE_COLS/ERA5_VARS từ mã nguồn.")
        return _ConstOnly(_constants_from_source(path))


prepare_features = _load_prepare_features()

FEATURES = list(prepare_features.FEATURE_COLS)
ERA5_VARS = list(prepare_features.ERA5_VARS)
TARGET = "LST_target"
AUG_COLS = ["clim_t", "anom_t", "clim_target"]

SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test_temporal": "test_temporal.csv", "test_spatial": "test_spatial.csv"}
# (tên kịch bản hiển thị — khớp 05/06, khóa tập)
SCENARIOS = [("Validation", "val"), ("Thời gian", "test_temporal"), ("Không gian", "test_spatial")]
SCENARIO_OF = {key: name for name, key in SCENARIOS}

DELTA_EDGES = [3.0, 8.0, 16.0]
DELTA_LABELS = ["1-3", "4-8", "9-16", ">16"]

MODEL_LABELS = {"xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost", "rf": "Random Forest"}

# Bảng màu phân loại (thứ tự cố định, đã kiểm chứng) + màu chữ/lưới cho ảnh PNG
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


COL_ALIASES = {"Mô hình": "model", "Anomaly persistence": "anomaly_persistence", "Damped anomaly": "damped_anomaly",
               "Persistence": "persistence", "Climatology": "climatology", "share_p_lt_0.05": "share_p_lt_005"}
EXACT_ALIASES = {"I": "moran_i", "window": "time_window"}  # 'window' là từ khoá SQL, 'I' viết hoa phải đặt trong ngoặc kép


def db_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột sang ASCII/không dấu chấm để CSV đẩy thẳng vào PostgreSQL (16) truy vấn được không cần ngoặc kép."""
    ren = {}
    for c in df.columns:
        n = EXACT_ALIASES.get(c, c)
        for k, v in COL_ALIASES.items():
            n = n.replace(k, v)
        ren[c] = n
    return df.rename(columns=ren)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# NẠP DỮ LIỆU
# ---------------------------------------------------------------------

def read_csv_safe(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def load_splits(data_dir: str, verbose: bool = True) -> dict:
    """Trả về dict train/val/test_temporal/test_spatial. Train/Val bắt buộc; Test rỗng -> DataFrame rỗng (có cảnh báo)."""
    need = FEATURES + [TARGET, "date_t", "date_target", "point_id", "ADM1_NAME", "ADM2_NAME"]
    out = {}
    for key, fn in SPLIT_FILES.items():
        df = read_csv_safe(os.path.join(data_dir, fn))
        if df is not None and len(df):
            miss = [c for c in need if c not in df.columns]
            if miss:
                raise SystemExit(f"❌ {fn} thiếu cột {miss} — dataset từ bản 04 CŨ; chạy lại 04_build_dataset.py bản mới (--rebuild_raw).")
        out[key] = df
    for key in ("train", "val"):
        if out[key] is None or len(out[key]) == 0:
            raise SystemExit(f"❌ {SPLIT_FILES[key]} không có/rỗng trong {data_dir} — chạy 04_build_dataset.py trước.")
    cols = list(out["train"].columns)
    for key in ("test_temporal", "test_spatial"):
        if out[key] is None or len(out[key]) == 0:
            if verbose:
                print(f"⚠️  {SPLIT_FILES[key]} RỖNG — bỏ qua kịch bản '{SCENARIO_OF[key]}'.")
            out[key] = pd.DataFrame(columns=cols)
    out["train"] = out["train"].sort_values(["date_t", "point_id"]).reset_index(drop=True)
    if verbose:
        print("Kích thước: " + " | ".join(f"{k}={len(v):,}" for k, v in out.items()))
    return out


# ---------------------------------------------------------------------
# CHỈ SỐ, NHÓM delta_t
# ---------------------------------------------------------------------

def metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "bias": np.nan, "n": int(len(y))}
    e = p - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
        "r2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
        "bias": float(np.mean(e)), "n": int(len(y)),
    }


def delta_bin(delta_t) -> np.ndarray:
    """Nhãn nhóm delta_t: '1-3' | '4-8' | '9-16' | '>16' (None nếu NaN)."""
    d = np.asarray(delta_t, dtype=float)
    idx = np.digitize(d, DELTA_EDGES, right=True)
    out = np.array(DELTA_LABELS, dtype=object)[idx]
    out[np.isnan(d)] = None
    return out


# ---------------------------------------------------------------------
# CLIMATOLOGY (chỉ học từ TRAIN)
# ---------------------------------------------------------------------

class Climatology:
    """Trung bình LST theo (ADM1, ADM2, tháng), có đường lùi cho đơn vị chưa thấy (Test không gian):
        (đơn vị, tháng) -> (nhóm NDVI tại t, tháng) -> (tháng) -> trung bình toàn cục.
    Nhóm NDVI = 5 khoảng theo phân vị của NDVI tại t trên Train: phân biệt vùng xanh (nông thôn/rừng) với vùng ít
    cây (đô thị) mịn hơn hẳn nhãn 'Thực vật/Xây dựng' (nhãn thô này gộp cả vùng ven đô vào 'Thực vật' nên làm
    climatology của quận nông thôn lệch cao hàng °C).
    Mỗi ô phải có >= min_n quan sát mới được dùng. Quan sát = LST tại ngày t VÀ LST tại ngày nhãn
    của các dòng truyền vào (khử trùng theo điểm-ngày), nên truyền ĐÚNG phần Train/fold-train."""

    N_NDVI_BINS = 5

    def __init__(self, min_n: int = 20):
        self.min_n = int(min_n)
        self.t_district = self.t_surface = self.t_month = None
        self.ndvi_edges = None
        self.global_mean = np.nan

    def _ndvi_bin(self, ndvi) -> np.ndarray:
        return np.digitize(np.asarray(ndvi, dtype=float), self.ndvi_edges).astype(int)

    def fit(self, df: pd.DataFrame) -> "Climatology":
        t = df[["point_id", "date_t", "ADM1_NAME", "ADM2_NAME", "LST_t", "NDVI", "NDBI", "NDWI"]].drop_duplicates(["point_id", "date_t"])
        t = t.rename(columns={"date_t": "date", "LST_t": "LST"}).reset_index(drop=True)
        self.ndvi_edges = np.quantile(t["NDVI"].to_numpy(float), np.linspace(0, 1, self.N_NDVI_BINS + 1)[1:-1])
        t["ndvi_bin"] = self._ndvi_bin(t["NDVI"])
        t["month"] = pd.to_datetime(t["date"]).dt.month
        g = df[["point_id", "date_target", "ADM1_NAME", "ADM2_NAME", "LST_target"]].drop_duplicates(["point_id", "date_target"])
        g = g.rename(columns={"date_target": "date", "LST_target": "LST"})
        pooled = pd.concat([t[["point_id", "date", "ADM1_NAME", "ADM2_NAME", "LST"]], g], ignore_index=True)
        pooled = pooled.drop_duplicates(["point_id", "date"]).reset_index(drop=True)
        pooled["month"] = pd.to_datetime(pooled["date"]).dt.month

        d = pooled.groupby(["ADM1_NAME", "ADM2_NAME", "month"])["LST"].agg(clim="mean", n="size").reset_index()
        self.t_district = d[d["n"] >= self.min_n][["ADM1_NAME", "ADM2_NAME", "month", "clim"]].reset_index(drop=True)
        s = t.groupby(["ndvi_bin", "month"])["LST"].agg(clim="mean", n="size").reset_index()
        self.t_surface = s[s["n"] >= self.min_n][["ndvi_bin", "month", "clim"]].reset_index(drop=True)
        self.t_month = pooled.groupby("month")["LST"].mean().rename("clim").reset_index()
        self.global_mean = float(pooled["LST"].mean())
        return self

    def lookup(self, df: pd.DataFrame, date_col: str) -> np.ndarray:
        if len(df) == 0:
            return np.array([], dtype=float)
        q = pd.DataFrame({
            "ADM1_NAME": df["ADM1_NAME"].to_numpy(), "ADM2_NAME": df["ADM2_NAME"].to_numpy(),
            "month": pd.to_datetime(df[date_col]).dt.month.to_numpy(),
            "ndvi_bin": self._ndvi_bin(df["NDVI"].to_numpy(float)),
        })
        v1 = q.merge(self.t_district, on=["ADM1_NAME", "ADM2_NAME", "month"], how="left")["clim"].to_numpy(dtype=float)
        v2 = q.merge(self.t_surface, on=["ndvi_bin", "month"], how="left")["clim"].to_numpy(dtype=float)
        v3 = q.merge(self.t_month, on="month", how="left")["clim"].to_numpy(dtype=float)
        out = np.where(np.isnan(v1), v2, v1)
        out = np.where(np.isnan(out), v3, out)
        return np.where(np.isnan(out), self.global_mean, out)

    def to_dict(self) -> dict:
        def rec(x):
            return json.loads(x.to_json(orient="records", force_ascii=False))
        return {"min_n": self.min_n, "global_mean": self.global_mean, "ndvi_edges": [float(x) for x in self.ndvi_edges],
                "district": rec(self.t_district), "surface": rec(self.t_surface), "month": rec(self.t_month)}

    @classmethod
    def from_dict(cls, d: dict) -> "Climatology":
        c = cls(d.get("min_n", 20))
        c.t_district = pd.DataFrame(d["district"], columns=["ADM1_NAME", "ADM2_NAME", "month", "clim"])
        c.t_surface = pd.DataFrame(d["surface"], columns=["ndvi_bin", "month", "clim"])
        c.ndvi_edges = np.array(d["ndvi_edges"], dtype=float)
        c.t_month = pd.DataFrame(d["month"], columns=["month", "clim"])
        c.global_mean = float(d["global_mean"])
        return c


class DampedAnomaly:
    """Dự báo = clim(ngày nhãn) + alpha(delta_t) × (LST_t − clim(ngày t)); alpha khớp bình phương tối thiểu qua gốc
    TRÊN TRAIN theo từng nhóm delta_t (kẹp [0, 1.2]). alpha = 1 là 'Anomaly persistence' thuần."""

    def __init__(self, clim: Climatology, min_rows: int = 30):
        self.clim, self.min_rows = clim, min_rows
        self.alpha = {lab: 1.0 for lab in DELTA_LABELS}

    def fit(self, df: pd.DataFrame) -> "DampedAnomaly":
        a_t = df["LST_t"].to_numpy(float) - self.clim.lookup(df, "date_t")
        a_g = df[TARGET].to_numpy(float) - self.clim.lookup(df, "date_target")
        bins = delta_bin(df["delta_t"])
        den_all, num_all = float((a_t ** 2).sum()), float((a_t * a_g).sum())
        glob = float(np.clip(num_all / den_all, 0, 1.2)) if den_all > 0 else 1.0
        for lab in DELTA_LABELS:
            m = bins == lab
            den = float((a_t[m] ** 2).sum())
            self.alpha[lab] = float(np.clip((a_t[m] * a_g[m]).sum() / den, 0, 1.2)) if (m.sum() >= self.min_rows and den > 0) else glob
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        a = np.array([self.alpha.get(b, 1.0) if b is not None else 1.0 for b in delta_bin(df["delta_t"])])
        return self.clim.lookup(df, "date_target") + a * (df["LST_t"].to_numpy(float) - self.clim.lookup(df, "date_t"))


def baseline_predictions(df: pd.DataFrame, clim: Climatology, damped: DampedAnomaly) -> dict:
    """Bốn baseline (tên hiển thị -> mảng dự báo)."""
    c_t, c_g = clim.lookup(df, "date_t"), clim.lookup(df, "date_target")
    lst_t = df["LST_t"].to_numpy(float)
    return {
        "Persistence": lst_t,
        "Climatology": c_g,
        "Anomaly persistence": c_g + (lst_t - c_t),
        "Damped anomaly": damped.predict(df),
    }


# ---------------------------------------------------------------------
# VARIANT = kiểu target × bộ feature
# ---------------------------------------------------------------------

class Variant:
    MODES = ("absolute", "delta", "anomaly")
    FSETS = ("base", "aug")

    def __init__(self, mode: str = "absolute", fset: str = "base", clim: Climatology = None):
        if mode not in self.MODES or fset not in self.FSETS:
            raise ValueError(f"Variant không hợp lệ: {mode}:{fset} (mode ∈ {self.MODES}, fset ∈ {self.FSETS})")
        self.mode, self.fset, self.clim = mode, fset, clim

    @classmethod
    def parse(cls, text: str, clim: Climatology = None) -> "Variant":
        mode, _, fset = text.strip().partition(":")
        return cls(mode, fset or "base", clim)

    @property
    def name(self) -> str:
        return f"{self.mode}:{self.fset}"

    @property
    def needs_clim(self) -> bool:
        return self.mode == "anomaly" or self.fset == "aug"

    @property
    def features(self) -> list:
        return FEATURES + (AUG_COLS if self.fset == "aug" else [])

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.needs_clim:
            if self.clim is None:
                raise RuntimeError("Variant cần climatology nhưng chưa được fit.")
            out["clim_t"] = self.clim.lookup(out, "date_t")
            out["clim_target"] = self.clim.lookup(out, "date_target")
            out["anom_t"] = out["LST_t"].to_numpy(float) - out["clim_t"].to_numpy(float)
        return out

    def base(self, dfp: pd.DataFrame) -> np.ndarray:
        if self.mode == "delta":
            return dfp["LST_t"].to_numpy(float)
        if self.mode == "anomaly":
            return dfp["clim_target"].to_numpy(float)
        return np.zeros(len(dfp))

    def y(self, dfp: pd.DataFrame) -> np.ndarray:
        return dfp[TARGET].to_numpy(float) - self.base(dfp)

    def predict(self, model, dfp: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict(dfp[self.features]), dtype=float) + self.base(dfp)

    def to_dict(self) -> dict:
        return {"target_mode": self.mode, "feature_set": self.fset, "climatology": self.clim.to_dict() if self.clim is not None else None}

    @classmethod
    def from_dict(cls, d: dict) -> "Variant":
        clim = Climatology.from_dict(d["climatology"]) if d.get("climatology") else None
        return cls(d["target_mode"], d["feature_set"], clim)


def parse_variants(text: str, clim: Climatology) -> list:
    return [Variant.parse(t, clim) for t in text.split(",") if t.strip()]


# ---------------------------------------------------------------------
# HUẤN LUYỆN 4 HỌ MÔ HÌNH
# ---------------------------------------------------------------------

def fit_model(key: str, X_tr, y_tr, X_va, y_va, seed: int = 42, n_jobs: int = -1, params: dict = None,
              rf_trees: int = 200, max_estimators: int = 2000, es_rounds: int = 50, min_leaf: int = 50, fixed_trees: int = None):
    """Huấn luyện 1 mô hình; early stopping trên (X_va, y_va) cho 3 họ boosting.
    fixed_trees: nếu đặt -> KHÔNG early stopping, dùng đúng số cây này (X_va/y_va bị bỏ qua; truyền None được). Dùng cho CV (15),
    nơi tập giữ lại không được phép tham gia dừng sớm. Với RF, fixed_trees thay rf_trees.
    min_leaf: số dòng tối thiểu mỗi lá (XGBoost min_child_weight, LightGBM min_child_samples, RF min_samples_leaf).
    Các dòng CÙNG NGÀY chia sẻ thời tiết nên tương quan rất mạnh -> cây có lá nhỏ sẽ 'ghi nhớ' từng ngày (thấy rõ
    khi early stopping dừng ở vài cây); lá lớn hơn giảm hiện tượng này.
    Trả về (model, n_trees hoặc None, giây). RMSE trên y đã biến đổi == RMSE trên LST gốc (base chỉ là hằng theo dòng)."""
    params = dict(params or {})
    t0 = time.time()
    if key == "xgb":
        from xgboost import XGBRegressor

        p = dict(n_estimators=max_estimators, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=min_leaf, tree_method="hist", random_state=seed, n_jobs=n_jobs, early_stopping_rounds=es_rounds)
        p.update(params)
        if fixed_trees:
            p.pop("early_stopping_rounds", None)
            p["n_estimators"] = int(fixed_trees)
        m = XGBRegressor(**p)
        if fixed_trees:
            m.fit(X_tr, y_tr, verbose=False)
            n_trees = int(fixed_trees)
        else:
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            n_trees = int(m.best_iteration) + 1
    elif key == "lgbm":
        import warnings

        import lightgbm as lgb

        warnings.filterwarnings("ignore", message=".*eval_set.*")
        p = dict(n_estimators=max_estimators, learning_rate=0.05, num_leaves=63, min_child_samples=min_leaf, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs, verbose=-1)
        p.update(params)
        if fixed_trees:
            p["n_estimators"] = int(fixed_trees)
        m = lgb.LGBMRegressor(**p)
        if fixed_trees:
            m.fit(X_tr, y_tr)
            n_trees = int(fixed_trees)
        else:
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse", callbacks=[lgb.early_stopping(es_rounds, verbose=False)])
            n_trees = int(m.best_iteration_)
    elif key == "cat":
        from catboost import CatBoostRegressor

        p = dict(iterations=max_estimators, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=seed, verbose=0,
                 thread_count=n_jobs if n_jobs and n_jobs > 0 else -1, allow_writing_files=False, early_stopping_rounds=es_rounds)
        p.update(params)
        if fixed_trees:
            p.pop("early_stopping_rounds", None)
            p["iterations"] = int(fixed_trees)
        m = CatBoostRegressor(**p)
        if fixed_trees:
            m.fit(X_tr, y_tr)
            n_trees = int(fixed_trees)
        else:
            m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            n_trees = int(m.get_best_iteration()) + 1
    elif key == "rf":
        from sklearn.ensemble import RandomForestRegressor

        p = dict(n_estimators=int(fixed_trees or rf_trees), max_depth=20, min_samples_leaf=max(3, min_leaf // 5), n_jobs=n_jobs, random_state=seed)
        p.update(params)
        m = RandomForestRegressor(**p)
        m.fit(X_tr, y_tr)
        n_trees = None
    else:
        raise ValueError(f"Họ mô hình không hợp lệ: {key} (xgb|lgbm|cat|rf)")
    return m, n_trees, time.time() - t0


def make_bundle(model, variant: Variant, model_key: str, extra: dict = None) -> dict:
    """Gói lưu .pkl — nạp lại bằng predict_bundle(); KHÁC format 06 (có target_mode + climatology) nên lưu file RIÊNG (best_model_v2*.pkl)."""
    b = {"model": model, "features": variant.features, "model_key": model_key, "model_name": MODEL_LABELS.get(model_key, model_key),
         "target_mode": variant.mode, "feature_set": variant.fset, "variant": variant.to_dict(), "created": now_iso()}
    b.update(extra or {})
    return b


def predict_bundle(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    v = Variant.from_dict(bundle["variant"])
    return v.predict(bundle["model"], v.prepare(df))


# ---------------------------------------------------------------------
# WALK-FORWARD THEO NGÀY CÓ PURGE (cùng logic với 06_train_ml_models.py)
# ---------------------------------------------------------------------

def purged_walk_forward_splits(df: pd.DataFrame, n_splits: int = 3):
    """Chia ngày t thành n_splits+1 khối liên tiếp; fold k huấn luyện trên các khối trước, xác thực trên khối kế tiếp;
    LOẠI khỏi train mọi dòng có date_target >= ngày đầu của khối xác thực. Trả list (train_idx, val_idx) vị trí."""
    d_t = pd.to_datetime(df["date_t"]).to_numpy()
    d_n = pd.to_datetime(df["date_target"]).to_numpy()
    dates = np.unique(d_t)
    blocks = np.array_split(dates, n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_dates = np.concatenate(blocks[: k + 1])
        val_dates = blocks[k + 1]
        tr_mask = np.isin(d_t, train_dates) & (d_n < val_dates.min())
        va_mask = np.isin(d_t, val_dates)
        tr_idx, va_idx = np.where(tr_mask)[0], np.where(va_mask)[0]
        if len(tr_idx) and len(va_idx):
            splits.append((tr_idx, va_idx))
    if not splits:
        raise SystemExit("❌ Không tạo được fold walk-forward nào — dữ liệu Train quá ít ngày.")
    return splits


# ---------------------------------------------------------------------
# KHOẢNG TIN CẬY: split conformal theo nhóm delta_t + CQR
# ---------------------------------------------------------------------

def conformal_quantile(scores, coverage: float) -> float:
    """Phân vị conformal hữu hạn mẫu: phần tử thứ ceil((n+1)·coverage) của điểm không phù hợp đã sắp."""
    s = np.sort(np.asarray(scores, dtype=float))
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    k = min(int(np.ceil((n + 1) * coverage)), n)
    return float(s[k - 1])


def fit_binned_conformal(scores, delta_t, coverage: float = 0.9, min_cal: int = 200) -> dict:
    """Phân vị conformal RIÊNG cho từng nhóm delta_t. Nhóm có < min_cal mẫu hiệu chuẩn dùng
    max(phân vị toàn cục, phân vị của nhóm delta_t NHỎ HƠN liền kề) — thận trọng (không để nhóm thưa, thường là delta_t lớn,
    có khoảng hẹp hơn nhóm ít xa hơn)."""
    scores = np.asarray(scores, dtype=float)
    bins = delta_bin(delta_t)
    q_glob = conformal_quantile(scores, coverage)
    out_bins, n_bins, prev = {}, {}, q_glob
    for lab in DELTA_LABELS:
        m = bins == lab
        n_bins[lab] = int(m.sum())
        if m.sum() >= min_cal:
            q = conformal_quantile(scores[m], coverage)
        else:
            q = max(q_glob, prev)
        out_bins[lab] = float(q)
        prev = q
    return {"global": float(q_glob), "bins": out_bins, "n": n_bins}


def lookup_q(cfg: dict, delta_t) -> np.ndarray:
    bins = delta_bin(delta_t)
    return np.array([cfg["bins"].get(b, cfg["global"]) if b is not None else cfg["global"] for b in bins], dtype=float)


def quantile_alphas(coverage: float):
    a = (1 - coverage) / 2
    return a, 1 - a


def fit_quantile_xgb(X_tr, y_tr, coverage: float = 0.9, seed: int = 42, n_jobs: int = -1, n_estimators: int = 400, params: dict = None):
    """XGBoost phân vị (reg:quantileerror) cho 2 phân vị (1−coverage)/2 và 1−(1−coverage)/2 trong 1 mô hình.
    Không early stopping (phân vị Val được dành cho hiệu chuẩn CQR)."""
    from xgboost import XGBRegressor

    lo, hi = quantile_alphas(coverage)
    p = dict(objective="reg:quantileerror", quantile_alpha=np.array([lo, hi]), n_estimators=n_estimators, learning_rate=0.05,
             max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=seed, n_jobs=n_jobs)
    p.update(params or {})
    m = XGBRegressor(**p)
    m.fit(X_tr, y_tr, verbose=False)
    return m


def predict_quantiles(qbundle: dict, df: pd.DataFrame):
    """(lo, hi) trong không gian LST gốc từ bundle phân vị."""
    v = Variant.from_dict(qbundle["variant"])
    dfp = v.prepare(df)
    raw = np.asarray(qbundle["model"].predict(dfp[v.features]), dtype=float)
    base = v.base(dfp)
    lo, hi = raw[:, 0] + base, raw[:, 1] + base
    return np.minimum(lo, hi), np.maximum(lo, hi)


def predict_with_interval(bundle: dict, calib: dict, df: pd.DataFrame, method: str = "conformal", qbundle: dict = None) -> pd.DataFrame:
    """DataFrame (index như df) gồm LST_pred, LST_lo, LST_hi, delta_t_bin, delta_t_out_of_range.
    method='conformal': ŷ ± q(delta_t) | method='cqr': [lo − q, hi + q] (kẹp để ŷ nằm trong khoảng)."""
    pred = predict_bundle(bundle, df)
    dts = df["delta_t"].to_numpy(float)
    if method == "cqr":
        if qbundle is None or "cqr" not in calib:
            raise SystemExit("❌ method='cqr' cần quantile_model_v2.pkl và mục 'cqr' trong interval_calibration.json (chạy 13).")
        lo_r, hi_r = predict_quantiles(qbundle, df)
        q = lookup_q(calib["cqr"], dts)
        lo, hi = np.minimum(lo_r - q, pred), np.maximum(hi_r + q, pred)
    else:
        q = lookup_q(calib["conformal"], dts)
        lo, hi = pred - q, pred + q
    rng = calib.get("train_delta_t", {})
    oor = np.zeros(len(df), dtype=bool)
    if rng:
        oor = (dts < rng.get("min", -np.inf)) | (dts > rng.get("max", np.inf))
    return pd.DataFrame({"LST_pred": pred, "LST_lo": lo, "LST_hi": hi, "delta_t_bin": delta_bin(dts),
                         "delta_t_out_of_range": oor}, index=df.index)


def load_bundle(data_dir: str, name: str = "best_model_v2.pkl") -> dict:
    import pickle

    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise SystemExit(f"❌ Không thấy {path} — chạy 11_improved_models.py (hoặc 12) trước.")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, obj) -> None:
    import pickle

    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------
# TIỆN ÍCH VẼ ẢNH
# ---------------------------------------------------------------------

def style_axes(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
