import { useEffect, useState } from 'react';
import './auth.css';

export const APP_VERSION = 'V1.0.0';
export const APP_NAME = 'Đảo nhiệt TP.HCM';

export function useTheme() {
  const [theme, setTheme] = useState(() => localStorage.getItem('uhi_theme') || 'dark');
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('uhi_theme', theme);
  }, [theme]);
  return [theme, () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))];
}

export function Logo({ size = 56 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-label={APP_NAME} role="img">
      <defs>
        <linearGradient id="uhi-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#3bb8f2" /><stop offset="1" stopColor="#ff8a3d" />
        </linearGradient>
      </defs>
      <path d="M32 4C21 4 13 12 13 22c0 14 19 36 19 36s19-22 19-36C51 12 43 4 32 4z" fill="none" stroke="url(#uhi-g)" strokeWidth="3" />
      <rect x="29" y="13" width="6" height="19" rx="3" fill="url(#uhi-g)" />
      <circle cx="32" cy="35" r="6" fill="url(#uhi-g)" />
    </svg>
  );
}

export function AuthShell({ children, wide }) {
  const [theme, toggle] = useTheme();
  return (
    <div className="au-page">
      <button type="button" className="au-theme" onClick={toggle} aria-label="Đổi giao diện sáng/tối">
        {theme === 'dark' ? '🌙 Tối' : '☀️ Sáng'}
      </button>
      <div className={`au-wrap${wide ? ' wide' : ''}`}>{children}</div>
      <span className="au-ver">{APP_VERSION}</span>
    </div>
  );
}

export function Field({ id, label, hint, error, children }) {
  return (
    <div className="au-field">
      <label className="au-label" htmlFor={id}>{label}</label>
      {children}
      {error ? <div className="au-msg bad">{error}</div> : hint ? <div className="au-msg">{hint}</div> : null}
    </div>
  );
}

export const RULES = [
  { key: 'len', label: '8–64 ký tự', test: (p) => p.length >= 8 && p.length <= 64 },
  { key: 'up', label: 'Một chữ hoa (A–Z)', test: (p) => /[A-Z]/.test(p) },
  { key: 'low', label: 'Một chữ thường (a–z)', test: (p) => /[a-z]/.test(p) },
  { key: 'num', label: 'Một chữ số (0–9)', test: (p) => /\d/.test(p) },
];
export const passwordOk = (p) => RULES.every((r) => r.test(p));
const LEVELS = ['', 'Yếu', 'Trung bình', 'Khá', 'Mạnh'];

export function StrengthMeter({ value }) {
  const score = RULES.filter((r) => r.test(value)).length;
  return (
    <div>
      <div className={`au-meter lv${score}`}>{[0, 1, 2, 3].map((i) => <i key={i} className={i < score ? 'on' : ''} />)}</div>
      <div className="au-strength-head"><span>Độ mạnh mật khẩu</span><b className={`lv${score}`}>{LEVELS[score]}</b></div>
      <ul className="au-rules">
        {RULES.map((r) => {
          const ok = r.test(value);
          return <li key={r.key} className={ok ? 'ok' : ''}><span>{ok ? '✓' : '○'}</span>{r.label}</li>;
        })}
      </ul>
    </div>
  );
}
