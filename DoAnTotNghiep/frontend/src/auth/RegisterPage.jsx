import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthShell, Logo, Field, StrengthMeter, passwordOk } from './AuthShared';
import { register } from './authApi';

const MONTHS = ['Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6', 'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'Tháng 12'];
const pad = (n) => String(n).padStart(2, '0');

export default function RegisterPage() {
  const nav = useNavigate();
  const [f, setF] = useState({ username: '', email: '', password: '', confirm: '', month: '', day: '', year: '' });
  const [err, setErr] = useState({});
  const [top, setTop] = useState('');
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });

  const now = new Date().getFullYear();
  const years = Array.from({ length: 100 }, (_, i) => now - i);
  const maxDay = f.month ? new Date(f.year || 2000, Number(f.month), 0).getDate() : 31;

  function validate() {
    const v = {};
    if (!/^[A-Za-z0-9]{3,12}$/.test(f.username)) v.username = 'Tên đăng nhập gồm 3–12 chữ cái và số, không dấu, không khoảng trắng.';
    if (!/^\S+@\S+\.\S+$/.test(f.email)) v.email = 'Địa chỉ email không hợp lệ.';
    if (!passwordOk(f.password)) v.password = 'Mật khẩu chưa đáp ứng đủ các yêu cầu bên dưới.';
    if (f.password !== f.confirm) v.confirm = 'Mật khẩu xác nhận không khớp.';
    const n = [f.month, f.day, f.year].filter(Boolean).length;
    if (n > 0 && n < 3) v.birth = 'Chọn đủ tháng, ngày, năm hoặc để trống cả ba.';
    return v;
  }

  async function onSubmit(e) {
    e.preventDefault();
    const v = validate(); setErr(v); setTop('');
    if (Object.keys(v).length) return;
    setBusy(true);
    try {
      await register({
        username: f.username, email: f.email.trim(), password: f.password,
        birthdate: f.year ? `${f.year}-${pad(f.month)}-${pad(f.day)}` : null,
      });
      nav('/login', { state: { registered: true } });
    } catch (ex) { setTop(ex.message); } finally { setBusy(false); }
  }

  return (
    <AuthShell>
      <form className="au-card" onSubmit={onSubmit} noValidate>
        <div className="au-logo"><Logo size={64} /></div>
        <h1 className="au-title">Đăng ký<small>/ Tạo tài khoản</small></h1>
        {top && <div className="au-alert bad" role="alert">{top}</div>}

        <Field id="username" label="Tên đăng nhập" hint="3–12 chữ cái và số" error={err.username}>
          <input id="username" className="au-input" maxLength={12} autoComplete="username" value={f.username} onChange={set('username')} />
        </Field>
        <Field id="email" label="Địa chỉ email" error={err.email}>
          <input id="email" type="email" className="au-input" autoComplete="email" value={f.email} onChange={set('email')} />
        </Field>
        <Field id="password" label="Mật khẩu" error={err.password}>
          <input id="password" type="password" className="au-input" maxLength={64} autoComplete="new-password" value={f.password} onChange={set('password')} />
          <StrengthMeter value={f.password} />
        </Field>
        <Field id="confirm" label="Xác nhận mật khẩu" error={err.confirm}>
          <input id="confirm" type="password" className="au-input" maxLength={64} autoComplete="new-password" value={f.confirm} onChange={set('confirm')} />
          {f.confirm && !err.confirm && (f.confirm === f.password
            ? <div className="au-msg ok">✓ Mật khẩu khớp</div>
            : <div className="au-msg bad">Mật khẩu chưa khớp</div>)}
        </Field>

        <Field id="month" label="Ngày sinh (không bắt buộc)" error={err.birth}>
          <select id="month" className="au-input" value={f.month} onChange={set('month')}>
            <option value="">Tháng</option>
            {MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
          </select>
          <div className="au-row" style={{ marginTop: 16 }}>
            <select aria-label="Ngày" className="au-input" value={f.day} onChange={set('day')}>
              <option value="">Ngày</option>
              {Array.from({ length: maxDay }, (_, i) => <option key={i} value={i + 1}>{i + 1}</option>)}
            </select>
            <select aria-label="Năm" className="au-input" value={f.year} onChange={set('year')}>
              <option value="">Năm</option>
              {years.map((y) => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
        </Field>

        <button className="au-btn pri" disabled={busy}>{busy ? 'Đang tạo tài khoản…' : 'Tạo tài khoản'}</button>
        <div style={{ height: 16 }} />
        <Link className="au-btn ghost" to="/login">Đăng nhập</Link>
      </form>
    </AuthShell>
  );
}
