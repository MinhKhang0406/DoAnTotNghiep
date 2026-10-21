import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { AuthShell, Logo, Field } from './AuthShared';
import { login } from './authApi';

export default function LoginPage() {
  const nav = useNavigate();
  const { state } = useLocation();
  const [f, setF] = useState({ identifier: '', password: '', remember: false });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    if (!f.identifier.trim() || !f.password) return setError('Vui lòng nhập email/tên đăng nhập và mật khẩu.');
    setBusy(true); setError('');
    try {
      await login({ identifier: f.identifier.trim(), password: f.password, remember: f.remember });
      nav(state?.from || '/', { replace: true });
    } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  return (
    <AuthShell>
      <form className="au-card" onSubmit={onSubmit} noValidate>
        <div className="au-logo"><Logo size={64} /></div>
        <h1 className="au-title">Đăng nhập<small>/ Điền thông tin</small></h1>
        {state?.registered && <div className="au-alert ok">Tạo tài khoản thành công. Hãy đăng nhập để tiếp tục.</div>}
        {error && <div className="au-alert bad" role="alert">{error}</div>}
        <Field id="identifier" label="Email hoặc tên đăng nhập">
          <input id="identifier" className="au-input" placeholder="Email / Tên đăng nhập" autoComplete="username"
            value={f.identifier} onChange={(e) => setF({ ...f, identifier: e.target.value })} />
        </Field>
        <Field id="password" label="Mật khẩu">
          <input id="password" type="password" className="au-input" placeholder="Mật khẩu" autoComplete="current-password"
            value={f.password} onChange={(e) => setF({ ...f, password: e.target.value })} />
        </Field>
        <label className="au-check">
          <input type="checkbox" checked={f.remember} onChange={(e) => setF({ ...f, remember: e.target.checked })} />
          Ghi nhớ đăng nhập trong 7 ngày
        </label>
        <button className="au-btn pri" disabled={busy}>{busy ? 'Đang đăng nhập…' : 'Đăng nhập'}</button>
        <Link className="au-link" to="/forgot-password">Quên mật khẩu?</Link>
        <Link className="au-btn ghost" to="/register">Tạo tài khoản</Link>
      </form>
    </AuthShell>
  );
}
