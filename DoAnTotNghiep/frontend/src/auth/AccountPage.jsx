import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthShell, Logo, Field, StrengthMeter, passwordOk, APP_NAME } from './AuthShared';
import { getMe, getActivity, changePassword, logout } from './authApi';

const ROLE = { admin: 'Quản trị viên', user: 'Người dùng', guest: 'Khách' };
const fmtDate = (d) => (d ? new Date(d).toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' }) : '—');
const fmtTime = (d) => new Date(d).toLocaleString('vi-VN', { hour12: false });

export default function AccountPage() {
  const nav = useNavigate();
  const [me, setMe] = useState(null);
  const [acts, setActs] = useState([]);
  const [loadErr, setLoadErr] = useState('');
  const [pw, setPw] = useState({ current: '', next: '', confirm: '' });
  const [msg, setMsg] = useState(null); // {type:'ok'|'bad', text}
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getMe().then(setMe).catch(() => nav('/login', { state: { from: '/account' }, replace: true }));
    getActivity().then((a) => setActs(Array.isArray(a) ? a : a.items || []));
  }, [nav]);

  const signOut = () => { logout(); nav('/login', { replace: true }); };

  async function savePw(e) {
    e.preventDefault(); setMsg(null);
    if (!pw.current) return setMsg({ type: 'bad', text: 'Vui lòng nhập mật khẩu hiện tại.' });
    if (!passwordOk(pw.next)) return setMsg({ type: 'bad', text: 'Mật khẩu mới chưa đáp ứng đủ yêu cầu.' });
    if (pw.next !== pw.confirm) return setMsg({ type: 'bad', text: 'Mật khẩu xác nhận không khớp.' });
    if (pw.next === pw.current) return setMsg({ type: 'bad', text: 'Mật khẩu mới phải khác mật khẩu hiện tại.' });
    setBusy(true);
    try {
      await changePassword({ currentPassword: pw.current, newPassword: pw.next });
      setPw({ current: '', next: '', confirm: '' });
      setMsg({ type: 'ok', text: 'Đã đổi mật khẩu thành công.' });
    } catch (ex) { setMsg({ type: 'bad', text: ex.message }); } finally { setBusy(false); }
  }

  if (!me) return <AuthShell><div className="au-card" style={{ textAlign: 'center' }}>{loadErr || 'Đang tải hồ sơ…'}</div></AuthShell>;

  const list = acts.length ? acts : [{ label: 'Đăng nhập trên web', at: new Date().toISOString(), current: true }];

  return (
    <AuthShell wide>
      <div className="ac-bar">
        <div className="ac-brand"><Logo size={40} />{APP_NAME}</div>
        <div style={{ display: 'flex', gap: 12 }}>
          <Link className="au-btn ghost" style={{ width: 'auto', padding: '0 20px', height: 44, lineHeight: '42px' }} to="/">Về bản đồ</Link>
          <button className="au-btn ghost" onClick={signOut}>Đăng xuất</button>
        </div>
      </div>

      <section className="ac-panel">
        <div className="ac-head">
          <div>
            <p className="ac-eyebrow">Thông tin tài khoản</p>
            <h1 className="ac-h">HỒ SƠ NGƯỜI DÙNG</h1>
            <span className="ac-role">{ROLE[me.role] || ROLE.user}</span>
          </div>
          <div className="ac-side">
            <button className="au-btn pri" disabled>Chỉnh sửa hồ sơ</button>
            <div className="au-msg">Chưa hỗ trợ chỉnh sửa hồ sơ.</div>
          </div>
        </div>
        <div className="ac-stats">
          <div className="ac-stat"><small>Tên đăng nhập</small><b>{me.username}</b></div>
          <div className="ac-stat"><small>Email</small><b>{me.email}</b></div>
          <div className="ac-stat"><small>Thành viên từ</small><b>{fmtDate(me.created_at || me.createdAt)}</b></div>
        </div>
      </section>

      <div className="ac-grid">
        <section className="ac-panel">
          <h2 className="ac-h2">Cập nhật email</h2>
          <div className="au-alert">Chưa hỗ trợ đổi email.</div>
          <Field id="cur-email" label="Email hiện tại"><input id="cur-email" className="au-input" value={me.email} disabled readOnly /></Field>
          <Field id="new-email" label="Email mới"><input id="new-email" className="au-input" placeholder="email-moi@example.com" disabled /></Field>
          <button className="au-btn pri ac-btn" disabled>Lưu email</button>
        </section>

        <form className="ac-panel" onSubmit={savePw} noValidate>
          <h2 className="ac-h2">Đổi mật khẩu</h2>
          {msg && <div className={`au-alert ${msg.type}`} role="alert">{msg.text}</div>}
          <Field id="pw-cur" label="Mật khẩu hiện tại">
            <input id="pw-cur" type="password" className="au-input" placeholder="Mật khẩu hiện tại" autoComplete="current-password" value={pw.current} onChange={(e) => setPw({ ...pw, current: e.target.value })} />
          </Field>
          <Field id="pw-new" label="Mật khẩu mới">
            <input id="pw-new" type="password" className="au-input" placeholder="Mật khẩu mới" maxLength={64} autoComplete="new-password" value={pw.next} onChange={(e) => setPw({ ...pw, next: e.target.value })} />
            <StrengthMeter value={pw.next} />
          </Field>
          <Field id="pw-cf" label="Xác nhận mật khẩu mới">
            <input id="pw-cf" type="password" className="au-input" placeholder="Nhập lại mật khẩu mới" maxLength={64} autoComplete="new-password" value={pw.confirm} onChange={(e) => setPw({ ...pw, confirm: e.target.value })} />
          </Field>
          <button className="au-btn pri ac-btn" disabled={busy}>{busy ? 'Đang lưu…' : 'Lưu mật khẩu'}</button>
        </form>
      </div>

      <section className="ac-panel">
        <h2 className="ac-h2">Hoạt động gần đây</h2>
        <ul className="ac-act">
          {list.map((a, i) => (
            <li key={i}>
              <span>{a.label || 'Đăng nhập trên web'} · {fmtTime(a.at || a.created_at)}{a.current && <> <em>(phiên này)</em></>}</span>
            </li>
          ))}
        </ul>
      </section>
    </AuthShell>
  );
}
