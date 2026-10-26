import { Link } from 'react-router-dom';
import { AuthShell, Logo } from './AuthShared';

export default function ForgotPasswordPage() {
  return (
    <AuthShell>
      <div className="au-card">
        <div className="au-logo"><Logo size={64} /></div>
        <h1 className="au-title">Quên mật khẩu</h1>
        <div className="au-alert">
          Chưa hỗ trợ khôi phục mật khẩu qua email. Vui lòng liên hệ quản trị viên để được đặt lại mật khẩu.
        </div>
        <Link className="au-btn ghost" to="/login">Quay lại đăng nhập</Link>
      </div>
    </AuthShell>
  );
}
