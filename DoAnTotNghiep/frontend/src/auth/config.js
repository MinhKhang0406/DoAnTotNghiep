// Địa chỉ backend dùng chung cho bản đồ và các trang đăng nhập/đăng ký/tài khoản.
// Đặt VITE_API_URL trong frontend/.env, ví dụ: VITE_API_URL=http://localhost:3000
export const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:3000').replace(/\/$/, '');
