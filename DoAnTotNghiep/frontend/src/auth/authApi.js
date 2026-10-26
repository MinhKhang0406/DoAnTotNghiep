// Gọi backend Node/Express. Sửa EP nếu route backend của bạn khác.
import { API_BASE as BASE } from './config';

export const EP = {
  register: '/api/auth/register',
  login: '/api/auth/login',
  me: '/api/auth/me',
  changePassword: '/api/auth/change-password',
  activity: '/api/auth/activity',
};
const KEY = 'uhi_token';
const token = () => localStorage.getItem(KEY) || sessionStorage.getItem(KEY);
export const isAuthed = () => !!token();
export const logout = () => { localStorage.removeItem(KEY); sessionStorage.removeItem(KEY); };

async function req(path, { method = 'GET', body, auth } = {}) {
  const res = await fetch(BASE + path, {
    method,
    headers: { 'Content-Type': 'application/json', ...(auth && token() ? { Authorization: `Bearer ${token()}` } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try { data = await res.json(); } catch { /* không có body */ }
  if (!res.ok) throw new Error(data.message || data.error || `Lỗi máy chủ (${res.status})`);
  return data;
}

export const register = (b) => req(EP.register, { method: 'POST', body: b });
export async function login({ identifier, password, remember }) {
  const d = await req(EP.login, { method: 'POST', body: { identifier, password, remember } });
  logout();
  (remember ? localStorage : sessionStorage).setItem(KEY, d.token); // remember = giữ 7 ngày (backend cấp token 7d)
  return d;
}
export const getMe = () => req(EP.me, { auth: true });
export const changePassword = (b) => req(EP.changePassword, { method: 'POST', body: b, auth: true });
export const getActivity = () => req(EP.activity, { auth: true }).catch(() => []);
