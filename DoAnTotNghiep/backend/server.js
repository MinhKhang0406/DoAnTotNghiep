// CẬP NHẬT 2026-09-09:
//   1. BỔ SUNG Rate Limiting (Tuần 11 đề cương: "Cấu hình Rate Limiting
//      chống spam request để bảo mật API" — bản trước hoàn toàn chưa có).
//   2. BỔ SUNG endpoint /api/boundary trả ranh giới hành chính (đọc từ
//      bảng admin_boundaries mà 09_export_to_postgis.py mới đẩy lên) —
//      cần để frontend vẽ layer boundary + hiển thị "không có dữ liệu"
//      cho vùng không có điểm dự báo nào (Tuần 12).
//   3. Đọc chuỗi kết nối DB từ biến môi trường LST_DB_URL trước (đồng bộ
//      với 09_export_to_postgis.py), có fallback cho máy dev cá nhân —
//      ĐỪNG commit mật khẩu thật lên Git, set LST_DB_URL trong .env khi
//      deploy thật.
//
// Cần cài thêm: npm install express-rate-limit

const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const { Pool } = require('pg');

const app = express();
// Cho phép Frontend React truy cập
app.use(cors());

// Rate Limiting chống spam request (Tuần 11 - bảo mật API)
const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // cửa sổ 15 phút
    max: 100,                 // tối đa 100 request / IP / cửa sổ
    standardHeaders: true,
    legacyHeaders: false,
    message: { status: 'error', error: 'Quá nhiều request, vui lòng thử lại sau ít phút.' },
});
app.use('/api/', apiLimiter);

// Đọc chuỗi kết nối từ biến môi trường trước (đồng bộ với 09_export_to_postgis.py):
const pool = new Pool({
    connectionString: process.env.LST_DB_URL || 'postgresql://postgres:123456@localhost:5432/postgres'
});

// API Endpoint trả về dữ liệu dự báo LST
app.get('/api/lst-data', async (req, res) => {
    try {
        const query = 'SELECT * FROM lst_predictions;';
        const result = await pool.query(query);

        res.json({
            status: 'success',
            total_points: result.rowCount,
            data: result.rows
        });
    } catch (err) {
        console.error('Lỗi truy vấn Database:', err);
        res.status(500).json({ error: 'Lỗi Server' });
    }
});

// API Endpoint trả về ranh giới hành chính (GeoJSON) — dùng để vẽ layer
// boundary trên bản đồ và tính "quận/huyện nào có 0 điểm dự báo" để hiện
// trạng thái "không có dữ liệu" ở frontend.
app.get('/api/boundary', async (req, res) => {
    try {
        const query = `
            SELECT jsonb_build_object(
                'type', 'FeatureCollection',
                'features', COALESCE(jsonb_agg(feature), '[]'::jsonb)
            ) AS geojson
            FROM (
                SELECT jsonb_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(geometry)::jsonb,
                    'properties', to_jsonb(t) - 'geometry'
                ) AS feature
                FROM admin_boundaries AS t
            ) AS features;
        `;
        const result = await pool.query(query);
        res.json({ status: 'success', data: result.rows[0].geojson });
    } catch (err) {
        console.error('Lỗi truy vấn ranh giới:', err);
        res.status(500).json({
            error: 'Lỗi Server — đã chạy 09_export_to_postgis.py để tạo bảng admin_boundaries chưa?'
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 Backend Server đã khởi chạy thành công!`);
    console.log(`👉 Dữ liệu dự báo:    http://localhost:${PORT}/api/lst-data`);
    console.log(`👉 Ranh giới hành chính: http://localhost:${PORT}/api/boundary`);
});
