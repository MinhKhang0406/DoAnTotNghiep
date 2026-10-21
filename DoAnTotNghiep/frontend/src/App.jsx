import React, { useState, useEffect, useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, GeoJSON } from 'react-leaflet';
import axios from 'axios';
import chroma from 'chroma-js';
import 'leaflet/dist/leaflet.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

// Đăng ký các module của Chart.js
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [chartData, setChartData] = useState(null);
  // CẬP NHẬT 2026-09-09: state cho layer ranh giới hành chính (Tuần 12) —
  // dùng để vẽ boundary + phát hiện quận/huyện "không có dữ liệu"
  const [boundary, setBoundary] = useState(null);
  const [boundaryError, setBoundaryError] = useState(false);

  useEffect(() => {
    // Gọi API lấy dữ liệu dự báo
    axios.get('http://localhost:3000/api/lst-data')
      .then(response => {
        if (response.data.status === 'success') {
          const rawData = response.data.data;
          setData(rawData);

          // XỬ LÝ DỮ LIỆU BIỂU ĐỒ: Tính nhiệt độ trung bình theo Quận/Huyện
          const districtStats = {};
          rawData.forEach(point => {
            const name = point.ADM2_NAME;
            const temp = parseFloat(point.LST_pred);
            if (!districtStats[name]) {
              districtStats[name] = { sum: 0, count: 0 };
            }
            districtStats[name].sum += temp;
            districtStats[name].count += 1;
          });

          const labels = Object.keys(districtStats);
          const avgTemps = labels.map(name => (districtStats[name].sum / districtStats[name].count).toFixed(1));

          setChartData({
            labels: labels,
            datasets: [
              {
                label: 'Nhiệt độ dự báo TB (°C)',
                data: avgTemps,
                backgroundColor: 'rgba(244, 109, 67, 0.7)',
                borderColor: 'rgba(215, 48, 39, 1)',
                borderWidth: 1,
              }
            ]
          });
        }
        setLoading(false);
      })
      .catch(error => {
        console.error("Lỗi khi tải API lst-data:", error);
        setLoading(false);
      });

    // Gọi API lấy ranh giới hành chính (Tuần 12) — song song, không chặn
    // việc hiển thị các điểm dự báo nếu boundary lỗi/chưa có.
    axios.get('http://localhost:3000/api/boundary')
      .then(response => {
        if (response.data.status === 'success') {
          setBoundary(response.data.data);
        }
      })
      .catch(error => {
        console.error("Lỗi khi tải API boundary (đã chạy 09_export_to_postgis.py chưa?):", error);
        setBoundaryError(true);
      });
  }, []);

  const colorScale = chroma.scale(['#313695', '#74add1', '#ffffbf', '#f46d43', '#a50026']).domain([25, 45]);

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Nhiệt độ trung bình theo khu vực' },
    },
  };

  // Số điểm dự báo theo từng quận/huyện — dùng để tô "không có dữ liệu"
  // cho quận/huyện nào có 0 điểm (thường do mây che hết vùng đó trong cửa
  // sổ thời gian cần thiết — đúng nguyên tắc "không nội suy" của đồ án:
  // hiển thị rõ là THIẾU dữ liệu, không tự vẽ giá trị giả định).
  const pointCountByDistrict = useMemo(() => {
    const counts = {};
    data.forEach(point => {
      const name = point.ADM2_NAME;
      counts[name] = (counts[name] || 0) + 1;
    });
    return counts;
  }, [data]);

  const boundaryStyle = (feature) => {
    const name = feature.properties && feature.properties.ADM2_NAME;
    const hasData = name && pointCountByDistrict[name] > 0;
    if (hasData) {
      // Có dữ liệu: chỉ viền mỏng, để điểm dự báo hiển thị rõ bên trên
      return { color: '#334155', weight: 1, fillOpacity: 0 };
    }
    // Không có dữ liệu: tô xám để phân biệt rõ, KHÔNG suy diễn giá trị
    return { color: '#94a3b8', weight: 1, fillColor: '#cbd5e1', fillOpacity: 0.55, dashArray: '4' };
  };

  const onEachBoundaryFeature = (feature, layer) => {
    const name = (feature.properties && feature.properties.ADM2_NAME) || 'Không rõ';
    const hasData = pointCountByDistrict[name] > 0;
    layer.bindPopup(
      hasData
        ? `<b>${name}</b><br/>Số điểm dự báo: ${pointCountByDistrict[name]}`
        : `<b>${name}</b><br/><i>Không có dữ liệu</i> (có thể do mây che hết vùng này trong cửa sổ thời gian cần thiết)`
    );
  };

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* HEADER */}
      <header style={{ padding: '15px 20px', backgroundColor: '#1e293b', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ margin: 0 }}>WebGIS Dự Báo Nhiệt Độ Bề Mặt Đất (LST)</h2>
          <span style={{ fontSize: '14px', color: '#cbd5e1' }}>Khu vực: TP.HCM | Sinh viên thực hiện: Nguyễn Trần Minh Khang</span>
        </div>
        <div>
          <strong>Tổng số pixel dự báo: {data.length}</strong>
        </div>
      </header>

      {/* MAIN CONTENT: Chia 2 cột */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'row' }}>

        {/* CỘT TRÁI: BẢN ĐỒ (70% CHIỀU RỘNG) */}
        <div style={{ flex: 7, position: 'relative' }}>
          {loading ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <h2>Đang tải dữ liệu không gian...</h2>
            </div>
          ) : (
            <MapContainer center={[10.8231, 106.6297]} zoom={10} style={{ height: '100%', width: '100%' }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              />

              {/* Layer ranh giới hành chính + trạng thái "không có dữ liệu" (Tuần 12) */}
              {boundary && (
                <GeoJSON
                  key={`boundary-${data.length}`}
                  data={boundary}
                  style={boundaryStyle}
                  onEachFeature={onEachBoundaryFeature}
                />
              )}

              {data.map((point, index) => {
                const lstPred = parseFloat(point.LST_pred);
                const fillColor = colorScale(lstPred).hex();
                return (
                  <CircleMarker key={index} center={[point.lat, point.lon]} radius={5} fillColor={fillColor} color={fillColor} weight={1} fillOpacity={0.8}>
                    <Popup>
                      <div style={{ fontSize: '14px' }}>
                        <h4 style={{ margin: '0 0 5px 0', borderBottom: '1px solid #ccc', paddingBottom: '3px' }}>{point.ADM2_NAME}</h4>
                        Thời gian: <b>{point.date_t}</b><br/>
                        LST Quá khứ: {parseFloat(point.LST_t).toFixed(1)} °C<br/>
                        <b>LST Dự báo: <span style={{ color: 'red', fontSize: '16px' }}>{lstPred.toFixed(1)} °C</span></b>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>
          )}
          {boundaryError && (
            <div style={{ position: 'absolute', bottom: 10, left: 10, background: '#fef2f2', color: '#991b1b', padding: '6px 12px', borderRadius: '6px', fontSize: '13px', boxShadow: '0 1px 3px rgba(0,0,0,0.2)' }}>
              ⚠️ Chưa tải được lớp ranh giới hành chính (kiểm tra đã chạy 09_export_to_postgis.py chưa)
            </div>
          )}
        </div>

        {/* CỘT PHẢI: BIỂU ĐỒ & THÔNG TIN (30% CHIỀU RỘNG) */}
        <div style={{ flex: 3, padding: '20px', backgroundColor: '#f8fafc', borderLeft: '2px solid #e2e8f0', overflowY: 'auto' }}>
          {chartData ? (
            <Bar options={chartOptions} data={chartData} />
          ) : (
            <p>Đang xử lý dữ liệu biểu đồ...</p>
          )}

          <div style={{ marginTop: '40px', padding: '15px', backgroundColor: '#ffffff', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h3 style={{ marginTop: 0, color: '#334155', borderBottom: '2px solid #e2e8f0', paddingBottom: '10px' }}>Thông số Kỹ thuật</h3>
            <ul style={{ paddingLeft: '20px', color: '#475569', lineHeight: '1.8' }}>
              <li><b>Thuật toán:</b> XGBoost Regressor</li>
              <li><b>Độ phân giải không gian:</b> 30m (Landsat 8/9)</li>
              <li><b>Biến ảnh hưởng mạnh nhất:</b> NDBI (Đất xây dựng), Nhiệt độ không khí</li>
              <li><b>Hệ tọa độ:</b> EPSG:4326</li>
              <li><b>Database:</b> PostgreSQL + PostGIS</li>
            </ul>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
