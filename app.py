import streamlit as st
from pystac_client import Client
import planetary_computer
import odc.stac
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ページ設定
st.set_page_config(page_title="衛星画像取得プラットフォーム", layout="wide")

# ==========================================
# 0. 定数・設定
# ==========================================
LOCATIONS = {
    "--- 国内 (日本) ---": None,
    "筑波宇宙センター": {"lat": 36.0652, "lon": 140.1272},
    "種子島宇宙センター": {"lat": 30.3749, "lon": 130.9582},
    "東京駅": {"lat": 35.6812, "lon": 139.7671},
    "いろは坂": {"lat": 36.7369, "lon": 139.5168},
    "富士山": {"lat": 35.3606, "lon": 138.7274},
    "桜島": {"lat": 31.5932, "lon": 130.6573},
    "能登半島 (珠洲市付近)": {"lat": 37.4363, "lon": 137.2608},
    "ナイタイ高原牧場 (北海道)": {"lat": 43.1972, "lon": 143.1797},
    "--- 海外 (アジア) ---": None,
    "北京 (中国)": {"lat": 39.9042, "lon": 116.4074},
    "ニューデリー (インド)": {"lat": 28.6139, "lon": 77.2090},
    "--- 海外 (北米) ---": None,
    "ワシントンD.C. (アメリカ)": {"lat": 38.9072, "lon": -77.0369},
    "オタワ (カナダ)": {"lat": 45.4215, "lon": -75.6972},
    "--- 海外 (南米) ---": None,
    "ブラジリア (ブラジル)": {"lat": -15.7975, "lon": -47.8919},
    "ブエノスアイレス (アルゼンチン)": {"lat": -34.6037, "lon": -58.3816},
    "--- 海外 (ヨーロッパ) ---": None,
    "ロンドン (イギリス)": {"lat": 51.5074, "lon": -0.1278},
    "パリ (フランス)": {"lat": 48.8566, "lon": 2.3522},
    "ベルリン (ドイツ)": {"lat": 52.5200, "lon": 13.4050},
    "--- 海外 (アフリカ) ---": None,
    "カイロ (エジプト)": {"lat": 30.0444, "lon": 31.2357},
    "プレトリア (南アフリカ)": {"lat": -25.7479, "lon": 28.2293},
    "--- 海外 (オセアニア) ---": None,
    "キャンベラ (オーストラリア)": {"lat": -35.2809, "lon": 149.1300},
    "ウェリントン (ニュージーランド)": {"lat": -41.2865, "lon": 174.7762},
}

GROUND_STATIONS = [
    {"name": "Tsukuba", "lat": 36.06, "lon": 140.12, "color": "red"},
    {"name": "Katsuura", "lat": 35.15, "lon": 140.30, "color": "red"},
    {"name": "Okinawa", "lat": 26.50, "lon": 127.85, "color": "red"},
    {"name": "Svalbard", "lat": 78.22, "lon": 15.40, "color": "blue"},
    {"name": "Santiago", "lat": -33.15, "lon": -70.66, "color": "green"},
    {"name": "Maspalomas", "lat": 27.76, "lon": -15.63, "color": "green"},
    {"name": "Dongara", "lat": -29.25, "lon": 114.93, "color": "orange"},
]

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ==========================================
# 1. クラス・ヘルパー関数
# ==========================================
class SatSim:
    def __init__(self, target_lat, target_lon):
        self.t_lat = target_lat
        self.t_lon = target_lon

    def get_state(self, t_sec):
        # 簡易軌道モデル
        period = 5760 # 約96分
        omega = 2 * np.pi / period
        phase_offset = np.arcsin(np.clip(self.t_lat / 90.0, -1, 1))
        current_phase = phase_offset - omega * t_sec
        sat_lat = 90.0 * np.sin(current_phase)
        lon_drift = -0.06
        sat_lon = (self.t_lon - 2.0) + (lon_drift * t_sec)
        sat_lon = (sat_lon + 180) % 360 - 180

        # ステータス判定
        if abs(t_sec) <= 2: status = "CAPTURING"
        elif t_sec < -20: status = "PREV TASK"
        elif t_sec > 20: status = "NEXT TASK"
        else: status = "TARGET ACQ"

        # 姿勢角
        if -40 <= t_sec <= 40:
            d_lat = self.t_lat - sat_lat
            d_lon = (self.t_lon - sat_lon + 180) % 360 - 180
            # 簡易計算のため極付近の歪みは無視
            pitch = np.degrees(np.arctan2(d_lat, 10.0)) * 2
            roll = np.degrees(np.arctan2(d_lon, 10.0)) * 2
        else:
            pitch, roll = 0, 0

        return {
            "time": t_sec, "lat": sat_lat, "lon": sat_lon,
            "pitch": np.clip(pitch, -45, 45), "roll": np.clip(roll, -45, 45),
            "status": status
        }

def generate_circle_coords(lat, lon, radius_deg=20, n_points=30):
    """地図上の円（通信範囲）の座標を生成"""
    lats, lons = [], []
    for i in range(n_points + 1):
        angle = 2 * np.pi * i / n_points
        d_lat = radius_deg * np.cos(angle)
        d_lon = radius_deg * np.sin(angle) / np.cos(np.radians(lat)) # 緯度補正
        lats.append(np.clip(lat + d_lat, -90, 90))
        lons.append(lon + d_lon)
    return lats, lons

def rotate_point(x, y, z, roll, pitch):
    """3D回転行列"""
    r_rad, p_rad = np.radians(roll), np.radians(pitch)
    # Roll
    y_r = y * np.cos(r_rad) - z * np.sin(r_rad)
    z_r = y * np.sin(r_rad) + z * np.cos(r_rad)
    y, z = y_r, z_r
    # Pitch
    x_p = x * np.cos(p_rad) + z * np.sin(p_rad)
    z_p = -x * np.sin(p_rad) + z * np.cos(p_rad)
    return x_p, y, z_p

def normalize(band):
    valid_pixels = band[band > 0]
    if len(valid_pixels) == 0: return band
    p2, p98 = np.percentile(valid_pixels, (2, 98))
    return np.clip((band - p2) / (p98 - p2), 0, 1)

# ==========================================
# 2. UI レイアウト
# ==========================================
st.title("🛰️ 衛星画像取得プラットフォーム")

# Session State 初期化
if 'search_results' not in st.session_state: st.session_state.search_results = []
if 'search_performed' not in st.session_state: st.session_state.search_performed = False
if 'search_bbox' not in st.session_state: st.session_state.search_bbox = []
if 'selected_item_data' not in st.session_state: st.session_state.selected_item_data = None
if 'sim_time' not in st.session_state: st.session_state.sim_time = -40

# --- サイドバー (設定) ---
st.sidebar.header("検索条件 / Simulation Setup")
with st.sidebar.form(key='search_form'):
    location_mode = st.radio("場所の指定方法", ["リストから選択", "座標を直接入力"])
    selected_lat, selected_lon = 0.0, 0.0

    if location_mode == "リストから選択":
        valid_locations = [k for k, v in LOCATIONS.items() if v is not None]
        location_name = st.selectbox("場所を選択", valid_locations, index=valid_locations.index("筑波宇宙センター"))
        coords = LOCATIONS[location_name]
        selected_lat, selected_lon = coords["lat"], coords["lon"]
        st.info(f"座標: {selected_lat:.4f}, {selected_lon:.4f}")
    else:
        col1, col2 = st.columns(2)
        selected_lat = col1.number_input("緯度", value=36.0652, format="%.4f")
        selected_lon = col2.number_input("経度", value=140.1272, format="%.4f")

    buffer_deg = st.slider("取得範囲 (度)", 0.01, 0.5, 0.1, help="0.1度 ≒ 11km")
    target_date = st.date_input("希望する日付", datetime.date(2023, 1, 1))
    date_range_days = st.number_input("検索幅 (前後日数)", min_value=1, max_value=30, value=5)
    satellite_options = st.multiselect("使用する衛星", ["Sentinel-2", "Landsat 8/9"], default=["Sentinel-2"])
    max_cloud = st.slider("許容する雲量 (%)", 0, 100, 30)
    search_clicked = st.form_submit_button("画像を検索する")

# 検索ロジック
if search_clicked:
    st.session_state.search_results = []
    st.session_state.selected_item_data = None
    start_date = target_date - datetime.timedelta(days=date_range_days)
    end_date = target_date + datetime.timedelta(days=date_range_days)
    date_query = f"{start_date.isoformat()}/{end_date.isoformat()}"
    bbox = [selected_lon - buffer_deg/2, selected_lat - buffer_deg/2, selected_lon + buffer_deg/2, selected_lat + buffer_deg/2]
    collections = []
    if "Sentinel-2" in satellite_options: collections.append("sentinel-2-l2a")
    if "Landsat 8/9" in satellite_options: collections.append("landsat-c2-l2")

    if not collections:
        st.error("衛星を選択してください。")
    else:
        with st.spinner(f"カタログを検索中..."):
            try:
                client = Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
                search = client.search(collections=collections, bbox=bbox, datetime=date_query, query={"eo:cloud_cover": {"lt": max_cloud}}, sortby=[{"field": "properties.datetime", "direction": "desc"}])
                items = list(search.items())
                st.session_state.search_results = items
                st.session_state.search_bbox = bbox
                st.session_state.search_performed = True
            except Exception as e:
                st.error(f"検索エラー: {e}")

# ==========================================
# タブ構成
# ==========================================
tab1, tab2 = st.tabs(["📷 画像取得・解析", "📡 運用シミュレーション"])

# --- タブ1: 画像取得 ---
with tab1:
    if st.session_state.search_performed:
        items = st.session_state.search_results
        if not items:
            st.warning("画像が見つかりませんでした。")
        else:
            st.success(f"{len(items)} 件のデータが見つかりました。")
            item_options = {}
            for item in items:
                dt = datetime.datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
                sat_id = item.properties.get("platform", item.collection_id)
                cloud = item.properties.get("eo:cloud_cover", 0)
                sat_disp = "Sentinel-2" if "sentinel" in item.collection_id else ("Landsat" if "landsat" in item.collection_id else sat_id)
                label = f"[{sat_disp}] {dt.strftime('%Y-%m-%d %H:%M')} (雲: {cloud:.1f}%)"
                item_options[label] = item

            options_list = ["--- 画像を選択してください ---"] + list(item_options.keys())
            selected_label = st.selectbox("表示するデータを選択", options=options_list)

            if selected_label != "--- 画像を選択してください ---":
                selected_item = item_options[selected_label]
                st.session_state.selected_item_data = {"item": selected_item, "label": selected_label} # シミュレーション用にも保存

                col_img, col_info = st.columns([2, 1])
                with col_img:
                    st.markdown("**画像を生成中...**")
                    try:
                        collection_id = selected_item.collection_id
                        area_size = st.session_state.search_bbox[2] - st.session_state.search_bbox[0]
                        base_resolution = 10 if "sentinel-2" in collection_id else 30
                        resolution = base_resolution * 4 if area_size > 0.5 else base_resolution
                        if area_size > 0.5: st.warning(f"⚠️ メモリ保護モード: 解像度調整中 ({base_resolution}m → {resolution}m)")

                        bands = ["B04", "B03", "B02"] if "sentinel-2" in collection_id else ["red", "green", "blue"]
                        with st.spinner("ダウンロード中..."):
                            ds = odc.stac.load([selected_item], bands=bands, bbox=st.session_state.search_bbox, resolution=resolution)

                        if "B04" in bands:
                            r, g, b = ds["B04"].isel(time=0).values.astype(float), ds["B03"].isel(time=0).values.astype(float), ds["B02"].isel(time=0).values.astype(float)
                        else:
                            r, g, b = ds["red"].isel(time=0).values.astype(float), ds["green"].isel(time=0).values.astype(float), ds["blue"].isel(time=0).values.astype(float)

                        rgb = np.dstack((normalize(r), normalize(g), normalize(b)))
                        st.image(rgb, caption=f"合成画像: {selected_label}", clamp=True, use_column_width=True)

                        img_array = (rgb * 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_array)
                        buf = io.BytesIO()
                        img_pil.save(buf, format="PNG")
                        st.download_button("📥 画像をダウンロード (PNG)", buf.getvalue(), f"satellite_image_{target_date}.png", "image/png")
                    except Exception as e:
                        st.error("画像生成エラー")
                        st.caption(e)
                with col_info:
                    st.subheader("メタデータ")
                    st.json(selected_item.properties)

# --- タブ2: 運用シミュレーション ---
with tab2:
    if st.session_state.selected_item_data is None:
        st.info("👈 まずは「画像取得・解析」タブで、ターゲットとなる画像を選択してください。")
    else:
        current_label = st.session_state.selected_item_data["label"]
        st.markdown(f"### 🎯 Target Mission: {current_label}")
        
        # 1. 前後の画像（仮想ストリップ）表示
        st.markdown("#### 🎞️ Sequence Plan (Virtual)")
        col_prev, col_curr, col_next = st.columns(3)
        with col_prev:
            st.markdown(f"**Prev Task (-30s)**\n\n⬜ *No Data (Simulated)*\n\nTarget: Lat {selected_lat+0.5:.2f}")
        with col_curr:
            st.success(f"**Current Task (T=0)**\n\nTarget: Lat {selected_lat:.2f}\n\nSelected Image")
        with col_next:
            st.markdown(f"**Next Task (+30s)**\n\n⬜ *No Data (Simulated)*\n\nTarget: Lat {selected_lat-0.5:.2f}")

        # 2. シミュレーション制御
        st.markdown("#### 🛰️ Attitude & Orbit Simulation")
        
        # コントロールパネル（再生・スライダー）をグラフの上に配置
        col_ctrl1, col_ctrl2 = st.columns([1, 4])
        with col_ctrl1:
            if st.button("▶️ 再生 / 停止"):
                # 再生状態をトグルする簡易実装（Streamlitでのアニメーションは再実行が必要）
                for t in range(-40, 42, 2):
                    st.session_state.sim_time = t
                    time.sleep(0.1)
                    st.rerun()
        
        with col_ctrl2:
            sim_time = st.slider("Time Offset (sec)", -40, 40, st.session_state.sim_time, 2)
            st.session_state.sim_time = sim_time # スライダーの値を同期

        # --- 計算処理 ---
        sim = SatSim(selected_lat, selected_lon)
        state = sim.get_state(sim_time)
        
        # 描画用データ準備
        orbit_lats, orbit_lons = [], []
        # 軌道全体（背景）
        for t in range(-2000, 2001, 60):
            s = sim.get_state(t)
            orbit_lats.append(s["lat"])
            orbit_lons.append(s["lon"])
        
        # 現在のシミュレーション範囲
        active_lats, active_lons = [], []
        for t in range(-40, 41, 2):
            s = sim.get_state(t)
            active_lats.append(s["lat"])
            active_lons.append(s["lon"])

        # --- Plotlyによる可視化 ---
        # 2D Map と 3D Attitude を並べて表示
        
        # 1. 2D Map
        fig_map = go.Figure()

        # 背景軌道（グレー）
        fig_map.add_trace(go.Scattergeo(
            lon=orbit_lons, lat=orbit_lats, mode='lines',
            line=dict(width=1, color='lightgray', dash='dash'), name='Orbit Path'
        ))
        # アクティブ軌道（青）
        fig_map.add_trace(go.Scattergeo(
            lon=active_lons, lat=active_lats, mode='lines',
            line=dict(width=3, color='blue'), name='Active Sim'
        ))
        # ターゲット地点
        fig_map.add_trace(go.Scattergeo(
            lon=[selected_lon], lat=[selected_lat], mode='markers+text',
            marker=dict(size=12, color='red', symbol='star'),
            text=["TARGET"], textposition="top right", name='Target'
        ))
        
        # 地上局と通信範囲
        for gs in GROUND_STATIONS:
            # 範囲円
            c_lats, c_lons = generate_circle_coords(gs["lat"], gs["lon"])
            fig_map.add_trace(go.Scattergeo(
                lon=c_lons, lat=c_lats, mode='lines', fill='toself',
                fillcolor=gs["color"], line=dict(color=gs["color"], width=1),
                opacity=0.2, showlegend=False, hoverinfo='skip'
            ))
            # 局アイコン
            fig_map.add_trace(go.Scattergeo(
                lon=[gs["lon"]], lat=[gs["lat"]], mode='markers+text',
                marker=dict(size=8, color=gs["color"], symbol='triangle-up'),
                text=[gs["name"]], textposition="bottom center",
                name=gs["name"], showlegend=False
            ))

        # 現在の衛星位置
        sat_color = 'gold' if state["status"] == "CAPTURING" else 'blue'
        sat_size = 20 if state["status"] == "CAPTURING" else 12
        fig_map.add_trace(go.Scattergeo(
            lon=[state["lon"]], lat=[state["lat"]], mode='markers',
            marker=dict(size=sat_size, color=sat_color, line=dict(width=2, color='white')),
            name='Satellite'
        ))

        # マップ設定
        zoom_level = 3
        # 太平洋またぎ対策を簡易的に行うため、projection_typeを変更
        fig_map.update_geos(
            projection_type="mercator",
            showcoastlines=True, coastlinecolor="RebeccaPurple",
            showland=True, landcolor="whitesmoke",
            showocean=True, oceancolor="azure",
            showcountries=True, countrycolor="lightgray",
            lataxis_range=[min(active_lats)-30, max(active_lats)+30],
            lonaxis_range=[min(active_lons)-30, max(active_lons)+30],
            fitbounds=False # 手動ズームを有効にするためFalse
        )
        fig_map.update_layout(
            title_text=f"Mission Map (Lat/Lon) - Status: {state['status']}",
            margin={"r":0,"t":30,"l":0,"b":0},
            height=400
        )

        # 2. 3D Attitude
        fig_3d = go.Figure()

        # 地球（簡易的な球面ワイヤーフレーム）
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_e = 4 * np.cos(u)*np.sin(v)
        y_e = 4 * np.sin(u)*np.sin(v)
        z_e = 4 * np.cos(v) - 5.5 # 少し下にずらす
        fig_3d.add_trace(go.Surface(x=x_e, y=y_e, z=z_e, opacity=0.1, showscale=False, colorscale='Blues'))

        # 衛星本体（回転する直方体）
        # 基準キューブ
        cube_x = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * 0.5
        cube_y = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * 0.3
        cube_z = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * 0.3
        
        # 回転適用
        rot_x, rot_y, rot_z = [], [], []
        for i in range(8):
            rx, ry, rz = rotate_point(cube_x[i], cube_y[i], cube_z[i], state["roll"], state["pitch"])
            rot_x.append(rx); rot_y.append(ry); rot_z.append(rz)

        # メッシュ定義
        fig_3d.add_trace(go.Mesh3d(
            x=rot_x, y=rot_y, z=rot_z,
            color='silver', alphahull=0, name='Satellite Body',
            lighting=dict(ambient=0.5, diffuse=0.5)
        ))

        # センサ方向ベクトル（矢印の代わりにConeとLineを使用）
        # センサの向き（Z軸下向きを基準に回転）
        sx, sy, sz = rotate_point(0, 0, -1.5, state["roll"], state["pitch"])
        beam_color = 'gold' if state["status"] == "CAPTURING" else 'red'
        beam_width = 10 if state["status"] == "CAPTURING" else 5
        
        fig_3d.add_trace(go.Scatter3d(
            x=[0, sx], y=[0, sy], z=[0, sz],
            mode='lines', line=dict(color=beam_color, width=beam_width),
            name='Sensor Beam'
        ))

        # 進行方向ベクトル（X軸）
        fig_3d.add_trace(go.Scatter3d(
            x=[0, 1.5], y=[0, 0], z=[0, 0],
            mode='lines', line=dict(color='cyan', width=5),
            name='Velocity Vector'
        ))

        # カウントダウンテキスト
        count_val = state["time"]
        count_text = f"SHOOTING NOW" if abs(count_val) <= 2 else f"T {count_val:+.0f} s"
        text_color = "red" if abs(count_val) <= 2 else "black"
        
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                aspectmode='cube'
            ),
            title=dict(text=f"Attitude (3D) - {count_text}", font=dict(color=text_color, size=20)),
            margin={"r":0,"t":30,"l":0,"b":0},
            height=400
        )

        # 並べて表示
        col_map, col_3d = st.columns([3, 2])
        with col_map:
            st.plotly_chart(fig_map, use_container_width=True)
        with col_3d:
            st.plotly_chart(fig_3d, use_container_width=True)

        # 3. テレメトリグラフ
        st.markdown("#### 📈 Attitude Telemetry")
        # グラフ用データ作成
        times = range(-40, 41, 2)
        rolls, pitches = [], []
        for t in times:
            s = sim.get_state(t)
            rolls.append(s["roll"])
            pitches.append(s["pitch"])
            
        df_telem = pd.DataFrame({"Time": times, "Roll": rolls, "Pitch": pitches})
        
        # グラフ描画（現在時刻に縦線を入れる）
        fig_telem = go.Figure()
        fig_telem.add_trace(go.Scatter(x=df_telem["Time"], y=df_telem["Roll"], name="Roll", line=dict(color="orange")))
        fig_telem.add_trace(go.Scatter(x=df_telem["Time"], y=df_telem["Pitch"], name="Pitch", line=dict(color="green")))
        
        # 現在時刻バー
        fig_telem.add_vline(x=sim_time, line_width=2, line_dash="dash", line_color="red")
        
        fig_telem.update_layout(
            xaxis_title="Time (sec)", yaxis_title="Angle (deg)",
            height=250, margin={"t":10, "b":10}
        )
        st.plotly_chart(fig_telem, use_container_width=True)
