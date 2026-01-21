"""
Streamlit版：衛星画像検索・運用訓練ツール
- ① STAC API（Microsoft Planetary Computer）から衛星画像を検索し、RGB合成して表示
- ② 観測（ターゲット獲得〜撮影）を簡易に可視化する運用シミュレーション（アニメーション）
- ③ バス部テレメトリを模擬生成し、監視・可視化・CSV出力

Run:
  streamlit run app.py

Tips (Streamlit Community Cloud):
  - この app.py と requirements.txt をリポジトリに置いてデプロイ
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe

import streamlit as st

import pystac
import pystac_client
import planetary_computer
import odc.stac


# -----------------------------------------------------------------------------
# Constants (from the notebook)
# -----------------------------------------------------------------------------
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTIONS = ["sentinel-2-l2a", "landsat-c2-l2"]

LOCATIONS: Dict[str, Optional[List[float]]] = {
    "--- 国内 ---": None,
    "筑波宇宙センター (JAXA)": [36.0621, 140.1265],
    "種子島宇宙センター (JAXA)": [30.4017, 130.9680],
    "東京駅": [35.6812, 139.7671],
    "いろは坂 (栃木)": [36.7376, 139.5161],
    "富士山": [35.3606, 138.7274],
    "桜島": [31.5814, 130.6573],
    "能登半島 (石川)": [37.3941, 136.9034],
    "ナイタイ高原牧場 (北海道)": [43.1670, 143.1590],
    "--- 海外 (アジア・オセアニア) ---": None,
    "東京 (日本)": [35.6895, 139.6917],
    "キャンベラ (オーストラリア)": [-35.2809, 149.1300],
    "ニューデリー (インド)": [28.6139, 77.2090],
    "--- 海外 (北米・南米) ---": None,
    "ワシントンD.C. (アメリカ)": [38.9072, -77.0369],
    "オタワ (カナダ)": [45.4215, -75.6972],
    "ブラジリア (ブラジル)": [-15.7975, -47.8919],
    "--- 海外 (欧州・アフリカ) ---": None,
    "ロンドン (イギリス)": [51.5074, -0.1278],
    "パリ (フランス)": [48.8566, 2.3522],
    "ケープタウン (南アフリカ)": [-33.9249, 18.4241],
    "カイロ (エジプト)": [30.0444, 31.2357],
}

GROUND_STATIONS = [
    {"name": "Tsukuba", "lat": 36.06, "lon": 140.12, "color": "tab:red"},
    {"name": "Katsuura", "lat": 35.15, "lon": 140.30, "color": "tab:red"},
    {"name": "Okinawa", "lat": 26.50, "lon": 127.85, "color": "tab:red"},
    {"name": "Svalbard", "lat": 78.22, "lon": 15.40, "color": "tab:blue"},
    {"name": "Santiago", "lat": -33.15, "lon": -70.66, "color": "tab:green"},
    {"name": "Maspalomas", "lat": 27.76, "lon": -15.63, "color": "tab:green"},
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def normalize_band(band: np.ndarray) -> np.ndarray:
    """2%〜98%で正規化（notebookと同趣旨）"""
    valid = band[band > 0]
    if valid.size == 0:
        return band
    p2, p98 = np.percentile(valid, (2, 98))
    if p98 == p2:
        return np.clip(band, 0, 1)
    return np.clip((band - p2) / (p98 - p2), 0, 1)


def bbox_from_center(lat: float, lon: float, fov_deg: float) -> List[float]:
    delta = fov_deg / 2.0
    return [lon - delta, lat - delta, lon + delta, lat + delta]


@dataclass(frozen=True)
class Scene:
    label: str
    item_dict: Dict[str, Any]
    datetime_utc: str
    cloud_cover: Optional[float]
    collection_id: str


# -----------------------------------------------------------------------------
# STAC search / load
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def search_scenes(
    lat: float,
    lon: float,
    target_date: dt.date,
    fov_deg: float,
    cloud_lt: float,
) -> List[Scene]:
    """STAC検索を行い、候補シーンを返す（itemはdictで保持）。"""
    bbox = bbox_from_center(lat, lon, fov_deg)

    # 指定日の前後3日
    t = pd.Timestamp(target_date)
    start_date = (t - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (t + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    date_range = f"{start_date}/{end_date}"

    catalog = pystac_client.Client.open(
        STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=COLLECTIONS,
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": float(cloud_lt)}},
    )
    items = list(search.get_items())

    scenes: List[Scene] = []
    for item in sorted(items, key=lambda x: x.datetime or dt.datetime.min.replace(tzinfo=dt.timezone.utc)):
        dt_str = item.datetime.isoformat() if item.datetime else "Unknown"
        cloud = item.properties.get("eo:cloud_cover", None)
        platform = item.properties.get("platform", "Unknown")
        collection_id = item.collection_id or "Unknown"

        label = f"{dt_str} | {platform} | {collection_id} | cloud={cloud if cloud is not None else 'NA'}"
        scenes.append(
            Scene(
                label=label,
                item_dict=item.to_dict(),
                datetime_utc=dt_str,
                cloud_cover=cloud,
                collection_id=collection_id,
            )
        )

    return scenes


def load_rgb_from_scene(scene: Scene, bbox: List[float]) -> np.ndarray:
    """シーンをRGBにして返す（0-1）。"""
    item = pystac.Item.from_dict(scene.item_dict)
    item = planetary_computer.sign(item)

    if "sentinel-2" in scene.collection_id:
        bands = ["B04", "B03", "B02"]  # RGB
        res = 10
    elif "landsat" in scene.collection_id:
        # Planetary Computer の landsat-c2-l2 は色名 band が使える
        bands = ["red", "green", "blue"]
        res = 30
    else:
        # フォールバック：とにかくRGBっぽいものを探す
        bands = ["red", "green", "blue"]
        res = 30

    ds = odc.stac.load(
        [item],
        bands=bands,
        bbox=bbox,
        resolution=res,
    )

    r = ds[bands[0]].isel(time=0).values.astype(float)
    g = ds[bands[1]].isel(time=0).values.astype(float)
    b = ds[bands[2]].isel(time=0).values.astype(float)

    rgb = np.dstack((normalize_band(r), normalize_band(g), normalize_band(b)))
    return rgb


# -----------------------------------------------------------------------------
# Operation simulation (from notebook, adapted)
# -----------------------------------------------------------------------------
class SatSim:
    def __init__(self, target_lat: float, target_lon: float):
        self.t_lat = float(target_lat)
        self.t_lon = float(target_lon)

    def get_state(self, t_sec: float) -> Dict[str, float]:
        period = 5760  # 約96分
        omega = 2 * np.pi / period

        phase_offset = np.arcsin(np.clip(self.t_lat / 90.0, -1, 1))
        current_phase = phase_offset - omega * t_sec

        sat_lat = 90.0 * np.sin(current_phase)
        lon_drift = -0.06
        sat_lon = (self.t_lon - 2.0) + (lon_drift * t_sec)
        sat_lon = (sat_lon + 180) % 360 - 180

        if abs(t_sec) <= 2:
            status = "CAPTURING!"
        elif t_sec < -20:
            status = "PREV TASK"
        elif t_sec > 20:
            status = "NEXT TASK"
        else:
            status = "TARGET ACQ"

        if -40 <= t_sec <= 40:
            d_lat = self.t_lat - sat_lat
            d_lon = (self.t_lon - sat_lon + 180) % 360 - 180
            pitch = np.degrees(np.arctan2(d_lat, 10.0)) * 2
            roll = np.degrees(np.arctan2(d_lon, 10.0)) * 2
        else:
            pitch, roll = 0.0, 0.0

        return {
            "time": float(t_sec),
            "lat": float(sat_lat),
            "lon": float(sat_lon),
            "pitch": float(np.clip(pitch, -45, 45)),
            "roll": float(np.clip(roll, -45, 45)),
            "status": status,
        }

    def predict_next_downlink(self, current_time: float) -> Tuple[Optional[float], Optional[str]]:
        for dt_sec in range(0, 6000, 60):
            t_check = current_time + dt_sec
            state = self.get_state(t_check)
            for gs in GROUND_STATIONS:
                dist = np.sqrt((state["lat"] - gs["lat"]) ** 2 + (state["lon"] - gs["lon"]) ** 2)
                if dist < 20.0:
                    return float(t_check), str(gs["name"])
        return None, None


def get_cube(center=(0, 0, 0), size=(1, 1, 1), roll=0.0, pitch=0.0):
    cx, cy, cz = center
    sx, sy, sz = np.array(size) / 2
    vertices = np.array([
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
    ])

    r = np.deg2rad(roll)
    p = np.deg2rad(pitch)
    R_roll = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    R_pitch = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    R = R_pitch @ R_roll

    rotated = (vertices - np.array(center)) @ R.T + np.array(center)
    faces = [
        [rotated[j] for j in [0, 1, 2, 3]],
        [rotated[j] for j in [4, 5, 6, 7]],
        [rotated[j] for j in [0, 1, 5, 4]],
        [rotated[j] for j in [2, 3, 7, 6]],
        [rotated[j] for j in [1, 2, 6, 5]],
        [rotated[j] for j in [4, 7, 3, 0]],
    ]
    return faces


def handle_dateline(lons: List[float], lats: List[float]):
    new_lons, new_lats = [], []
    for i in range(len(lons)):
        if i > 0 and abs(lons[i] - lons[i - 1]) > 300:
            new_lons.append(np.nan)
            new_lats.append(np.nan)
        new_lons.append(lons[i])
        new_lats.append(lats[i])
    return new_lons, new_lats


def build_operation_animation(target_lat: float, target_lon: float) -> Tuple[str, Dict[str, Any]]:
    """JSHTMLアニメーションと、簡易レポートを返す。"""
    # geopandas は環境によっては導入が重いので optional 扱い
    try:
        import geopandas as gpd  # type: ignore
        has_gpd = True
    except Exception:
        gpd = None
        has_gpd = False

    sim = SatSim(target_lat, target_lon)

    times = np.arange(-40, 41, 2)
    data = [sim.get_state(t) for t in times]

    bg_times = np.arange(-3000, 3001, 60)
    bg_data = [sim.get_state(t) for t in bg_times]
    bg_lons, bg_lats = handle_dateline([d["lon"] for d in bg_data], [d["lat"] for d in bg_data])

    lons = [d["lon"] for d in data]
    lats = [d["lat"] for d in data]

    fig = plt.figure(figsize=(16, 8))
    ax_map = fig.add_subplot(1, 2, 1)

    if has_gpd:
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            world.plot(ax=ax_map, color="whitesmoke", edgecolor="lightgray")
        except Exception:
            ax_map.set_facecolor("whitesmoke")
    else:
        ax_map.set_facecolor("whitesmoke")

    ax_map.plot(bg_lons, bg_lats, "-", color="lightgray", linewidth=2, zorder=1)
    ax_map.plot(lons, lats, "-", color="tab:blue", linewidth=3, zorder=2, label="Orbit")

    # 地上局
    visible_stations: List[str] = []
    lons_for_zoom = lons + [target_lon]
    lats_for_zoom = lats + [target_lat]

    for gs in GROUND_STATIONS:
        d_min = np.min(np.sqrt((np.array(lats) - gs["lat"]) ** 2 + (np.array(lons) - gs["lon"]) ** 2))
        is_visible_pass = bool(d_min < 20)

        ax_map.plot(gs["lon"], gs["lat"], "^", color=gs["color"], markersize=8, zorder=3)
        txt = ax_map.text(gs["lon"] + 2, gs["lat"], gs["name"], fontsize=9, fontweight="bold", color=gs["color"], zorder=4)
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

        alpha_fill = 0.2 if is_visible_pass else 0.05
        circle = plt.Circle(
            (gs["lon"], gs["lat"]),
            20,
            facecolor=gs["color"],
            alpha=alpha_fill,
            edgecolor=gs["color"],
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        ax_map.add_patch(circle)

        if is_visible_pass:
            visible_stations.append(gs["name"])
            lons_for_zoom.append(gs["lon"])
            lats_for_zoom.append(gs["lat"])

    # ターゲット
    ax_map.plot(target_lon, target_lat, "*", markersize=18, color="gold", zorder=5, label="Target")
    ttxt = ax_map.text(target_lon + 2, target_lat + 2, "TARGET", fontsize=11, fontweight="bold", color="gold", zorder=6)
    ttxt.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])

    # ズーム
    lon_min, lon_max = min(lons_for_zoom) - 20, max(lons_for_zoom) + 20
    lat_min, lat_max = min(lats_for_zoom) - 20, max(lats_for_zoom) + 20
    ax_map.set_xlim(lon_min, lon_max)
    ax_map.set_ylim(lat_min, lat_max)

    ax_map.set_title("1. Orbit & Ground Stations (2D Map)")
    ax_map.set_xlabel("Longitude [deg]")
    ax_map.set_ylabel("Latitude [deg]")
    ax_map.grid(True, linestyle=":", alpha=0.6)

    # 3D view
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa

    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_3d.set_title("2. Attitude Control (3D View)")
    ax_3d.set_xlim(-2, 2); ax_3d.set_ylim(-2, 2); ax_3d.set_zlim(-2, 2)
    ax_3d.axis("off")

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x_e = 1.6 * np.outer(np.cos(u), np.sin(v))
    y_e = 1.6 * np.outer(np.sin(u), np.sin(v))
    z_e = 1.6 * np.outer(np.ones_like(u), np.cos(v))
    ax_3d.plot_wireframe(x_e, y_e, z_e, alpha=0.2)

    sat_point, = ax_map.plot([], [], "o", color="tab:blue", markersize=10, zorder=6)

    def update(frame: int):
        d = data[frame]
        is_shooting = abs(d["time"]) <= 2
        highlight_color = "gold" if is_shooting else "tab:blue"

        sat_point.set_data([d["lon"]], [d["lat"]])
        sat_point.set_color(highlight_color)
        sat_point.set_markersize(15 if is_shooting else 10)

        ax_3d.clear()
        ax_3d.set_title("2. Attitude Control (3D View)")
        ax_3d.set_xlim(-2, 2); ax_3d.set_ylim(-2, 2); ax_3d.set_zlim(-2, 2)
        ax_3d.axis("off")
        ax_3d.plot_wireframe(x_e, y_e, z_e, alpha=0.2)

        faces = get_cube((0, 0, 0), (0.8, 0.5, 0.5), d["roll"], d["pitch"])
        poly = Poly3DCollection(faces, alpha=0.9, facecolors="silver", edgecolors="k")
        ax_3d.add_collection3d(poly)

        sensor_color = "gold" if is_shooting else "tab:red"
        ax_3d.plot([0, 0], [0, 0], [0, -1.2], color=sensor_color, linewidth=4)

        status_text = f"T={d['time']:+.0f}s  |  {d['status']}  |  roll={d['roll']:+.1f}°, pitch={d['pitch']:+.1f}°"
        bg_color = "gold" if is_shooting else "white"
        ax_3d.text2D(0.05, 0.95, status_text, transform=ax_3d.transAxes, fontsize=12, bbox=dict(facecolor=bg_color, alpha=0.8))

    ani = animation.FuncAnimation(fig, update, frames=len(data), interval=200)
    plt.close(fig)

    html = ani.to_jshtml()

    # report
    report: Dict[str, Any] = {"visible_stations": visible_stations}
    if not visible_stations:
        next_t, next_gs = sim.predict_next_downlink(40)
        report["next_downlink_time_s"] = next_t
        report["next_downlink_gs"] = next_gs

    return html, report


# -----------------------------------------------------------------------------
# Telemetry simulator (from notebook, adapted)
# -----------------------------------------------------------------------------
class BusSimulator:
    def __init__(self, duration_mode: str):
        self.mode = duration_mode  # 'Event (80s)' or 'Long-term (1day)'

        if self.mode == "Event (80s)":
            self.t_start = -40
            self.t_end = 40
            self.step = 1
        else:
            self.t_start = 0
            self.t_end = 86400
            self.step = 600

        self.times = np.arange(self.t_start, self.t_end + 1, self.step)

    def generate_data(self) -> pd.DataFrame:
        np.random.seed(42)

        data: Dict[str, List[float]] = {
            "Time": [],
            "Lat": [], "Lon": [], "Yaw": [],
            "Gen_Power": [], "Cons_Power": [],
            "Battery": [],
            "Roll": [], "Pitch": [],
            "Temp_In": [], "Temp_Ex": [],
            "RW_Speed": [],
            "Mem_Usage": [],
        }

        batt_level = 80.0
        temp_in = 25.0
        temp_ex = 30.0
        mem_usage = 10.0

        for t in self.times:
            is_eclipse = (t % 5400) > 3600 if self.mode != "Event (80s)" else (t < -10 or t > 10)

            # 1) Power
            if is_eclipse:
                gen = 50 + np.random.normal(0, 3)
            else:
                gen = 180 + np.random.normal(0, 8)

            cons_base = 120 if self.mode != "Event (80s)" else 140
            cons = cons_base + np.random.normal(0, 5)

            # 2) Battery
            batt_level += (gen - cons) * (0.002 if self.mode == "Event (80s)" else 0.0002)
            batt_level = float(np.clip(batt_level, 0, 100))

            # 3) Attitude / RW
            if self.mode == "Event (80s)":
                if -40 <= t <= 40:
                    roll = np.sin(t / 10) * 20
                    pitch = np.cos(t / 10) * 10
                    rw_rpm = 1000 + abs(roll) * 100
                else:
                    roll, pitch = 0.0, 0.0
                    rw_rpm = 1000 + np.random.normal(0, 10)
            else:
                roll, pitch = 0.0, 0.0
                rw_rpm = 1000 + np.random.normal(0, 50)

            # 4) Thermal
            target_temp_ex = -20 if is_eclipse else 80
            dt_factor = 0.1 if self.mode == "Event (80s)" else 0.5
            temp_ex += (target_temp_ex - temp_ex) * dt_factor * 0.1
            temp_in += (temp_ex - temp_in) * dt_factor * 0.05

            # 5) Memory usage
            if self.mode == "Event (80s)":
                if -5 <= t <= 5:
                    mem_usage += 2.0
                elif 10 <= t <= 20:
                    mem_usage -= 10.0
            else:
                # 1day: 観測で増え、ダウンリンクで減る（粗いモデル）
                if (t % 5400) < 1800:
                    mem_usage += 0.8
                if 3600 <= (t % 5400) <= 4200:
                    mem_usage -= 3.0
            mem_usage = float(np.clip(mem_usage, 0, 100))

            data["Time"].append(float(t))
            data["Lat"].append(0.0); data["Lon"].append(0.0); data["Yaw"].append(0.0)
            data["Gen_Power"].append(float(gen)); data["Cons_Power"].append(float(cons))
            data["Battery"].append(float(batt_level))
            data["Roll"].append(float(roll)); data["Pitch"].append(float(pitch))
            data["Temp_In"].append(float(temp_in)); data["Temp_Ex"].append(float(temp_ex))
            data["RW_Speed"].append(float(rw_rpm))
            data["Mem_Usage"].append(float(mem_usage))

        return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="衛星画像検索・運用訓練ツール (Streamlit)", layout="wide")

st.title("衛星画像検索・運用訓練ツール (Streamlit)")

page = st.sidebar.radio(
    "メニュー",
    ["1) 衛星画像検索", "2) 運用シミュレーション", "3) テレメトリ監視"],
)

# Common location choices (exclude separators)
location_names = [k for k, v in LOCATIONS.items() if v is not None]


if page == "1) 衛星画像検索":
    st.subheader("1) 衛星画像検索（STAC / Planetary Computer）")

    colA, colB, colC = st.columns([2, 1, 1])

    with colA:
        loc_name = st.selectbox("場所を選択", location_names, index=0)
        lat, lon = LOCATIONS[loc_name]  # type: ignore[misc]

    with colB:
        dflt = dt.date(2024, 1, 1)
        target_date = st.date_input("日付（前後3日で検索）", value=dflt)
        fov_deg = st.slider("表示範囲（度）", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

    with colC:
        cloud_lt = st.slider("雲量しきい値（%未満）", min_value=0, max_value=100, value=50, step=5)

    bbox = bbox_from_center(lat, lon, fov_deg)

    if st.button("衛星画像を検索", type="primary"):
        with st.spinner("STAC検索中..."):
            scenes = search_scenes(lat, lon, target_date, fov_deg, cloud_lt)
        st.session_state["scenes"] = scenes

    scenes = st.session_state.get("scenes", [])
    if not scenes:
        st.info("上のボタンで検索すると、ここに候補が表示されます。")
    else:
        st.success(f"候補シーン数: {len(scenes)}")
        labels = [s.label for s in scenes]
        sel = st.selectbox("表示する画像を選択", labels)

        selected_scene = next(s for s in scenes if s.label == sel)

        if st.button("画像を表示", type="secondary"):
            try:
                with st.spinner("画像をダウンロードしてRGB合成しています..."):
                    rgb = load_rgb_from_scene(selected_scene, bbox)
                fig = plt.figure(figsize=(8, 8))
                plt.imshow(rgb)
                plt.title(f"{loc_name} ({lat:.4f}, {lon:.4f})\n{selected_scene.datetime_utc}")
                plt.axis("off")
                st.pyplot(fig, clear_figure=True)

                meta_col1, meta_col2, meta_col3 = st.columns(3)
                meta_col1.metric("Collection", selected_scene.collection_id)
                meta_col2.metric("Cloud cover", f"{selected_scene.cloud_cover:.1f}%" if selected_scene.cloud_cover is not None else "NA")
                meta_col3.metric("BBox", f"{bbox[0]:.3f},{bbox[1]:.3f} .. {bbox[2]:.3f},{bbox[3]:.3f}")

                st.caption("黒い部分はデータがない場所、またはタイルの端である可能性があります。")

            except Exception as e:
                st.error(f"画像表示エラー: {e}")


elif page == "2) 運用シミュレーション":
    st.subheader("2) 運用シミュレーション（ターゲット獲得〜撮影・通信可視性）")

    colA, colB = st.columns([2, 1])

    with colA:
        loc_name = st.selectbox("ターゲット地点", location_names, index=0, key="sim_loc")
        t_lat, t_lon = LOCATIONS[loc_name]  # type: ignore[misc]

        st.write("簡易モデルで、衛星の通過（T=-40s〜+40s）と姿勢制御をアニメーション表示します。")

    with colB:
        st.write("地上局（通信範囲）")
        st.dataframe(pd.DataFrame(GROUND_STATIONS)[["name", "lat", "lon"]], use_container_width=True, hide_index=True)

    if st.button("運用シミュレーションを実行", type="primary"):
        with st.spinner("アニメーション生成中（数十秒かかる場合があります）..."):
            html, report = build_operation_animation(t_lat, t_lon)

        st.components.v1.html(html, height=650, scrolling=False)

        st.markdown("### 運用解析レポート")
        if report.get("visible_stations"):
            st.success(f"通信可能局: {', '.join(report['visible_stations'])}")
        else:
            next_t = report.get("next_downlink_time_s")
            next_gs = report.get("next_downlink_gs")
            st.warning("このパスでは通信可能局が見つかりませんでした（簡易判定）。")
            if next_t is not None and next_gs is not None:
                wait_min = int(max(0, (next_t - 40)) // 60)
                st.info(f"次回通信予報: 約 {wait_min} 分後（{next_gs}）")


else:
    st.subheader("3) バス部運用：テレメトリ監視モニター（模擬）")

    colA, colB = st.columns([1, 2])

    with colA:
        mode = st.radio("期間モード", ["Event (80s)", "Long-term (1day)"], index=0)

        options = {
            "Power": "電力 (Gen/Cons)",
            "Battery": "バッテリー残量",
            "Attitude": "姿勢角 (Roll/Pitch)",
            "RW_Speed": "ホイール回転数 (RW)",
            "Temperature": "温度 (In/Ex)",
            "Memory": "データレコーダ (Mem)",
        }
        selected = st.multiselect(
            "表示データ選択",
            list(options.keys()),
            default=["Power", "Attitude", "RW_Speed"],
            format_func=lambda k: options[k],
        )

        run = st.button("テレメトリ生成・表示", type="primary")

    if run:
        sim = BusSimulator(mode)
        df = sim.generate_data()
        st.session_state["telemetry_df"] = df
        st.session_state["telemetry_selected"] = selected

    df: Optional[pd.DataFrame] = st.session_state.get("telemetry_df")
    selected = st.session_state.get("telemetry_selected", selected)

    with colB:
        if df is None:
            st.info("左側で項目を選び「テレメトリ生成・表示」を押すと、ここに可視化が出ます。")
        else:
            st.write(f"サンプル数: {len(df)}（mode={mode}）")
            df_idx = df.set_index("Time")

            if not selected:
                st.warning("少なくとも1つの項目を選択してください。")
            else:
                if "Power" in selected:
                    st.markdown("#### 電力")
                    st.line_chart(df_idx[["Gen_Power", "Cons_Power"]], height=220, use_container_width=True)

                if "Battery" in selected:
                    st.markdown("#### バッテリー残量")
                    st.line_chart(df_idx[["Battery"]], height=220, use_container_width=True)

                if "Attitude" in selected:
                    st.markdown("#### 姿勢角")
                    st.line_chart(df_idx[["Roll", "Pitch"]], height=220, use_container_width=True)

                if "RW_Speed" in selected:
                    st.markdown("#### ホイール回転数")
                    st.line_chart(df_idx[["RW_Speed"]], height=220, use_container_width=True)

                if "Temperature" in selected:
                    st.markdown("#### 温度")
                    st.line_chart(df_idx[["Temp_In", "Temp_Ex"]], height=220, use_container_width=True)

                if "Memory" in selected:
                    st.markdown("#### メモリ使用率")
                    st.line_chart(df_idx[["Mem_Usage"]], height=220, use_container_width=True)

            # 簡易アラーム例
            alarms = []
            if (df["Battery"] < 20).any():
                alarms.append(("Battery", "LOW", int((df["Battery"] < 20).sum())))
            if (df["Temp_Ex"] > 75).any():
                alarms.append(("Temp_Ex", "HIGH", int((df["Temp_Ex"] > 75).sum())))
            if (df["Mem_Usage"] > 85).any():
                alarms.append(("Mem_Usage", "HIGH", int((df["Mem_Usage"] > 85).sum())))

            st.markdown("### アラームサマリ（例）")
            if alarms:
                st.dataframe(pd.DataFrame(alarms, columns=["Signal", "Type", "Count"]), use_container_width=True, hide_index=True)
            else:
                st.success("（この簡易モデルでは）重大な閾値違反は検出されませんでした。")

            st.download_button(
                label="CSVダウンロード",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="telemetry.csv",
                mime="text/csv",
            )
