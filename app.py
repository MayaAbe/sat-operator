import streamlit as st
from pystac_client import Client
import planetary_computer
import odc.stac
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import io

# ページ設定
st.set_page_config(page_title="衛星画像取得プラットフォーム", layout="wide")

# ==========================================
# 1. 定数・設定
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

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ==========================================
# 2. ヘルパー関数
# ==========================================
def normalize(band):
    """画素値を0-1の範囲に調整"""
    valid_pixels = band[band > 0]
    if len(valid_pixels) == 0: return band
    p2, p98 = np.percentile(valid_pixels, (2, 98))
    return np.clip((band - p2) / (p98 - p2), 0, 1)

# ==========================================
# 3. UI レイアウト
# ==========================================
# 変更点: タイトル変更
st.title("🛰️ 衛星画像取得プラットフォーム")
st.markdown("指定した場所・日時の衛星データをダウンロードし、可視化・保存できます。")

# Session State 初期化
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False
if 'search_bbox' not in st.session_state:
    st.session_state.search_bbox = []

# --- サイドバー (フォーム化) ---
st.sidebar.header("検索条件")

with st.sidebar.form(key='search_form'):
    location_mode = st.radio("場所の指定方法", ["リストから選択", "座標を直接入力"])

    selected_lat = 0.0
    selected_lon = 0.0

    if location_mode == "リストから選択":
        valid_locations = [k for k, v in LOCATIONS.items() if v is not None]
        location_name = st.selectbox("場所を選択", valid_locations, index=valid_locations.index("筑波宇宙センター"))
        coords = LOCATIONS[location_name]
        selected_lat = coords["lat"]
        selected_lon = coords["lon"]
        st.info(f"座標: 北緯{selected_lat}, 東経{selected_lon}")
    else:
        col1, col2 = st.columns(2)
        selected_lat = col1.number_input("緯度", value=36.0652, format="%.4f")
        selected_lon = col2.number_input("経度", value=140.1272, format="%.4f")

    buffer_deg = st.slider("取得範囲 (度)", 0.01, 0.5, 0.1, help="0.1度 ≒ 11km")
    target_date = st.date_input("希望する日付", datetime.date(2023, 1, 1))
    date_range_days = st.number_input("検索幅 (前後日数)", min_value=1, max_value=30, value=5)

    satellite_options = st.multiselect(
        "使用する衛星データ",
        ["Sentinel-2", "Landsat 8/9"],
        default=["Sentinel-2"]
    )

    max_cloud = st.slider("許容する雲量 (%)", 0, 100, 30)
    
    search_clicked = st.form_submit_button("画像を検索する")


# ==========================================
# 4. 検索ロジック (ボタン押下時のみ実行)
# ==========================================
if search_clicked:
    st.session_state.search_results = []
    
    start_date = target_date - datetime.timedelta(days=date_range_days)
    end_date = target_date + datetime.timedelta(days=date_range_days)
    date_query = f"{start_date.isoformat()}/{end_date.isoformat()}"
    
    bbox = [
        selected_lon - buffer_deg/2, selected_lat - buffer_deg/2,
        selected_lon + buffer_deg/2, selected_lat + buffer_deg/2
    ]

    collections = []
    if "Sentinel-2" in satellite_options:
        collections.append("sentinel-2-l2a")
    if "Landsat 8/9" in satellite_options:
        collections.append("landsat-c2-l2")

    if not collections:
        st.error("衛星を選択してください。")
    else:
        with st.spinner(f"カタログを検索中..."):
            try:
                client = Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
                search = client.search(
                    collections=collections,
                    bbox=bbox,
                    datetime=date_query,
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                    sortby=[{"field": "properties.datetime", "direction": "desc"}]
                )
                items = list(search.items())
                
                st.session_state.search_results = items
                st.session_state.search_bbox = bbox
                st.session_state.search_performed = True
                
            except Exception as e:
                st.error(f"検索エラー: {e}")

# ==========================================
# 5. 結果表示 & 画像生成ロジック
# ==========================================
if st.session_state.search_performed:
    st.header(f"📡 検索結果")
    items = st.session_state.search_results

    if not items:
        st.warning("画像が見つかりませんでした。条件を変更してください。")
    else:
        st.success(f"{len(items)} 件のデータが見つかりました。")

        item_options = {}
        for item in items:
            dt = datetime.datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
            sat_id = item.properties.get("platform", item.collection_id)
            cloud = item.properties.get("eo:cloud_cover", 0)
            
            if "sentinel" in item.collection_id: sat_disp = "Sentinel-2"
            elif "landsat" in item.collection_id: sat_disp = "Landsat"
            else: sat_disp = sat_id

            label = f"[{sat_disp}] {dt.strftime('%Y-%m-%d %H:%M')} (雲: {cloud:.1f}%)"
            item_options[label] = item

        options_list = ["--- 画像を選択してください ---"] + list(item_options.keys())

        selected_label = st.selectbox(
            "表示するデータを選択してください（選択するとダウンロードを開始します）", 
            options=options_list
        )
        
        if selected_label != "--- 画像を選択してください ---":
            selected_item = item_options[selected_label]
            
            col_img, col_info = st.columns([2, 1])
            
            with col_img:
                st.markdown("**画像を生成中...**")
                
                try:
                    collection_id = selected_item.collection_id
                    
                    # --- 変更点: メモリ保護の緩和 ---
                    area_size = st.session_state.search_bbox[2] - st.session_state.search_bbox[0]
                    base_resolution = 10 if "sentinel-2" in collection_id else 30
                    
                    # 閾値を 0.1 -> 0.5 (約55km) に緩和しました
                    if area_size > 0.5:
                        resolution = base_resolution * 4 
                        st.warning(f"⚠️ **メモリ保護モード**: 指定範囲が非常に広いため、解像度を落として表示しています。\n（{base_resolution}m → {resolution}m）")
                    else:
                        resolution = base_resolution
                    # ------------------------------

                    if "sentinel-2" in collection_id:
                        bands = ["B04", "B03", "B02"]
                    elif "landsat" in collection_id:
                        bands = ["red", "green", "blue"]
                    else:
                        bands = ["red", "green", "blue"]

                    load_bbox = st.session_state.search_bbox

                    with st.spinner("クラウドからデータを取得・合成中..."):
                        ds = odc.stac.load(
                            [selected_item],
                            bands=bands,
                            bbox=load_bbox,
                            resolution=resolution
                        )

                    if "B04" in bands:
                        r = ds["B04"].isel(time=0).values.astype(float)
                        g = ds["B03"].isel(time=0).values.astype(float)
                        b = ds["B02"].isel(time=0).values.astype(float)
                    else:
                        r = ds["red"].isel(time=0).values.astype(float)
                        g = ds["green"].isel(time=0).values.astype(float)
                        b = ds["blue"].isel(time=0).values.astype(float)

                    rgb = np.dstack((normalize(r), normalize(g), normalize(b)))
                    
                    # --- 変更点: レスポンシブ対応 (use_column_width=True) ---
                    st.image(rgb, caption=f"合成画像: {selected_label}", clamp=True, use_column_width=True)
                    
                    # --- 変更点: ダウンロードボタンの追加 ---
                    # NumPy配列を画像バイトデータに変換
                    img_array = (rgb * 255).astype(np.uint8) # 0-1 float -> 0-255 int
                    img_pil = Image.fromarray(img_array)
                    buf = io.BytesIO()
                    img_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    st.download_button(
                        label="📥 画像をダウンロード (PNG)",
                        data=byte_im,
                        file_name=f"satellite_image_{target_date}.png",
                        mime="image/png"
                    )
                    st.success("表示完了")

                except Exception as e:
                    st.error("画像生成エラー")
                    st.caption(e)

            with col_info:
                st.subheader("メタデータ")
                props = selected_item.properties
                st.write(f"**衛星**: {props.get('platform', 'Unknown')}")
                st.write(f"**日時**: {props.get('datetime')}")
                st.write(f"**雲量**: {props.get('eo:cloud_cover')}%")
                with st.expander("詳細メタデータ"):
                    st.json(props)
