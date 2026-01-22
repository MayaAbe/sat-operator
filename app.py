import streamlit as st
from pystac_client import Client
import datetime

# ページ設定
st.set_page_config(page_title="衛星画像取得ビューア", layout="wide")

# ==========================================
# 1. 定数・設定（場所リストの定義）
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

# STAC APIのエンドポイント（Sentinel-2用: AWS Earth Search）
STAC_API_URL = "https://earth-search.aws.element84.com/v1"

# ==========================================
# 2. UI レイアウト
# ==========================================
st.title("🛰️ 衛星画像取得シミュレーター")
st.markdown("指定した場所・日時の衛星画像を検索し、プレビューを表示します。")

# サイドバー：検索条件の設定
st.sidebar.header("検索条件")

# (1) 場所の選択
location_mode = st.sidebar.radio("場所の指定方法", ["リストから選択", "座標を直接入力"])

selected_lat = 0.0
selected_lon = 0.0

if location_mode == "リストから選択":
    # セパレーター（Noneの値）を除外したリストを作成
    valid_locations = [k for k, v in LOCATIONS.items() if v is not None]
    location_name = st.sidebar.selectbox("場所を選択", valid_locations, index=valid_locations.index("筑波宇宙センター"))
    coords = LOCATIONS[location_name]
    selected_lat = coords["lat"]
    selected_lon = coords["lon"]
    st.sidebar.info(f"座標: 北緯{selected_lat}, 東経{selected_lon}")

else:
    col1, col2 = st.sidebar.columns(2)
    selected_lat = col1.number_input("緯度", value=36.0652, format="%.4f")
    selected_lon = col2.number_input("経度", value=140.1272, format="%.4f")

# (2) 範囲（バッファサイズ）
buffer_deg = st.sidebar.slider("取得範囲 (度)", 0.01, 0.5, 0.1, help="中心座標からの広さ（約0.1度=約11km）")

# (3) 日付指定
target_date = st.sidebar.date_input("希望する日付", datetime.date(2023, 1, 1))
date_range_days = st.sidebar.number_input("検索幅 (前後日数)", min_value=1, max_value=30, value=5)

# (4) 衛星とクラウドカバー
# 複数衛星への対応（ここではSentinel-2をメインにしつつ、Landsatも選択肢に入れる教育的配慮）
satellite_options = st.sidebar.multiselect(
    "使用する衛星データ",
    ["Sentinel-2", "Landsat 8/9"],
    default=["Sentinel-2"]
)

max_cloud = st.sidebar.slider("許容する雲量 (%)", 0, 100, 20)

search_clicked = st.sidebar.button("画像を検索する")

# ==========================================
# 3. 検索ロジックと結果表示
# ==========================================
if search_clicked:
    st.header(f"📡 検索結果")
    
    # 検索期間の計算
    start_date = target_date - datetime.timedelta(days=date_range_days)
    end_date = target_date + datetime.timedelta(days=date_range_days)
    date_query = f"{start_date.isoformat()}/{end_date.isoformat()}"
    
    # Bounding Boxの計算
    bbox = [
        selected_lon - buffer_deg, selected_lat - buffer_deg,
        selected_lon + buffer_deg, selected_lat + buffer_deg
    ]

    # コレクションIDのマッピング
    collections = []
    if "Sentinel-2" in satellite_options:
        collections.append("sentinel-2-l2a")
    if "Landsat 8/9" in satellite_options:
        collections.append("landsat-c2-l2") # 注意: Earth Search APIではLandsatが含まれない場合がある

    if not collections:
        st.error("衛星を選択してください。")
        st.stop()

    with st.spinner(f"{start_date} から {end_date} の期間でデータを検索中..."):
        try:
            client = Client.open(STAC_API_URL)
            search = client.search(
                collections=collections,
                bbox=bbox,
                datetime=date_query,
                query={"eo:cloud_cover": {"lt": max_cloud}},
                sortby=[{"field": "properties.datetime", "direction": "desc"}] # "sortby" に修正
            )
            items = list(search.items())
        except Exception as e:
            st.error(f"検索エラー: {e}")
            st.stop()

    if not items:
        st.warning("条件に合う画像が見つかりませんでした。雲量の条件を緩めるか、日付を変更してください。")
    else:
        st.success(f"{len(items)} 件の画像が見つかりました。")

        # プルダウン用のリスト作成
        # 表示形式: [衛星名] 日時 (雲量: XX%)
        item_options = {}
        for item in items:
            dt = datetime.datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
            sat_id = item.properties.get("platform", item.collection_id)
            cloud = item.properties.get("eo:cloud_cover", 0)
            
            label = f"[{sat_id}] {dt.strftime('%Y-%m-%d %H:%M')} (雲量: {cloud:.1f}%)"
            item_options[label] = item

        # 結果選択プルダウン
        selected_label = st.selectbox("表示する画像を選択 (撮影日時・時刻)", options=list(item_options.keys()))
        
        # 選択されたアイテムの表示
        if selected_label:
            selected_item = item_options[selected_label]
            
            col_img, col_info = st.columns([2, 1])
            
            with col_img:
                # サムネイル画像の取得
                # Sentinel-2 (Earth Search) は 'thumbnail' アセットを持っていることが多い
                if "thumbnail" in selected_item.assets:
                    st.image(selected_item.assets["thumbnail"].href, caption="サムネイル画像", use_column_width=True)
                elif "visual" in selected_item.assets: # Landsat等の場合
                    st.image(selected_item.assets["visual"].href, caption="Visual画像", use_column_width=True)
                else:
                    st.warning("表示可能なサムネイル画像が見つかりませんでした。メタデータのみ表示します。")
                    # ヒント: 実際の解析ではここでCOG(Cloud Optimized GeoTIFF)を読み込む処理が入ります

            with col_info:
                st.subheader("メタデータ情報")
                props = selected_item.properties
                st.write(f"**衛星/プラットフォーム**: {props.get('platform', 'Unknown')}")
                st.write(f"**撮影日時**: {props.get('datetime')}")
                st.write(f"**雲量**: {props.get('eo:cloud_cover')}%")
                st.write(f"**太陽高度**: {props.get('view:sun_elevation', 'N/A')}")
                st.write(f"**データID**: {selected_item.id}")
                
                with st.expander("全メタデータを見る"):
                    st.json(props)

            st.markdown("---")
            st.markdown("##### 💡 教育用メモ")
            st.info("""
            このアプリケーションでは、カタログ検索（STAC）を行い、サムネイル（プレビュー）を表示しています。
            実際の解析（NDVIの計算など）を行う場合は、この検索結果で得られたURL（href）を使って、
            特定の波長バンド（赤、近赤外など）のデータをダウンロード・計算するプロセスが必要になります。
            """)
