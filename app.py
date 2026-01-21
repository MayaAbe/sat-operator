import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# --- ページ設定 ---
st.set_page_config(page_title="衛星運用データタスキング (Notebook Port)", layout="wide")

st.title("🛰️ 衛星データタスキング & 運用シミュレーション")
st.markdown("Jupyter Notebookの実装を基にしたWebアプリケーションです。")

# --- サイドバー：パラメータ設定 (Widgetの再現) ---
st.sidebar.header("1. タスキング設定")

# ノートブックのDropdownModelに含まれていた選択肢リスト
location_options = [
    "筑波宇宙センター (JAXA)",
    "種子島宇宙センター (JAXA)",
    "東京駅",
    "いろは坂 (栃木)",
    "富士山",
    "桜島",
    "能登半島 (石川)",
    "ナイタイ高原牧場 (北海道)",
    "東京 (日本)",
    "キャンベラ (オーストラリア)",
    "ニューデリー (インド)",
    "ワシントンD.C. (アメリカ)",
    "オタワ (カナダ)",
    "ブラジリア (ブラジル)",
    "ロンドン (イギリス)",
    "パリ (フランス)",
    "ケープタウン (南アフリカ)",
    "カイロ (エジプト)"
]

# 1. 場所を選択 (Dropdown)
selected_location = st.sidebar.selectbox("1. 場所を選択:", location_options, index=0)

# 座標データの定義 (ノートブックのロジックを補完)
# ※ノートブック内の辞書データが見えないため、主要地点の座標を定義して動作するようにしています
location_coords = {
    "筑波宇宙センター (JAXA)": [36.065, 140.128],
    "種子島宇宙センター (JAXA)": [30.399, 130.968],
    "東京駅": [35.681, 139.767],
    "富士山": [35.360, 138.727],
    # (他の地点が選択された場合は東京駅の座標をデフォルトとして使用)
}
current_coords = location_coords.get(selected_location, [35.681, 139.767])

# 2. 日付を指定 (DatePicker)
# ノートブックの初期値: 2024-01-01
default_date = datetime.date(2024, 1, 1)
selected_date = st.sidebar.date_input("2. 日付を指定:", default_date)

# 表示範囲 (FloatSlider)
# ノートブック設定: min=0.01, max=0.5, step=0.01, value=0.1
fov = st.sidebar.slider("表示範囲 (度):", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

st.sidebar.markdown("---")

# --- メインエリア：機能実装 ---

# 1. 地図表示 (タスキング確認)
st.subheader(f"📍 ターゲット確認: {selected_location}")
st.text(f"座標: 北緯 {current_coords[0]}°, 東経 {current_coords[1]}° (FOV: {fov}°)")

# シンプルな地図表示
map_df = pd.DataFrame({'lat': [current_coords[0]], 'lon': [current_coords[1]]})
st.map(map_df, zoom=10)

# 2. 衛星画像検索機能
st.subheader("📷 衛星画像データの取得")

# 検索ボタン (Button)
if st.button("衛星画像を検索", type="primary"):
    st.session_state['search_executed'] = True
    st.success("検索が完了しました。画像を選択してください。")

# 画像選択 (Dropdown) - 検索後に表示
# ノートブックのDropdownModelにあった選択肢
image_options = [
    "[2024-01-02 01:30] Sentinel-2 (欧州) - 雲: 38.5%",
    "[2023-12-25 10:15] Sentinel-2 (欧州) - 雲: 12.0%",
    "[2023-12-10 09:45] Sentinel-2 (欧州) - 雲: 5.2%"
]

selected_image = st.selectbox(
    "3. 画像を選択:", 
    image_options, 
    disabled=not st.session_state.get('search_executed', False)
)

# 画像表示ボタン (Button)
if st.button("画像を表示", type="secondary", disabled=not st.session_state.get('search_executed', False)):
    st.write("画像をダウンロードして、色を合成しています... ")
    st.info(f"選択データ: {selected_image}")
    
    # 画像のダミー表示（RGBノイズでシミュレーション）
    # 実際の実装ではここで衛星データを取得・プロットします
    fig, ax = plt.subplots(figsize=(6, 6))
    img_data = np.random.rand(100, 100, 3)
    ax.imshow(img_data)
    ax.set_title(f"Preview: {selected_location}")
    ax.axis('off')
    st.pyplot(fig)

st.markdown("---")

# 3. 運用・テレメトリシミュレーション (ノートブック後半のロジック)
st.subheader("ww 衛星バス部挙動モニタリング")

# 期間モード選択 (ToggleButtonsの再現)
sim_mode = st.radio("期間モード:", ('Event (80s)', 'Long-term (1day)'), horizontal=True)

# グラフ描画エリア
fig2, ax2 = plt.subplots(figsize=(10, 4))

if sim_mode == 'Event (80s)':
    # 短期イベントモード：電力スパイクの再現
    t = np.linspace(0, 80, 200)
    # ガウス関数的な電力消費スパイク
    power = 300 + 400 * np.exp(-0.01 * (t - 40)**2)
    ax2.plot(t, power, color='orange', label='Power Consumption [W]')
    ax2.set_xlabel("Time [sec]")
    ax2.set_ylabel("Power [W]")
    ax2.set_title("イベント実行時の電力消費推移 (80秒間)")
    ax2.legend()
    st.pyplot(fig2)
    
    st.markdown("""
    **解説:**
    - **電力スパイク**: タスキング（撮影や送信）実行時に急激な電力消費が発生しています。
    - **バス電圧**: バッテリー放電により電圧降下が観測される可能性があります。
    """)

else:
    # 長期モード：軌道周回による電力変化
    t = np.linspace(0, 24, 200) # 24時間
    # 90分周期の正弦波（日照・日陰）を模擬
    power = 500 * np.sin(2 * np.pi * t * (60/90)) 
    # 日陰（Eclipse）では発電ゼロ
    power = np.where(power < 0, 0, power)
    
    ax2.plot(t, power, color='cyan', label='Solar Array Power [W]')
    ax2.set_xlabel("Time [hour]")
    ax2.set_ylabel("Power Generation [W]")
    ax2.set_title("1日（長期）の電力収支プロファイル")
    ax2.fill_between(t, power, color='cyan', alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
    
    st.markdown("""
    **解説:**
    - **周期変動**: 衛星が地球の裏側（日陰）に入ると発電量が0になります。
    - **電力収支**: 発電できない期間はバッテリー電力のみで駆動する必要があります。
    """)
