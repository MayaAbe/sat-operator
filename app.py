import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# --- ページ設定 ---
st.set_page_config(
    page_title="衛星運用タスキング訓練アプリ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- タイトルと説明 ---
st.title("🛰️ 人工衛星 運用・タスキング訓練シミュレーター")
st.markdown("""
このアプリケーションは、衛星運用の基本的なタスキング（撮影計画）と、
コマンド送信時のテレメトリ変動を模擬体験するためのトレーニングツールです。
""")

# --- サイドバー：設定入力エリア ---
st.sidebar.header("1. タスキング設定 (Tasking)")

# 1. ターゲット選択
locations = {
    "筑波宇宙センター (JAXA)": {"lat": 36.065, "lon": 140.128},
    "種子島宇宙センター (JAXA)": {"lat": 30.399, "lon": 130.968},
    "東京駅": {"lat": 35.681, "lon": 139.767},
    "富士山": {"lat": 35.360, "lon": 138.727},
    "ワシントンD.C. (US)": {"lat": 38.907, "lon": -77.036},
    "ロンドン (UK)": {"lat": 51.507, "lon": -0.127},
}
selected_loc_name = st.sidebar.selectbox("観測ターゲットを選択:", list(locations.keys()))
target_loc = locations[selected_loc_name]

# 2. 日時指定
selected_date = st.sidebar.date_input("観測日を指定:", datetime.date(2026, 1, 1))

# 3. パラメータ調整
fov = st.sidebar.slider("観測視野角 (FOV) [度]:", 0.01, 0.5, 0.1, 0.01)

st.sidebar.markdown("---")
st.sidebar.header("2. 運用モード設定")
mode = st.sidebar.radio("シミュレーション期間:", ("Event Mode (80s)", "Long-term Mode (1 orbit)"))

# --- メインエリア：可視化と操作 ---

# タブによる機能切り替え
tab1, tab2, tab3 = st.tabs(["🗺️ 軌道・FOV確認", "📷 衛星画像検索", "ww テレメトリ監視"])

with tab1:
    st.subheader(f"ターゲット確認: {selected_loc_name}")

    # 簡易的な地図表示 (Streamlitの機能を使用)
    map_data = pd.DataFrame({
        'lat': [target_loc["lat"]],
        'lon': [target_loc["lon"]]
    })
    st.map(map_data, zoom=10)

    st.info(f"📍 座標: 北緯 {target_loc['lat']}°, 東経 {target_loc['lon']}° | 🔭 設定FOV: {fov}°")
    st.markdown("本来はここに衛星のグランドトラック（地上軌跡）と可視範囲がオーバーレイ表示されます。")

with tab2:
    st.subheader("アーカイブ画像検索")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("条件に合致する過去の衛星画像を検索します。")
        if st.button("検索実行", type="primary"):
            st.session_state['search_done'] = True

    with col2:
        if st.session_state.get('search_done'):
            # ダミーデータの生成
            dummy_images = [
                f"[2024-01-02] Cloud: 12% - {selected_loc_name}",
                f"[2023-12-25] Cloud: 45% - {selected_loc_name}",
                f"[2023-11-10] Cloud: 05% - {selected_loc_name}"
            ]
            img_choice = st.selectbox("画像を選択してください:", dummy_images)

            if st.button("画像を表示"):
                # ここではランダムノイズで画像を模擬していますが、本来は画像データを表示します
                st.image(np.random.rand(100,100,3), caption=img_choice, width=400)
                st.success("画像を取得しました。")

with tab3:
    st.subheader("システム挙動シミュレーション")

    # データの生成（シミュレーション）
    if mode == "Event Mode (80s)":
        t = np.linspace(0, 80, 100)
        # イベント時の電力消費スパイクを模擬
        power = 200 + 100 * np.exp(-((t - 40)**2) / 20) + np.random.normal(0, 5, 100)
        title_text = "イベント（撮影）時のバス電圧/消費電力推移"
        x_label = "Time [sec]"
    else:
        t = np.linspace(0, 90, 100) # 90分（約1周回）
        # 日照・日陰による発電量変化を模擬
        power = 500 * (np.sin(t * 0.1) > 0).astype(float) * np.sin(t*0.1) + 20
        title_text = "1周回(90分)の電力収支プロファイル"
        x_label = "Time [min]"

    # グラフ描画
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, power, color='orange', linewidth=2)
    ax.set_title(title_text)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Power [W]")
    ax.grid(True)

    st.pyplot(fig)

    # アラート機能（教育用追加機能）
    if np.max(power) > 600:
        st.error("⚠️ 警告: 電力消費が許容範囲を超過する可能性があります！")
    else:
        st.success("✅ ステータス: 正常範囲内")

# --- フッター ---
st.markdown("---")
st.caption("Satellite Tasking Trainer Prototype | Powered by Streamlit")
