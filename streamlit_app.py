import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os

# --- 0. 設定とキャッシュ処理 ---
try:
    temp_dir = os.path.join(tempfile.gettempdir(), "yfinance_cache_custom")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    yf.set_tz_cache_location(temp_dir)
except Exception:
    pass

st.set_page_config(page_title="基本4資産ポートフォリオ分析", layout="wide")
st.title("現代ポートフォリオ理論：基本4資産シミュレーター (CML付)")

# --- 1. サイドバー設定 ---
st.sidebar.header("設定")

# 基本4資産（国内株、先進国株、国内債、先進国債）すべて東証ETF
default_tickers = "1306.T,1550.T,2510.T,1677.T"

tickers_input = st.sidebar.text_input("銘柄コード (カンマ区切り)", default_tickers)

# 無リスク金利
risk_free_rate_pct = st.sidebar.number_input("無リスク金利 (%)", value=1.0, step=0.05)
risk_free_rate = risk_free_rate_pct / 100.0

# 開始日
start_date = st.sidebar.date_input("開始日", pd.to_datetime("2010-01-01"))
num_portfolios = st.sidebar.slider("シミュレーション回数", 1000, 10000, 5000)

# --- 2. メイン処理 ---
if st.sidebar.button("シミュレーション実行"):
    with st.spinner('データを取得・計算中...'):
        try:
            input_tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
            
            if not input_tickers:
                st.error("銘柄コードを入力してください。")
                st.stop()

            # データ取得
            raw_data = yf.download(input_tickers, start=start_date, progress=False, auto_adjust=False)

            if raw_data.empty:
                st.error("データを取得できませんでした。")
                st.stop()

            # データ整形
            if isinstance(raw_data.columns, pd.MultiIndex):
                if 'Adj Close' in raw_data.columns.get_level_values(0):
                    df = raw_data['Adj Close']
                else:
                    df = raw_data
            else:
                if 'Adj Close' in raw_data.columns:
                    df = raw_data[['Adj Close']]
                else:
                    df = raw_data

            df = df.dropna()
            if isinstance(df, pd.Series):
                df = df.to_frame()
            
            active_tickers = df.columns.tolist()
            
            if len(df) < 20:
                st.error("データ数が少なすぎます。期間または銘柄を確認してください。")
                st.stop()

            if len(active_tickers) < 2:
                st.warning("ポートフォリオを組むには最低2つの有効な銘柄が必要です。")
                st.stop()

            # --- 計算処理 ---
            log_returns = np.log(df / df.shift(1)).dropna()
            cov_matrix = log_returns.cov() * 252
            corr_matrix = log_returns.corr() 
            expected_returns = log_returns.mean() * 252
            
            num_assets = len(active_tickers)
            results = np.zeros((3, num_portfolios))
            weights_record = []

            for i in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                weights_record.append(weights)
                
                p_return = np.sum(weights * expected_returns)
                p_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # シャープレシオ (Rf反映)
                sharpe = (p_return - risk_free_rate) / p_std_dev
                
                results[0,i] = p_std_dev
                results[1,i] = p_return
                results[2,i] = sharpe

            results_frame = pd.DataFrame(results.T, columns=['Risk', 'Return', 'Sharpe'])

            # 最適ポートフォリオ
            max_sharpe_idx = results_frame['Sharpe'].idxmax()
            max_sharpe_port = results_frame.loc[max_sharpe_idx]
            max_sharpe_weights = weights_record[int(max_sharpe_idx)]

            min_risk_idx = results_frame['Risk'].idxmin()
            min_risk_port = results_frame.loc[min_risk_idx]
            min_risk_weights = weights_record[int(min_risk_idx)]

            # 個別銘柄データ準備
            ind_risks = np.sqrt(np.diag(cov_matrix))
            ind_returns = expected_returns.values
            
            # 日本語ラベル辞書
            ticker_map = {
                "1306.T": "国内株式(1306)",
                "1550.T": "先進国株(1550)",
                "2510.T": "国内債券(2510)",
                "1677.T": "先進国債(1677)"
            }
            display_names = [ticker_map.get(t, t) for t in active_tickers]

            ind_df = pd.DataFrame({
                'Risk': ind_risks,
                'Return': ind_returns,
                'Ticker': display_names, 
            })

            # --- 3. タブ表示 ---
            tab1, tab2 = st.tabs(["📊 効率的フロンティア", "🔥 相関行列ヒートマップ"])

            with tab1:
                # CML計算
                cml_x = np.linspace(0, results_frame['Risk'].max() * 1.5, 50)
                cml_y = risk_free_rate + max_sharpe_port['Sharpe'] * cml_x

                fig = go.Figure()

                # シミュレーション
                fig.add_trace(go.Scatter(
                    x=results_frame['Risk'], y=results_frame['Return'],
                    mode='markers',
                    marker=dict(
                        color=results_frame['Sharpe'],
                        colorscale='Viridis',
                        size=4,
                        showscale=True,
                        colorbar=dict(title='Sharpe Ratio')
                    ),
                    name='シミュレーション'
                ))

                # CML
                fig.add_trace(go.Scatter(
                    x=cml_x, y=cml_y, mode='lines',
                    line=dict(color='green', dash='dash', width=2),
                    name='資本市場線 (CML)'
                ))

                # Rf
                fig.add_trace(go.Scatter(
                    x=[0], y=[risk_free_rate], mode='markers+text',
                    text=["無リスク資産"], textposition="top right",
                    marker=dict(color='green', size=12, symbol='square'),
                    name='無リスク資産'
                ))

                # 個別銘柄
                fig.add_trace(go.Scatter(
                    x=ind_df['Risk'], y=ind_df['Return'], mode='markers+text',
                    text=ind_df['Ticker'], textposition="top center",
                    marker=dict(color='black', size=10, symbol='circle'),
                    name='個別銘柄'
                ))

                # 接点ポートフォリオ
                fig.add_trace(go.Scatter(
                    x=[max_sharpe_port['Risk']], y=[max_sharpe_port['Return']],
                    mode='markers', marker=dict(color='red', size=18, symbol='star'),
                    name='接点ポートフォリオ'
                ))

                # 最小分散ポートフォリオ
                fig.add_trace(go.Scatter(
                    x=[min_risk_port['Risk']], y=[min_risk_port['Return']],
                    mode='markers', marker=dict(color='blue', size=15, symbol='diamond'),
                    name='最小分散'
                ))

                fig.update_layout(
                    height=600,
                    title='効率的フロンティア：基本4資産 (円建て)',
                    xaxis_title='リスク (標準偏差)', yaxis_title='期待リターン (年率)',
                    xaxis=dict(range=[0, results_frame['Risk'].max() * 1.2]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                col_best, col_safe = st.columns(2)

                def display_stats(container, title, port_data, weights, color_code):
                    with container:
                        st.markdown(f"<h3 style='color: {color_code};'>{title}</h3>", unsafe_allow_html=True)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("期待リターン", f"{port_data['Return']:.2%}")
                        c2.metric("リスク", f"{port_data['Risk']:.2%}")
                        c3.metric("シャープレシオ", f"{port_data['Sharpe']:.2f}")

                        df_w = pd.DataFrame({
                            '銘柄': display_names,
                            '比率': weights * 100
                        }).sort_values('比率', ascending=False)
                        
                        st.dataframe(
                            df_w,
                            column_config={
                                "銘柄": "資産クラス",
                                "比率": st.column_config.ProgressColumn(
                                    "構成比率 (%)", format="%.1f%%", min_value=0, max_value=100
                                )
                            },
                            use_container_width=True, hide_index=True
                        )

                display_stats(col_best, "★ 接点ポートフォリオ (最大効率)", max_sharpe_port, max_sharpe_weights, "#FF4B4B")
                display_stats(col_safe, "◆ 最小分散ポートフォリオ (安定重視)", min_risk_port, min_risk_weights, "#1E90FF")

                # --- 【追加機能】個別銘柄の統計データ表示 ---
                st.markdown("---")
                st.subheader("個別銘柄のパフォーマンス指標")
                
                # 個別銘柄のシャープレシオ計算
                ind_sharpes = (ind_returns - risk_free_rate) / ind_risks
                
                # データフレーム作成
                asset_stats_df = pd.DataFrame({
                    '資産クラス': display_names,
                    '期待リターン': ind_returns,
                    'リスク (標準偏差)': ind_risks,
                    'シャープレシオ': ind_sharpes
                }).sort_values('シャープレシオ', ascending=False) # シャープレシオ順にソート

                st.dataframe(
                    asset_stats_df,
                    column_config={
                        "資産クラス": "資産クラス",
                        "期待リターン": st.column_config.NumberColumn(format="%.2%"),
                        "リスク (標準偏差)": st.column_config.NumberColumn(format="%.2%"),
                        "シャープレシオ": st.column_config.NumberColumn(format="%.2f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )

            with tab2:
                st.subheader("資産間の相関係数")
                corr_matrix_display = corr_matrix.copy()
                corr_matrix_display.index = display_names
                corr_matrix_display.columns = display_names

                heatmap_fig = px.imshow(
                    corr_matrix_display,
                    text_auto=True, aspect="auto",
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    title="相関行列ヒートマップ"
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)

        except Exception as e:
            st.error(f"エラー詳細: {e}")
            st.write("設定を変更して再度お試しください。")

else:
    st.info("サイドバーの設定を確認し、「シミュレーション実行」を押してください。")