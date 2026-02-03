import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
from scipy.optimize import minimize

# --- 定数（計算・表示用）---
ANNUALIZE_TRADING_DAYS = 252  # 年率化：株価は営業日ベース
MIN_OBSERVATIONS = 20  # 最小必要データ件数
CML_X_MULTIPLIER = 1.5  # CML 描画の x 軸範囲（リスク最大の倍数）
AXIS_RISK_MULTIPLIER = 1.2  # 効率的フロンティア図の x 軸上限

# --- 0. 設定とキャッシュ処理 ---
try:
    temp_dir = os.path.join(tempfile.gettempdir(), "yfinance_cache_custom")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    yf.set_tz_cache_location(temp_dir)
except Exception:
    pass

# 価格データ取得（銘柄・開始日でキャッシュ、最大1時間）
@st.cache_data(ttl=3600)
def fetch_price_data(tickers_tuple, start_date):
    raw_data = yf.download(
        list(tickers_tuple), start=start_date, progress=False, auto_adjust=False
    )
    if raw_data.empty:
        raise ValueError("データを取得できませんでした。銘柄コードまたは期間を確認してください。")
    if isinstance(raw_data.columns, pd.MultiIndex):
        if "Adj Close" in raw_data.columns.get_level_values(0):
            df = raw_data["Adj Close"].copy()
        else:
            df = raw_data.copy()
    else:
        if "Adj Close" in raw_data.columns:
            df = raw_data[["Adj Close"]].copy()
        else:
            df = raw_data.copy()
    df = df.dropna()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

# 対象資産プリセット（東証ETF）
PRESET_4_ASSETS = "1306.T,1550.T,2510.T,1677.T"  # 国内株・先進国株・国内債・先進国債
PRESET_MARKET_WIDE = (
    "1306.T,1550.T,1348.T,2510.T,1677.T,1328.T,1346.T,1329.T"
)  # 国内株・先進国株・新興国株・国内債・先進国債・金・不動産・コモディティ

# 銘柄の日本語表示名（プリセット外は銘柄コードのまま）
TICKER_DISPLAY_NAMES = {
    "1306.T": "国内株式(1306)",
    "1550.T": "先進国株(1550)",
    "1348.T": "新興国株(1348)",
    "2510.T": "国内債券(2510)",
    "1677.T": "先進国債(1677)",
    "1328.T": "金(1328)",
    "1346.T": "不動産(1346)",
    "1329.T": "コモディティ(1329)",
}

st.set_page_config(page_title="ポートフォリオ分析", layout="wide")
st.title("現代ポートフォリオ理論：シミュレーター (CML付)")

# --- 1. サイドバー設定 ---
st.sidebar.header("設定")

asset_universe = st.sidebar.radio(
    "対象資産",
    options=["基本4資産", "資産市場全体（プリセット）", "カスタム"],
    index=0,
    help="基本4資産＝国内株・先進国株・国内債・先進国債。資産市場全体＝株・債・金・不動産・コモディティ等の東証ETF。",
)

if asset_universe == "カスタム":
    tickers_input = st.sidebar.text_input(
        "銘柄コード (カンマ区切り)", PRESET_4_ASSETS
    )
else:
    tickers_input = PRESET_4_ASSETS if asset_universe == "基本4資産" else PRESET_MARKET_WIDE
    st.sidebar.caption(f"銘柄: {tickers_input}")

# 無リスク金利
risk_free_rate_pct = st.sidebar.number_input("無リスク金利 (%)", value=1.0, step=0.05)
risk_free_rate = risk_free_rate_pct / 100.0

# 開始日
start_date = st.sidebar.date_input("開始日", pd.to_datetime("2010-01-01"))
num_portfolios = st.sidebar.slider("シミュレーション回数", 1000, 10000, 5000)

# 取れるリスク（目標リスク）の指定
use_target_risk = st.sidebar.checkbox(
    "取れるリスク（目標リスク）を指定する",
    value=False,
    help="指定したリスク（年率・標準偏差）で期待リターンを最大化するポートフォリオを表示します。",
)
target_risk_pct = None
if use_target_risk:
    target_risk_pct = st.sidebar.number_input(
        "目標リスク（年率・標準偏差 %）",
        min_value=0.5,
        max_value=50.0,
        value=10.0,
        step=0.5,
        format="%.1f",
        help="例: 10 → 年率リスク（標準偏差）10%",
    )

# --- 2. セッション状態（キャッシュキー・結果の保持）---
if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None
if "portfolio_cache_key" not in st.session_state:
    st.session_state.portfolio_cache_key = None

def make_cache_key():
    tickers_list = sorted([t.strip() for t in tickers_input.split(",") if t.strip()])
    return (tuple(tickers_list), start_date, risk_free_rate_pct, num_portfolios, target_risk_pct)

current_cache_key = make_cache_key()
force_run = st.sidebar.button("シミュレーション実行")
# 初回表示時は結果がないので自動で1回実行する
run_simulation = force_run or (st.session_state.portfolio_result is None)

def run_and_store_result():
    input_tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if not input_tickers:
        st.error("銘柄コードを入力してください。")
        return None

    df = fetch_price_data(tuple(sorted(input_tickers)), start_date)
    active_tickers = df.columns.tolist()

    if len(df) < MIN_OBSERVATIONS:
        raise ValueError(
            f"データ数が少なすぎます（{MIN_OBSERVATIONS}件未満）。開始日を遅らせるか、銘柄を確認してください。"
        )
    if len(active_tickers) < 2:
        raise ValueError(
            "有効な銘柄が2つ未満です。取得できた銘柄のみでポートフォリオを組むには2つ以上必要です。"
        )

    # 年率化済み期待リターン・共分散（営業日ベース）
    log_returns = np.log(df / df.shift(1)).dropna()
    cov_matrix = log_returns.cov().values * ANNUALIZE_TRADING_DAYS
    corr_matrix = log_returns.corr()
    mu = log_returns.mean().values * ANNUALIZE_TRADING_DAYS  # (num_assets,)
    num_assets = len(active_tickers)

    # --- シミュレーション（ベクトル化）---
    W = np.random.random((num_portfolios, num_assets))
    W /= W.sum(axis=1, keepdims=True)
    p_returns = W @ mu
    p_vars = (W @ cov_matrix * W).sum(axis=1)
    p_stds = np.sqrt(np.maximum(p_vars, 1e-12))
    sharpes = (p_returns - risk_free_rate) / p_stds
    results_frame = pd.DataFrame({
        "Risk": p_stds,
        "Return": p_returns,
        "Sharpe": sharpes,
    })

    # --- scipy で接点（最大シャープ）・最小分散を厳密に最適化 ---
    bnds = tuple((0.0, 1.0) for _ in range(num_assets))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    x0 = np.ones(num_assets) / num_assets

    def port_variance(w):
        return w @ cov_matrix @ w

    min_risk_res = minimize(
        port_variance, x0=x0, method="SLSQP", bounds=bnds, constraints=cons
    )
    min_risk_weights = min_risk_res.x
    min_risk_std = np.sqrt(min_risk_weights @ cov_matrix @ min_risk_weights)
    min_risk_return = min_risk_weights @ mu
    min_risk_sharpe = (min_risk_return - risk_free_rate) / max(min_risk_std, 1e-12)
    min_risk_port = pd.Series({
        "Risk": min_risk_std,
        "Return": min_risk_return,
        "Sharpe": min_risk_sharpe,
    })

    def neg_sharpe(w):
        r = w @ mu
        s = np.sqrt(max(w @ cov_matrix @ w, 1e-12))
        return -(r - risk_free_rate) / s

    max_sharpe_res = minimize(
        neg_sharpe, x0=x0, method="SLSQP", bounds=bnds, constraints=cons
    )
    max_sharpe_weights = max_sharpe_res.x
    max_sharpe_std = np.sqrt(max_sharpe_weights @ cov_matrix @ max_sharpe_weights)
    max_sharpe_return = max_sharpe_weights @ mu
    max_sharpe_sharpe = (max_sharpe_return - risk_free_rate) / max(max_sharpe_std, 1e-12)
    max_sharpe_port = pd.Series({
        "Risk": max_sharpe_std,
        "Return": max_sharpe_return,
        "Sharpe": max_sharpe_sharpe,
    })

    # --- 目標リスクを指定した場合：そのリスクで期待リターン最大化 ---
    target_risk_port = None
    target_risk_weights = None
    if target_risk_pct is not None:
        target_sigma = (target_risk_pct / 100.0) ** 2  # 分散に変換
        if target_risk_pct / 100.0 < min_risk_std - 1e-6:
            # 指定リスクが最小分散より小さい場合は最小分散ポートフォリオを表示
            target_risk_weights = min_risk_weights.copy()
            target_risk_port = min_risk_port.copy()
        else:
            def neg_return(w):
                return -(w @ mu)

            cons_target = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w: (w @ cov_matrix @ w) - target_sigma},
            )
            res_target = minimize(
                neg_return, x0=x0, method="SLSQP", bounds=bnds, constraints=cons_target
            )
            if res_target.success:
                target_risk_weights = res_target.x
                tr_std = np.sqrt(target_risk_weights @ cov_matrix @ target_risk_weights)
                tr_ret = target_risk_weights @ mu
                tr_sharpe = (tr_ret - risk_free_rate) / max(tr_std, 1e-12)
                target_risk_port = pd.Series({
                    "Risk": tr_std,
                    "Return": tr_ret,
                    "Sharpe": tr_sharpe,
                })

    ind_risks = np.sqrt(np.diag(cov_matrix))
    ind_returns = mu
    display_names = [TICKER_DISPLAY_NAMES.get(t, t) for t in active_tickers]
    ind_df = pd.DataFrame({
        "Risk": ind_risks,
        "Return": ind_returns,
        "Ticker": display_names,
    })
    ind_sharpes = (ind_returns - risk_free_rate) / ind_risks
    asset_stats_df = pd.DataFrame({
        "資産クラス": display_names,
        "期待リターン": ind_returns,
        "リスク (標準偏差)": ind_risks,
        "シャープレシオ": ind_sharpes,
    }).sort_values("シャープレシオ", ascending=False)

    return {
        "results_frame": results_frame,
        "max_sharpe_port": max_sharpe_port,
        "max_sharpe_weights": max_sharpe_weights,
        "min_risk_port": min_risk_port,
        "min_risk_weights": min_risk_weights,
        "target_risk_port": target_risk_port,
        "target_risk_weights": target_risk_weights,
        "corr_matrix": corr_matrix,
        "display_names": display_names,
        "ind_df": ind_df,
        "asset_stats_df": asset_stats_df,
        "risk_free_rate": risk_free_rate,
    }

def render_result(res):
    results_frame = res["results_frame"]
    max_sharpe_port = res["max_sharpe_port"]
    max_sharpe_weights = res["max_sharpe_weights"]
    min_risk_port = res["min_risk_port"]
    min_risk_weights = res["min_risk_weights"]
    target_risk_port = res.get("target_risk_port")
    target_risk_weights = res.get("target_risk_weights")
    corr_matrix = res["corr_matrix"]
    display_names = res["display_names"]
    ind_df = res["ind_df"]
    asset_stats_df = res["asset_stats_df"]
    risk_free_rate = res["risk_free_rate"]

    tab1, tab2 = st.tabs(["📊 効率的フロンティア", "🔥 相関行列ヒートマップ"])

    with tab1:
        risk_max = results_frame["Risk"].max()
        cml_x = np.linspace(0, risk_max * CML_X_MULTIPLIER, 50)
        cml_y = risk_free_rate + max_sharpe_port["Sharpe"] * cml_x

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_frame["Risk"], y=results_frame["Return"],
            mode="markers",
            marker=dict(
                color=results_frame["Sharpe"],
                colorscale="Viridis",
                size=4,
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
            ),
            name="シミュレーション",
        ))
        fig.add_trace(go.Scatter(
            x=cml_x, y=cml_y, mode="lines",
            line=dict(color="green", dash="dash", width=2),
            name="資本市場線 (CML)",
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[risk_free_rate], mode="markers+text",
            text=["無リスク資産"], textposition="top right",
            marker=dict(color="green", size=12, symbol="square"),
            name="無リスク資産",
        ))
        fig.add_trace(go.Scatter(
            x=ind_df["Risk"], y=ind_df["Return"], mode="markers+text",
            text=ind_df["Ticker"], textposition="top center",
            marker=dict(color="black", size=10, symbol="circle"),
            name="個別銘柄",
        ))
        fig.add_trace(go.Scatter(
            x=[max_sharpe_port["Risk"]], y=[max_sharpe_port["Return"]],
            mode="markers", marker=dict(color="red", size=18, symbol="star"),
            name="接点ポートフォリオ",
        ))
        fig.add_trace(go.Scatter(
            x=[min_risk_port["Risk"]], y=[min_risk_port["Return"]],
            mode="markers", marker=dict(color="blue", size=15, symbol="diamond"),
            name="最小分散",
        ))
        if target_risk_port is not None and target_risk_weights is not None:
            fig.add_trace(go.Scatter(
                x=[target_risk_port["Risk"]], y=[target_risk_port["Return"]],
                mode="markers", marker=dict(color="orange", size=16, symbol="hexagon"),
                name="目標リスクポートフォリオ",
            ))
        fig.update_layout(
            height=600,
            title="効率的フロンティア (円建て)",
            xaxis_title="リスク (標準偏差)", yaxis_title="期待リターン (年率)",
            xaxis=dict(range=[0, risk_max * AXIS_RISK_MULTIPLIER]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        n_cols = 3 if (target_risk_port is not None and target_risk_weights is not None) else 2
        cols = st.columns(n_cols)

        def display_stats(container, title, port_data, weights, color_code):
            with container:
                st.markdown(f"<h3 style='color: {color_code};'>{title}</h3>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("期待リターン", f"{port_data['Return']:.2%}")
                c2.metric("リスク", f"{port_data['Risk']:.2%}")
                c3.metric("シャープレシオ", f"{port_data['Sharpe']:.2f}")
                df_w = pd.DataFrame({
                    "銘柄": display_names,
                    "比率": weights * 100,
                }).sort_values("比率", ascending=False)
                st.dataframe(
                    df_w,
                    column_config={
                        "銘柄": "資産クラス",
                        "比率": st.column_config.ProgressColumn(
                            "構成比率 (%)", format="%.1f%%", min_value=0, max_value=100
                        ),
                    },
                    use_container_width=True, hide_index=True,
                )

        display_stats(cols[0], "★ 接点ポートフォリオ (最大効率)", max_sharpe_port, max_sharpe_weights, "#FF4B4B")
        display_stats(cols[1], "◆ 最小分散ポートフォリオ (安定重視)", min_risk_port, min_risk_weights, "#1E90FF")
        if target_risk_port is not None and target_risk_weights is not None:
            display_stats(cols[2], "◎ 目標リスクポートフォリオ", target_risk_port, target_risk_weights, "#FF8C00")

        st.markdown("---")
        st.subheader("個別銘柄のパフォーマンス指標")
        st.dataframe(
            asset_stats_df,
            column_config={
                "資産クラス": "資産クラス",
                "期待リターン": st.column_config.NumberColumn(format="%.2%"),
                "リスク (標準偏差)": st.column_config.NumberColumn(format="%.2%"),
                "シャープレシオ": st.column_config.NumberColumn(format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
        )

    with tab2:
        st.subheader("資産間の相関係数")
        corr_matrix_display = corr_matrix.copy()
        corr_matrix_display.index = display_names
        corr_matrix_display.columns = display_names
        heatmap_fig = px.imshow(
            corr_matrix_display,
            text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="相関行列ヒートマップ",
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

# --- 3. メイン処理 ---
if run_simulation:
    with st.spinner("データを取得・計算中..."):
        try:
            result = run_and_store_result()
            if result is not None:
                st.session_state.portfolio_result = result
                st.session_state.portfolio_cache_key = current_cache_key
                render_result(result)
        except (ConnectionError, TimeoutError, OSError) as e:
            st.error("**ネットワークエラー**：価格データの取得に失敗しました。")
            st.caption("インターネット接続を確認するか、しばらく経ってから再実行してください。")
            with st.expander("詳細"):
                st.code(str(e))
        except ValueError as e:
            st.error("**入力・データエラー**")
            st.write(str(e))
            st.caption("銘柄コード・開始日・銘柄数を確認してください。")
        except KeyError as e:
            st.error("**データ形式エラー**：取得した価格データの形式が想定と異なります。")
            st.caption("銘柄や期間を変えるか、しばらく経ってから再試行してください。")
            with st.expander("詳細"):
                st.code(str(e))
        except Exception as e:
            st.error("**予期しないエラー**が発生しました。")
            st.write("設定を変更して再度お試しください。")
            with st.expander("エラー詳細（開発者向け）"):
                st.code(str(e))
                st.exception(e)

elif (
    st.session_state.portfolio_result is not None
    and st.session_state.portfolio_cache_key == current_cache_key
):
    st.caption("現在の設定に基づくキャッシュ結果を表示しています。設定変更後は「シミュレーション実行」で再計算してください。")
    render_result(st.session_state.portfolio_result)

else:
    st.info("設定を変更しました。「シミュレーション実行」を押すと再計算します。初回は自動で1回実行されます。")