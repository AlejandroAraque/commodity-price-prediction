import streamlit as st
import requests
import numpy as np
import pandas as pd
import sys
import os
import datetime
import plotly.graph_objects as go 
import yfinance as yf


# --- CONFIGURACIÓN ---
API_URL = "http://localhost:8000/predict_direction/"

sys.path.append(".")
from src.dataset import CommodityDataModule


# --- PAGE CONFIG (Dark Mode Default) ---
st.set_page_config(
    page_title="Commodity AI", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- INYECCIÓN CSS FINTECH/BLOOMBERG STYLE ---
st.markdown("""
<style>
    /* === DARK THEME BASE === */
    .stApp {
        background: linear-gradient(180deg, #31333f 0%, #0d1117 100%);
    }
    
    /* === HEADER STYLING === */
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid #00d4ff33;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.1);
    }
    .main-header h1 {
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #e6f1ff;
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* === METRIC CARDS FINTECH === */
    .metric-fintech {
        background: linear-gradient(145deg, #1a1a2e 0%, #0f0f1a 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .metric-fintech:hover {
        border-color: #00d4ff;
        box-shadow: 0 4px 25px rgba(0, 212, 255, 0.15);
        transform: translateY(-2px);
    }
    .metric-label {
        color: #ff4757;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #ff4757;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }
    .metric-delta-up {
        color: #ff4757;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .metric-delta-down {
        color: #ff4757;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* === SIGNAL CARDS === */
    .signal-long {
        background: linear-gradient(145deg, #0a2e1a 0%, #0f3d1f 100%);
        border: 2px solid #00ff88;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
    }
    .signal-short {
        background: linear-gradient(145deg, #2e0a0a 0%, #3d0f0f 100%);
        border: 2px solid #ff4757;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 71, 87, 0.2);
    }
    .signal-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .signal-long .signal-title { color: #00ff88; }
    .signal-short .signal-title { color: #ff4757; }
    
    /* === SIDEBAR STYLING === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
    }
    
    /* === BUTTONS === */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%);
        color: #0a0a0f;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* === TABS STYLING === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0d1117;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #161b22;
        border-radius: 8px;
        color: #e6f1ff;
        border: 1px solid #21262d;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff33, #00ff8833);
        border-color: #00d4ff;
        color: #00d4ff;
    }
    
    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        color: #e6f1ff;
    }
    
    /* === DIVIDER === */
    hr {
        border-color: #21262d;
    }
    
    /* === INFO/WARNING BOXES === */
    .stAlert {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
    }
    
    /* === METRICS OVERRIDE === */
    [data-testid="stMetricValue"] {
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 1.5rem;
        color:#00ff88;
    }
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    
    /* === TICKER BADGE === */
    .ticker-badge {
        display: inline-block;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
        color: #0a0a0f;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-left: 1rem;
    }
    
    /* === STATUS INDICATOR === */
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #00ff88;
        font-size: 0.8rem;
    }
    .status-live::before {
        content: '';
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
            
            /* === SIDEBAR: Títulos y Labels === */
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label p {
    color: #ffffff !important;
}

/* === EXPANDERS: "Sobre el Modelo", "Variables Macro" === */
section[data-testid="stSidebar"] .streamlit-expanderHeader p,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #ffffff !important;
}
/* === EXPANDER "Motor de Inteligencia Artificial" === */

/* Fondo del contenido expandido - gris suave en lugar de negro */
[data-testid="stExpander"] details[open] > div {
    background: #1a1f2e !important;  /* Gris azulado suave */
    border-radius: 0 0 10px 10px;
    padding: 1rem;
}

/* Letra más oscura/gris dentro del expander expandido */
[data-testid="stExpander"] details[open] p,
[data-testid="stExpander"] details[open] span,
[data-testid="stExpander"] details[open] label,
[data-testid="stExpander"] details[open] .stMarkdown {
    color: #a0aec0 !important;  /* Gris claro legible */
}
/* === EXPANDER CERRADO: Título blanco === */
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] details:not([open]) summary span,
[data-testid="stExpander"] details:not([open]) summary p {
    color: #ffffff !important;
}

/* Icono de flecha también blanco */
[data-testid="stExpander"] summary svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* Header del expander (el título clickeable) */
[data-testid="stExpander"] summary {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px;
}

/* Cuando está abierto, conectar header con contenido */
[data-testid="stExpander"] details[open] > summary {
    border-radius: 10px 10px 0 0 !important;
    border-bottom: none !important;
}

/* === MÉTRICAS: "Precio", "Log Return", "Volumen", "Rango" === */
[data-testid="stMetricLabel"] label,
[data-testid="stMetricLabel"] value,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
    color: #ffffff !important;
}

/* === METRIC DELTA TEXT (el texto pequeño debajo del valor) === */
[data-testid="stMetricDelta"] {
    color: #ffffff !important;
}
/* === PANTALLA DE BIENVENIDA === */

/* Títulos h3 (### Bienvenido, ### Activos Disponibles, ### Panel de Métricas) */
.stApp .stMarkdown h3 {
    color: #a0aec0 !important;  /* Gris claro */
}

/* Párrafos y texto general */
.stApp .stMarkdown p {
    color: #8892b0 !important;  /* Gris medio */
}

/* Listas (los bullets de "Para comenzar" y "Activos") */
.stApp .stMarkdown li {
    color: #9ca3af !important;  /* Gris suave */
}

/* Texto en negrita dentro de párrafos */
.stApp .stMarkdown strong {
    color: #c9d1d9 !important;  /* Gris más claro para destacar */
}

</style>
""", unsafe_allow_html=True)


# --- CABECERA FINTECH ---
st.markdown("""
<div class="main-header">
    <h1>🐂 Commodity AI</h1>
    <p>Sistema de Inteligencia Artificial para Predicción Direccional | LSTM/CNN Neural Networks</p>
</div>
""", unsafe_allow_html=True)


# --- BARRA LATERAL MEJORADA ---
with st.sidebar:
    st.markdown("### ⚙️ Panel de Control")
    st.markdown("---")
    
    ticker_select = st.selectbox(
        "🎯 Activo Objetivo",
        ("Oro (GC=F)", "Plata (SI=F)", "Petróleo (CL=F)"), 
        index=0
    )
    
    ticker_map = {"Oro (GC=F)": "GC=F", "Plata (SI=F)": "SI=F", "Petróleo (CL=F)": "CL=F"}
    api_ticker_map = {"GC=F": "GOLD", "SI=F": "SILVER", "CL=F": "OIL"}
    ticker = ticker_map[ticker_select]
    
    st.markdown("---")
    
    # Status de conexión
    st.markdown('<div class="status-live">API EN LÍNEA</div>', unsafe_allow_html=True)
    st.code(API_URL, language=None)
    
    st.markdown("---")
    
    with st.expander("📊 Sobre el Modelo", expanded=False):
        st.markdown("""
        **Arquitectura:** LSTM + CNN Híbrido  
        **Input:** Retornos Logarítmicos + Macros  
        **Ventana:** 30 días  
        **Estrategia:** Conservadora (>50% Conf.)
        """)
    
    with st.expander("📈 Variables Macro", expanded=False):
        st.markdown("""
        - `DXY` - Índice Dólar  
        - `VIX` - Volatilidad  
        - `US10Y` - Bono 10 años  
        - `SPX` - S&P 500  
        """)
    
    st.markdown("---")
    analyze_button = st.button("🔮 ANALIZAR MERCADO", type="primary", use_container_width=True)


# --- LÓGICA PRINCIPAL ---
main_container = st.container()

if analyze_button:
    with main_container:
        with st.spinner(f"📡 Conectando con mercados en tiempo real..."):
            try:
                # ---------------------------------------------------------
                # 1. CAPA VISUAL (UI): Descarga de precios para el Humano
                # ---------------------------------------------------------
                ticker_obj = yf.Ticker(ticker)
                ui_data = ticker_obj.history(period="6mo", interval="1d")
                
                # --- LIMPIEZA NUCLEAR PARA GRAFICA ---
                ui_data.reset_index(inplace=True)
                
                new_cols = {}
                for col in ui_data.columns:
                    c_name = str(col).lower()
                    if 'date' in c_name or 'time' in c_name:
                        new_cols[col] = 'Date'
                    elif 'open' in c_name: new_cols[col] = 'Open'
                    elif 'high' in c_name: new_cols[col] = 'High'
                    elif 'low' in c_name: new_cols[col] = 'Low'
                    elif 'close' in c_name: new_cols[col] = 'Close'
                    elif 'volume' in c_name: new_cols[col] = 'Volume'
                
                ui_data.rename(columns=new_cols, inplace=True)
                
                if 'Date' in ui_data.columns:
                    ui_data['Date'] = pd.to_datetime(ui_data['Date']).dt.tz_localize(None)
                    ui_data['Date_Str'] = ui_data['Date'].dt.strftime('%Y-%m-%d')
                
                if 'Volume' in ui_data.columns:
                    ui_data = ui_data[ui_data['Volume'] > 0]
                ui_data = ui_data.dropna(subset=['Close'])
                
                if not ui_data.empty and 'Close' in ui_data.columns:
                    last_close = float(ui_data['Close'].iloc[-1])
                    prev_close = float(ui_data['Close'].iloc[-2])
                    log_ret = np.log(last_close / prev_close)
                    pct_change = (last_close - prev_close) / prev_close * 100
                    
                    # === MÉTRICAS EN GRID MODERNO ===
                    st.markdown(f"### 📊 Dashboard en Tiempo Real <span class='ticker-badge'>{ticker}</span>", unsafe_allow_html=True)
                    
                    m1, m2, m3, m4 = st.columns(4)
                    
                    with m1:
                        delta_color = "normal" if pct_change >= 0 else "inverse"
                        st.metric(
                            label="💰 Precio Actual",
                            value=f"${last_close:,.2f}",
                            delta=f"{pct_change:+.2f}%",
                            delta_color=delta_color
                        )
                    
                    with m2:
                        st.metric(
                            label="📐 Log Return (1D)",
                            value=f"{log_ret:.4f}",
                            delta="Input modelo",
                            delta_color="off"
                        )
                    
                    with m3:
                        if 'Volume' in ui_data.columns:
                            vol = int(ui_data['Volume'].iloc[-1])
                            vol_prev = int(ui_data['Volume'].iloc[-2])
                            vol_change = ((vol - vol_prev) / vol_prev * 100) if vol_prev > 0 else 0
                            st.metric(
                                label="📊 Volumen",
                                value=f"{vol:,}",
                                delta=f"{vol_change:+.1f}%"
                            )
                        else:
                            st.metric(label="📊 Volumen", value="N/A")
                    
                    with m4:
                        high_52w = ui_data['High'].max()
                        low_52w = ui_data['Low'].min()
                        range_pct = ((last_close - low_52w) / (high_52w - low_52w)) * 100
                        st.metric(
                            label="📈 Rango 6M",
                            value=f"{range_pct:.0f}%",
                            delta=f"${low_52w:,.0f} - ${high_52w:,.0f}",
                            delta_color="off"
                        )
                    
                    st.markdown("---")
                    
                    # === TABS PARA ORGANIZACIÓN ===
                    tab_chart, tab_stats, tab_data = st.tabs(["📉 Gráfico ", "📊 Estadísticas", "🗂️ Datos "])
                    
                    with tab_chart:
                        # === GRÁFICO CANDLESTICK ESTILO BLOOMBERG ===
                        fig = go.Figure(data=[go.Candlestick(
                            x=ui_data['Date_Str'], 
                            open=ui_data['Open'],
                            high=ui_data['High'],
                            low=ui_data['Low'],
                            close=ui_data['Close'],
                            name=ticker,
                            increasing_line_color='#00ff88',
                            decreasing_line_color='#ff4757',
                            increasing_fillcolor='#00ff88',
                            decreasing_fillcolor='#ff4757',
                            increasing_line_width=1,
                            decreasing_line_width=1
                        )])
                        
                        # Añadir SMA 20 como referencia
                        ui_data['SMA20'] = ui_data['Close'].rolling(window=20).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=ui_data['Date_Str'],
                            y=ui_data['SMA20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='#00d4ff', width=1.5, dash='dot'),
                            opacity=0.7
                        ))
                        
                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='#0d1117',
                            height=500,
                            margin=dict(l=10, r=10, t=30, b=10),
                            yaxis_title="Precio (USD)",
                            yaxis=dict(
                                gridcolor='#21262d',
                                zerolinecolor='#21262d',
                                tickfont=dict(color='#ffffff'),
                                title=dict(
                                    text="Precio (USD)",
                                    font=dict(color='#ffffff')  # ✅ CORRECTO
                                )
                            ),
                            xaxis=dict(
                                type='category',
                                categoryorder='category ascending',
                                nticks=10,
                                tickangle=-45,
                                gridcolor='#21262d',
                                tickfont=dict(color='#ffffff', size=10)
                            ),
                            xaxis_rangeslider_visible=False,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                font=dict(color='#ffffff')
                            ),
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab_stats:
                        stat_col1, stat_col2 = st.columns(2)
                        
                        with stat_col1:
                            st.markdown("#### 📈 Estadísticas de Precio")
                            stats_data = {
                                "Métrica": ["Máximo 6M", "Mínimo 6M", "Promedio", "Volatilidad (Std)", "Último Cierre"],
                                "Valor": [
                                    f"${ui_data['High'].max():,.2f}",
                                    f"${ui_data['Low'].min():,.2f}",
                                    f"${ui_data['Close'].mean():,.2f}",
                                    f"${ui_data['Close'].std():,.2f}",
                                    f"${last_close:,.2f}"
                                ]
                            }
                            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
                        
                        with stat_col2:
                            st.markdown("#### 📊 Retornos Logarítmicos")
                            ui_data['LogRet'] = np.log(ui_data['Close'] / ui_data['Close'].shift(1))
                            ret_stats = {
                                "Métrica": ["Retorno Promedio", "Retorno Máximo", "Retorno Mínimo", "Sharpe Ratio (aprox)", "Sesgo"],
                                "Valor": [
                                    f"{ui_data['LogRet'].mean()*100:.4f}%",
                                    f"{ui_data['LogRet'].max()*100:.4f}%",
                                    f"{ui_data['LogRet'].min()*100:.4f}%",
                                    f"{(ui_data['LogRet'].mean() / ui_data['LogRet'].std()) * np.sqrt(252):.2f}",
                                    f"{ui_data['LogRet'].skew():.3f}"
                                ]
                            }
                            st.dataframe(pd.DataFrame(ret_stats), use_container_width=True, hide_index=True)
                    
                    with tab_data:
                        st.markdown("#### 🗂️ Últimos 10 Registros")
                        display_cols = ['Date_Str', 'Open', 'High', 'Low', 'Close', 'Volume']
                        display_df = ui_data[display_cols].tail(10).copy()
                        display_df.columns = ['Fecha', 'Apertura', 'Máximo', 'Mínimo', 'Cierre', 'Volumen']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                else:
                    st.error("❌ No se pudieron descargar datos visuales de Yahoo Finance.")

                # ---------------------------------------------------------
                # 2. CAPA IA (Model): Generación de Features
                # ---------------------------------------------------------
                st.markdown("---")
                
                with st.expander("🧠 Motor de Inteligencia Artificial", expanded=True):
                    
                    progress_col1, progress_col2 = st.columns([3, 1])
                    
                    with progress_col1:
                        status_placeholder = st.empty()
                        status_placeholder.info("⚙️ Generando vectores de características multivariantes...")
                    
                    today = datetime.date.today()
                    #today = datetime.date(2025, 11, 28)
                    start_fast = today - datetime.timedelta(days=730)
                    
                    dm = CommodityDataModule(
                        ticker=ticker, 
                        start_date=start_fast.strftime("%Y-%m-%d"), 
                        window_size=30
                    )
                    dm.prepare_data()
                    dm.setup(stage=None)
                    
                    if len(dm.test_dataset) > 0:
                        last_window_tensor = dm.test_dataset.tensors[0][-1]
                        features_list = last_window_tensor.numpy().flatten().tolist()
                        
                        status_placeholder.success(f"✅ Vector generado: {len(features_list)} features")

                        payload = {
                            "ticker": api_ticker_map[ticker],
                            "features": features_list
                        }
                        
                        # ---------------------------------------------------------
                        # 3. LLAMADA A LA API
                        # ---------------------------------------------------------
                        try:
                            response = requests.post(API_URL, json=payload)
                            
                            if response.status_code == 200:
                                result = response.json()
                                
                                direction = result["prediction_direction"]
                                prob = result["probability_up"]
                                conf_str = result["confidence"]
                                model_used = result.get('model_path_used', 'Desconocido')
                                
                                st.markdown("---")
                                st.markdown("## 🤖 Predicción Neuronal (T+1)")
                                
                                res_col1, res_col2 = st.columns([2, 1])
                                
                                with res_col1:
                                    if "SUBE" in direction:
                                        st.markdown("""
                                        <div class="signal-long">
                                            <div class="signal-title">🚀 SEÑAL: COMPRA (LONG)</div>
                                            <p style="color: #a8e6cf; font-size: 1.1rem;">El modelo neuronal estima una tendencia alcista para el próximo cierre.</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.markdown(f"**Probabilidad de alza:** {conf_str}")
                                    else:
                                        st.markdown("""
                                        <div class="signal-short">
                                            <div class="signal-title">🔻 SEÑAL: VENTA (SHORT)</div>
                                            <p style="color: #ffb3b3; font-size: 1.1rem;">El modelo neuronal estima una tendencia bajista para el próximo cierre.</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        st.markdown(f"**Probabilidad de baja:** {100 - (prob*100):.2f}%")
                                    
                                with res_col2:
                                    st.markdown("#### 🎯 Confianza del Modelo")
                                    
                                    # Gauge visual de confianza
                                    confidence_value = prob if "SUBE" in direction else (1 - prob)
                                    
                                    fig_gauge = go.Figure(go.Indicator(
                                        mode = "gauge+number",
                                        value = confidence_value * 100,
                                        number = {'suffix': "%", 'font': {'color': '#00d4ff', 'size': 40}},
                                        gauge = {
                                            'axis': {'range': [0, 100], 'tickcolor': "#e6f1ff"},
                                            'bar': {'color': "#00ff88" if confidence_value >= 0.5 else "#ff4757"},
                                            'bgcolor': "#0d1117",
                                            'borderwidth': 2,
                                            'bordercolor': "#21262d",
                                            'steps': [
                                                {'range': [0, 50], 'color': '#1a0a0a'},
                                                {'range': [50, 100], 'color': '#0a1a0a'}
                                            ],
                                            'threshold': {
                                                'line': {'color': "#00d4ff", 'width': 2},
                                                'thickness': 0.8,
                                                'value': 50
                                            }
                                        }
                                    ))
                                    
                                    fig_gauge.update_layout(
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        height=200,
                                        margin=dict(l=20, r=20, t=30, b=10),
                                        font={'color': "#e6f1ff"}
                                    )
                                    
                                    st.plotly_chart(fig_gauge, use_container_width=True)
                                    st.caption(f"🧠 Cerebro: `{model_used.split('/')[-1]}`")
                                
                                with st.expander("🔍 Detalles Técnicos del Input"):
                                    detail_col1, detail_col2 = st.columns(2)
                                    
                                    with detail_col1:
                                        st.markdown("**📤 Respuesta API:**")
                                        st.json(result)
                                    
                                    with detail_col2:
                                        st.markdown("**📥 Vector de Entrada (primeros 10 valores):**")
                                        st.code(str(features_list[:10]))
                                        st.caption(f"Total features: {len(features_list)}")
                                        
                            else:
                                st.error(f"❌ Error en API ({response.status_code}): {response.text}")
                                
                        except requests.exceptions.ConnectionError:
                            st.error("❌ No se pudo conectar con la API.")
                            st.info("💡 Asegúrate de que el contenedor Docker esté corriendo: `podman-compose up`")
                    else:
                        st.warning("⚠️ No hay suficientes datos procesados para generar una ventana de 30 días.")

            except Exception as e:
                st.error(f"❌ Ocurrió un error inesperado: {str(e)}")

else:
    # === PANTALLA DE INICIO MEJORADA ===
    st.markdown("---")
    
    welcome_col1, welcome_col2 = st.columns([2, 1])
    
    with welcome_col1:
        st.markdown("""
        ### 👋 Bienvenido al Commodity AI
        
        Este sistema utiliza **redes neuronales LSTM/CNN** entrenadas con datos históricos 
        y variables macroeconómicas para predecir la dirección del precio de commodities.
        
        **Para comenzar:**
        1. Selecciona un activo en el panel lateral
        2. Pulsa **"ANALIZAR MERCADO"**
        3. Recibe predicciones en tiempo real
        """)
    
    with welcome_col2:
        st.markdown("""
        ### 📊 Activos Disponibles
        - 🥇 **Oro** (GC=F)
        - 🥈 **Plata** (SI=F)  
        - 🛢️ **Petróleo** (CL=F)
        """)
    
    # Preview de métricas vacías
    st.markdown("---")
    st.markdown("### 📈 Panel de Métricas")
    
    empty_cols = st.columns(4)
    for i, col in enumerate(empty_cols):
        with col:
            st.metric(
                label=["💰 Precio", "📐 Log Return", "📊 Volumen", "📈 Rango"][i],
                value="--",
                delta="Esperando análisis..."
            )
