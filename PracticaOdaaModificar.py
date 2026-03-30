import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import base64
from sklearn.metrics import mean_squared_error

# =============================================================================
# 1. CONFIGURACIÓN E IDENTIDAD INSTITUCIONAL UCV
# =============================================================================
st.set_page_config(
    page_title="Tesis UCV - Simulación Dinámica PID",
    page_icon="🛠️",
    layout="wide"
)

# Inicialización del estado de la sesión
if 'ejecutando' not in st.session_state:
    st.session_state.ejecutando = False

# Estilos CSS Profesionales (UCV Style)
st.markdown("""
    <style>
    /* Configuración Global */
    html, body, [data-testid="stAppViewContainer"] {
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' style='font-size: 24px;'><text y='20'>⚙️</text></svg>") 16 16, auto;
    }
    .main { background-color: #f8f9fa; }
    
    /* Encabezado con Animación */
    .header-container {
        background: linear-gradient(-45deg, #1a5276, #21618c, #154360, #1a5276);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 30px; border-radius: 15px; margin-bottom: 25px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2); color: white; text-align: center;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Paneles de Métricas */
    [data-testid="stMetricValue"] { font-size: 2rem; color: #1a5276; font-weight: bold; }
    div.stMetric {
        background-color: #ffffff; padding: 25px; border-radius: 15px;
        border-left: 10px solid #1a5276; box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }

    /* Botones Personalizados */
    .stButton>button {
        background-color: #1a5276; color: white; border-radius: 12px;
        font-weight: bold; height: 3.8em; width: 100%; transition: all 0.4s ease;
        border: none; text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #21618c; transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    div.stButton > button:first-child[kind="secondary"] {
        background-color: #943126; color: white;
    }

    /* Indicador de Flujo */
    .flow-indicator {
        background: #d4e6f1; border-left: 6px solid #2980b9;
        padding: 15px; border-radius: 8px; font-weight: bold;
        color: #1a5276; margin: 10px 0; animation: blink 1.5s infinite;
    }
    @keyframes blink { 50% { opacity: 0.5; } }
    </style>
    """, unsafe_allow_html=True)

# Manejo de Logotipos en Base64
def get_base64_img(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_ucv = get_base64_img("logo_ucv.png")
logo_eiq = get_base64_img("logoquimicaborde.png")

st.markdown(f"""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="width: 150px;">{f'<img src="data:image/png;base64,{logo_ucv}" width="110">' if logo_ucv else "<b>UCV</b>"}</div>
        <div>
            <h1 style="color: white !important; font-size: 2.5rem; margin-bottom: 5px;">Simulador Dinámico de Control de Nivel</h1>
            <p style="color: #d4e6f1 !important; font-size: 1.1rem; margin: 0;">Proyecto de Tesis | Escuela de Ingeniería Química - UCV</p>
        </div>
        <div style="width: 150px;">{f'<img src="data:image/png;base64,{logo_eiq}" width="150">' if logo_eiq else "<b>EIQ</b>"}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 2. MARCO TEÓRICO COMPLETO
# =============================================================================
with st.expander("📖 Fundamentos Físicos y Matemáticos (Marco Teórico)", expanded=False):
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### Balance de Masa")
        st.latex(r"A(h) \frac{dh}{dt} = Q_{in} - Q_{out} \pm Q_p")
        st.markdown("""
        * **Q_in**: Flujo de entrada (Variable de control $u$).
        * **Q_out**: Flujo de salida (Ley de Torricelli: $C_d \cdot a \cdot \sqrt{2gh}$).
        * **Q_p**: Flujo de perturbación (fugas o carga extra).
        """)
    with t2:
        st.markdown("### Algoritmo PID")
        st.latex(r"u(t) = K_p e(t) + K_i \int e(t)dt + K_d \frac{de(t)}{dt}")
        st.markdown("""
        * **Proporcional ($K_p$)**: Reacción al error actual.
        * **Integral ($K_i$)**: Eliminación del error en estado estacionario.
        * **Derivativo ($K_d$)**: Predicción del error futuro.
        """)

# =============================================================================
# 3. BARRA LATERAL: PARÁMETROS TÉCNICOS
# =============================================================================
st.sidebar.header("🕹️ Panel de Control")

with st.sidebar.container(border=True):
    op_tipo = st.sidebar.selectbox("🎯 Modo de Operación", ["Llenado", "Vaciado"])
    geom_tanque = st.sidebar.selectbox("📐 Geometría", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📏 Dimensiones del Equipo", expanded=True):
    r_max = st.number_input("Radio [m]", 0.5, 5.0, 1.0, 0.1)
    h_total = st.number_input("Altura Max [m]", 1.0, 10.0, 3.0, 0.5)
    sp_nivel = st.slider("Setpoint (Deseado) [m]", 0.1, float(h_total), 1.5)

with st.sidebar.expander("🌪️ Configuración de Falla"):
    p_activa = st.toggle("Activar Perturbación", value=True)
    p_mag = st.number_input("Caudal Fuga [m³/s]", -0.1, 0.1, -0.045, format="%.4f")
    p_t = st.slider("Tiempo de inicio [s]", 0, 600, 100)

with st.sidebar.expander("🧪 Sintonización PID"):
    kp = st.number_input("Ganancia Proporcional (Kp)", value=2.5)
    ki = st.number_input("Ganancia Integral (Ki)", value=0.4)
    kd = st.number_input("Ganancia Derivativa (Kd)", value=0.15)
    t_sim = st.select_slider("Duración [s]", options=[60, 120, 300, 600], value=300)

st.sidebar.markdown("---")
c_b1, c_b2 = st.sidebar.columns(2)
iniciar = c_b1.button("▶️ INICIAR")
reset = c_b2.button("🔄 RESET", type="secondary")

if reset:
    st.session_state.ejecutando = False
    st.rerun()

# =============================================================================
# 4. MOTOR DE CÁLCULO (MÉTODO DE EULER)
# =============================================================================
def calcular_paso(dt, h_act, sp, geom, r, hmax, qp_val, integral, error_prev):
    # Cálculo del Área Transversal según geometría
    if geom == "Cilíndrico":
        area = np.pi * (r**2)
    elif geom == "Cónico":
        area = np.pi * ((r/hmax) * max(h_act, 0.01))**2
    else: # Esférico
        area = np.pi * (2 * r * max(h_act, 0.01) - max(h_act, 0.01)**2)
    
    area = max(area, 0.01) # Evitar división por cero
    
    # Algoritmo PID
    error = sp - h_act
    integral += error * dt
    derivada = (error - error_prev) / dt
    
    u = (kp * error) + (ki * integral) + (kd * derivada)
    q_in = np.clip(u, 0, 0.6) # Saturación de la válvula
    
    # Salida por Torricelli
    q_out = 0.62 * 0.04 * np.sqrt(2 * 9.81 * h_act) if h_act > 0.01 else 0
    
    # Ecuación Diferencial
    dhdt = (q_in - q_out + q_p_inst) / area
    h_next = np.clip(h_act + dhdt * dt, 0, hmax)
    
    return h_next, q_in, error, integral

# =============================================================================
# 5. EJECUCIÓN Y RENDERIZADO DINÁMICO
# =============================================================================
if iniciar:
    st.session_state.ejecutando = True

if not st.session_state.ejecutando:
    st.info("👋 Bienvido. Configure los parámetros y presione Iniciar para ver la respuesta del sistema.")
    # Imagen de referencia si existe
    if os.path.exists("diagrama.png"): st.image("diagrama.png", use_container_width=True)
else:
    # Preparación de Columnas de Visualización
    col_vis, col_data = st.columns([2, 1])
    
    with col_vis:
        st.subheader("📡 Monitorización en Tiempo Real")
        ph_tanque = st.empty()
        ph_grafica_pid = st.empty()
        ph_grafica_u = st.empty()
    
    with col_data:
        st.subheader("📊 Variables Críticas")
        met_h = st.empty()
        met_e = st.empty()
        ph_flujo = st.empty()
        st.markdown("---")
        st.subheader("📋 Historial de Datos")
        ph_tabla = st.empty()
        ph_download = st.empty()

    # Inicialización de vectores
    dt = 1.0
    t_vec = np.arange(0, t_sim + dt, dt)
    h_hist, u_hist, sp_hist, e_hist = [], [], [], []
    
    h_actual = h_total if op_tipo == "Vaciado" else 0.1
    err_acum, err_pasado = 0.0, 0.0
    progreso = st.progress(0)

    # BUCLE DE SIMULACIÓN
    for i, t_actual in enumerate(t_vec):
        # Aplicación de perturbación
        q_p_inst = p_mag if (p_activa and t_actual >= p_t) else 0.0
        
        # Resolver paso
        h_actual, u_val, err_val, err_acum = calcular_paso(
            dt, h_actual, sp_nivel, geom_tanque, r_max, h_total, q_p_inst, err_acum, err_pasado
        )
        err_pasado = err_val
        
        # Guardar historial
        h_hist.append(h_actual); u_hist.append(u_val); sp_hist.append(sp_nivel); e_hist.append(err_val)
        
        # --- RENDERIZADO DE GRÁFICAS ---
        
        # 1. Dibujo del Tanque (Matplotlib)
        fig_tk, ax_tk = plt.subplots(figsize=(4, 5))
        ax_tk.set_xlim(-r_max*1.2, r_max*1.2)
        ax_tk.set_ylim(-0.2, h_total*1.1)
        if geom_tanque == "Cilíndrico":
            ax_tk.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, h_actual, color='#3498db', alpha=0.7))
            ax_tk.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='black', lw=3)
        elif geom_tanque == "Cónico":
            r_h = (r_max/h_total)*h_actual
            ax_tk.add_patch(plt.Polygon([[-r_h, h_actual], [r_h, h_actual], [0, 0]], color='#3498db', alpha=0.7))
            ax_tk.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='black', lw=3)
        else: # Esférico
            ax_tk.add_patch(plt.Circle((0, r_max), r_max, color='black', fill=False, lw=3))
            if h_actual > 0.01:
                ang = np.degrees(np.arccos(np.clip(1-(h_actual/r_max), -1, 1)))
                ax_tk.add_patch(plt.matplotlib.patches.Wedge((0, r_max), r_max, 270-ang, 270+ang, color='#3498db', alpha=0.7))
        
        ax_tk.axhline(sp_nivel, color='red', linestyle='--', label=f"Setpoint: {sp_nivel}m")
        ax_tk.axis('off')
        ph_tanque.pyplot(fig_tk)
        plt.close(fig_tk)

        # 2. Gráfica Respuesta Dinámica (PV vs SP)
        fig_pid, ax_pid = plt.subplots(figsize=(8, 3.5))
        ax_pid.plot(t_vec[:i+1], h_hist, color='#2980b9', lw=2, label="Nivel (PV)")
        ax_pid.plot(t_vec[:i+1], sp_hist, color='#c0392b', ls='--', label="Setpoint (SP)")
        ax_pid.set_ylabel("Altura [m]"); ax_pid.legend(); ax_pid.grid(alpha=0.3)
        ph_grafica_pid.pyplot(fig_pid)
        plt.close(fig_pid)

        # 3. Acción del Controlador (u)
        fig_u, ax_u = plt.subplots(figsize=(8, 2))
        ax_u.fill_between(t_vec[:i+1], u_hist, color='#e67e22', alpha=0.3)
        ax_u.plot(t_vec[:i+1], u_hist, color='#d35400', lw=1.5)
        ax_u.set_ylabel("Válvula [u]"); ax_u.set_xlabel("Tiempo [s]"); ax_u.grid(alpha=0.3)
        ph_grafica_u.pyplot(fig_u)
        plt.close(fig_u)

        # Actualización de Métricas y Tabla
        met_h.metric("Nivel Actual", f"{h_actual:.3f} m")
        met_e.metric("Error de Control", f"{err_val:.4f} m")
        if u_val > 0.01: ph_flujo.markdown("<div class='flow-indicator'>💧 FLUJO ENTRADA ACTIVO</div>", unsafe_allow_html=True)
        else: ph_flujo.empty()
        
        # Tabla resumen (últimos 5 datos)
        df_tmp = pd.DataFrame({"Tiempo": t_vec[:i+1], "Nivel": h_hist, "Error": e_hist, "u": u_hist})
        ph_tabla.dataframe(df_tmp.tail(5), use_container_width=True)
        
        progreso.progress((i+1)/len(t_vec))
        time.sleep(0.01)

    # FINALIZACIÓN
    st.success("🏁 Simulación finalizada satisfactoriamente.")
    st.balloons()
    
    # Análisis Final
    mse = mean_squared_error(sp_hist, h_hist)
    st.subheader("📝 Análisis de Desempeño")
    c_res1, c_res2, c_res3 = st.columns(3)
    c_res1.write(f"**Error Cuadrático Medio (MSE):** {mse:.6f}")
    c_res2.write(f"**Error Final:** {abs(e_hist[-1]):.5f} m")
    
    if abs(e_hist[-1]) < 0.05:
        c_res3.success("SISTEMA ESTABLE")
    else:
        c_res3.warning("SISTEMA CON OFFSET")

    # Botón de Descarga
    csv = df_tmp.to_csv(index=False).encode('utf-8')
    ph_download.download_button("📥 Descargar Reporte CSV", csv, "simulacion_ucv.csv", "text/csv")
