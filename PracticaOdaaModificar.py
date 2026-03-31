import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error
# =============================================================================
# 1. CONFIGURACIÓN E IDENTIDAD INSTITUCIONAL UCV
# =============================================================================
st.set_page_config(
    page_title="Tesis UCV - Simulación Dinámica",
    page_icon="🛠️",
    layout="wide"
)

# Inicialización del estado
if 'ejecutando' not in st.session_state:
    st.session_state.ejecutando = False

# Estilos CSS Unificados: Todo en un solo bloque corregido
st.markdown("""
    <style>
    /* 1. CONFIGURACIÓN GLOBAL Y CURSOR */
    html, body, [data-testid="stAppViewContainer"] {
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' style='font-size: 24px;'><text y='20'>⚙️</text></svg>") 16 16, auto;
    }
    button, [data-testid="stHeaderActionElements"] { cursor: pointer !important; }
    .main { background-color: #f4f7f9; }

    /* 2. ENCABEZADO DINÁMICO UCV */
    .header-container {
        background: linear-gradient(-45deg, #1a5276, #21618c, #154360, #1a5276);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 25px; border-radius: 15px; margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); color: white; text-align: center;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 3. INDICADOR DE FLUJO ACTIVO */
    .flow-indicator {
        background: #e8f4f8; border-left: 5px solid #3498db;
        padding: 10px; border-radius: 5px; display: flex;
        align-items: center; gap: 10px; font-weight: bold;
        color: #1a5276; animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; }
    }

    /* 4. MÉTRICAS Y BOTONES */
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1a5276; font-weight: bold; }
    div.stMetric {
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        border-left: 8px solid #1a5276; box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #1a5276; color: white; border-radius: 10px;
        font-weight: bold; height: 3.5em; width: 100%; transition: 0.3s;
    }
    div.stButton > button:first-child[kind="secondary"] {
        background-color: #943126; color: white; border: none;
    }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# ENCABEZADO INSTITUCIONAL CON FONDO (COLOCAR AQUÍ)
# =============================================================================
import base64

def get_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

logo_ucv_64 = get_base64("logo_ucv.png")
logo_eiq_64 = get_base64("logoquimicaborde.png")

st.markdown(f"""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="img-fluid" style="width: 120px;">
            {f'<img src="data:image/png;base64,{logo_ucv_64}" width="100">' if logo_ucv_64 else "UCV"}
        </div>
        <div>
            <h1 style="color: white !important; font-size: 2.2rem;">Práctica Virtual: Balance en estado no estacionario</h1>
            <p style="color: #d4e6f1 !important; margin: 0;">Escuela de Ingeniería Química | Facultad de Ingeniería - UCV</p>
        </div>
        <div class="img-fluid" style="width: 160px;">
            {f'<img src="data:image/png;base64,{logo_eiq_64}" width="150">' if logo_eiq_64 else "EIQ"}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 2. MARCO TEÓRICO: BALANCE DE MASA Y TORRICELLI
# =============================================================================
# =============================================================================
# 2. MARCO TEÓRICO INTEGRADO: FÍSICA Y CONTROL
# =============================================================================
col_teoria1, col_teoria2,col_teoria3 = st.columns(3)

with col_teoria1:
    with st.expander("Fundamento teórico: Ecuaciones de Conservación y Descarga", expanded=False):
        st.markdown(r"""
        La dinámica del sistema se describe mediante el **Balance Global de Masa** para un volumen de control con densidad constante ($\rho$):
        
        $$ \frac{dV}{dt} = Q_{in} - Q_{out} \pm Q_{p} $$
        
        Considerando que el volumen es función del nivel ($V = \int A(h)dh$), aplicamos la regla de la cadena para obtener la ecuación general de vaciado/llenado válida para **cualquier área transversal $A(h)$**:
        
        $$ A(h) \frac{dh}{dt} = Q_{in} - (C_d \cdot a \cdot \sqrt{2gh}) \pm Q_{p} $$
        
        Donde:
        * **$A(h)$**: Área de la sección transversal en función de la altura (m²).
        * **$Q_{in}$**: Flujo de entrada controlado (m³/s).
        * **$Q_{out}$**: Flujo de salida basado en la **Ley de Torricelli** (m³/s).
        * **$C_d$**: Coeficiente de descarga (adimensional).
        * **$a$**: Área del orificio de salida (m²).
        * **$Q_{p}$**: Flujo de perturbación o falla (m³/s).
        """)

with col_teoria2:
    with st.expander("Teoría: Estrategia de control PID", expanded=False):
        st.markdown(r"""
        El "cerebro" de la simulación es un controlador **Proporcional-Integral-Derivativo (PID)**, cuya acción de control $u(t)$ busca minimizar el error ($e = SP - h$):
        
        $$ u(t) = K_p e(t) + K_i \int_{0}^{t} e(\tau) d\tau + K_d \frac{de(t)}{dt} $$
        
        **Funciones de los parámetros sintonizables:**
        * **$K_p$ (Proporcional):** Proporciona una respuesta inmediata al error actual.
        * **$K_i$ (Integral):** Elimina el error residual (offset) acumulando desviaciones pasadas; es vital para el rechazo de perturbaciones ($Q_p$).
        * **$K_d$ (Derivativo):** Anticipa el comportamiento futuro del error para evitar sobrepicos y estabilizar la respuesta.
        
        En este simulador, las ecuaciones se resuelven numéricamente mediante el **Método de Euler** con un paso de tiempo $\Delta t = 1.0$ s.
        """)
with col_teoria3:
    with st.expander("📊 Criterios de Desempeño (IAE/ITAE)", expanded=False):
        st.markdown(r"""
        Para evaluar la eficiencia del control, se utilizan métricas integrales del error $e(t) = SP - PV$:

        1. **IAE (Integral del Error Absoluto):**
        $$IAE = \int_{0}^{t} |e(t)| dt$$
        Mide el rendimiento acumulado. Es ideal para evaluar la respuesta general del sistema.

        2. **ITAE (Integral del Tiempo por el Error Absoluto):**
        $$ITAE = \int_{0}^{t} t \cdot |e(t)| dt$$
        **Penaliza errores que duran mucho tiempo.** Es el criterio más estricto en tesis de control porque asegura que el sistema se estabilice rápido.
        """)

# =============================================================================
# 3. BARRA LATERAL: PARÁMETROS TÉCNICOS
# =============================================================================
st.sidebar.header("⚙️ Configuración del Sistema")

with st.sidebar.container(border=True):
    op_tipo = st.sidebar.selectbox("🎯 Operación Principal", ["Llenado", "Vaciado"])
    geom_tanque = st.sidebar.selectbox("📐 Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📏 Especificaciones del Tanque", expanded=True):
    r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
    h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
    h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
    sp_nivel = st.slider("Consigna de Nivel (Setpoint) [m]", 0.1, float(h_total), float(h_total/2))

with st.sidebar.expander("🌪️ Escenario de Perturbación ($Q_p$)"):
    p_activa = st.toggle("Simular Falla/Fuga Externas", value=True)
    p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.045, format="%.4f") if p_activa else 0.0
    p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 80) if p_activa else 0

with st.sidebar.expander("Parámetros del Controlador PID"):
    c1, c2, c3 = st.columns(3)
    kp_val = c1.number_input("Kp", value=2.6)
    ki_val = c2.number_input("Ki", value=0.5)
    kd_val = c3.number_input("Kd", value=0.1)
    tiempo_ensayo = st.sidebar.slider("Tiempo de simulación [s]", 60, 600, 300)

st.sidebar.markdown("---")
# Botones de Control
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    iniciar_sim = st.button("▶️ Iniciar", use_container_width=True)
with col_btn2:
    btn_reset = st.button("🔄 Reset", use_container_width=True, type="secondary")

if btn_reset:
    st.session_state.ejecutando = False
    st.rerun()

# =============================================================================
# 4. LÓGICA DE CÁLCULO: MÉTODO DE EULER
# =============================================================================
def resolver_sistema(dt, h_prev, sp, geom, r, h_t, q_p_val, e_sum, e_prev):
    if geom == "Cilíndrico":
        area_h = np.pi * (r**2)
    elif geom == "Cónico":
        area_h = np.pi * ((r/h_t) * max(h_prev, 0.01))**2
    else: # Esférico
        area_h = np.pi * (2 * r * max(h_prev, 0.01) - max(h_prev, 0.01)**2)
    
    area_h = max(area_h, 0.01) 

    err = sp - h_prev
    e_sum += err * dt
    e_der = (err - e_prev) / dt
    u_control = (kp_val * err) + (ki_val * e_sum) + (kd_val * e_der)
    
    q_entrada = np.clip(u_control, 0, 0.6)
    q_salida = 0.61 * 0.04 * np.sqrt(2 * 9.81 * h_prev) if h_prev > 0.005 else 0
    
    dh_dt = (q_entrada - q_salida + q_p_val) / area_h
    h_next = np.clip(h_prev + dh_dt * dt, 0, h_t)
    
    return h_next, q_entrada, err, e_sum, err
# =============================================================================
# 5 y 6. LÓGICA DE VISUALIZACIÓN Y SIMULACIÓN UNIFICADA (CORREGIDA)
# =============================================================================

if iniciar_sim:
    st.session_state.ejecutando = True

# Determinamos si el expander del diagrama debe estar abierto
estado_expander = not st.session_state.ejecutando

# --- PESTAÑA DEL DIAGRAMA ---
with st.expander("Diagrama del Proceso", expanded=estado_expander):
    col_img = st.columns([1, 5, 1])[1]
    with col_img:
        if os.path.exists("Captura de pantalla 2026-03-29 163125.png"):
            st.image("Captura de pantalla 2026-03-29 163125.png", use_container_width=True)
        else:
            st.info("📍 El diagrama del sistema se mostrará aquí.")

# --- LÓGICA DE CONTROL DE ESTADOS ---
if not st.session_state.ejecutando:
    st.info("💡 Ajuste los parámetros en la barra lateral y presione 'Iniciar' para comenzar.")
else:
    # 1. Dashboard de columnas
    col_graf, col_met = st.columns([2, 1])

    with col_graf:
        st.subheader("🖥️ Monitor del Proceso")
        placeholder_tanque = st.empty()
        st.subheader("📊 Tendencia Temporal")
        placeholder_grafico = st.empty()
        st.subheader("⚙️ Acción del Controlador")
        placeholder_u = st.empty()

    with col_met:
        st.subheader("📊 Métricas de Control")
        # Forzamos la aparición de las tarjetas con un valor inicial
        placeholder_iae = st.empty()
        placeholder_itae = st.empty()
        placeholder_iae.metric("IAE (Error Acumulado)", "0.00")
        placeholder_itae.metric("ITAE (Criterio Tesis)", "0.00")
        
        st.markdown("---")
        m_h = st.empty()
        m_e = st.empty()
        m_h.metric("Nivel PV [m]", "0.000")
        m_e.metric("Error [m]", "0.000")
        
        st.markdown("---")
        area_descarga = st.empty()

    # 2. Preparación de datos
    status_placeholder = st.empty()
    dt = 1.0 
    vector_t = np.arange(0, tiempo_ensayo, dt)
    h_log, u_log, sp_log, e_log = [], [], [], []
    h_corrida = h_total if op_tipo == "Vaciado" else 0.05
    err_int, err_pasado = 0, 0
    iae_acumulado = 0
    itae_acumulado = 0
    
    barra_p = st.progress(0)

    # 3. Bucle de Simulación
    for i, t_act in enumerate(vector_t):
        status_placeholder.markdown("<div class='flow-indicator'>💧 PROCESANDO...</div>", unsafe_allow_html=True)
        
        q_p_inst = p_magnitud if (p_activa and t_act >= p_tiempo) else 0.0
        
        h_corrida, u_inst, e_inst, err_int, err_pasado = resolver_sistema(
            dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst, err_int, err_pasado
        )
        
        # Cálculo de métricas integrales
        iae_acumulado += abs(e_inst) * dt
        itae_acumulado += (t_act * abs(e_inst)) * dt
        
        h_log.append(h_corrida)
        u_log.append(u_inst)
        sp_log.append(sp_nivel) 
        e_log.append(e_inst)
        
        if i % 2 == 0:
            # ACTUALIZACIÓN DE MÉTRICAS (AQUÍ ESTABA EL ERROR)
            placeholder_iae.metric("IAE (Error Acumulado)", f"{iae_acumulado:.2f}")
            placeholder_itae.metric("ITAE (Criterio Tesis)", f"{itae_acumulado:.2f}")
            m_h.metric("Nivel PV [m]", f"{h_corrida:.3f}")
            m_e.metric("Error [m]", f"{e_inst:.4f}")

            # Gráficos (Tanque)
            fig_t, ax_t = plt.subplots(figsize=(5, 4))
            ax_t.set_xlim(-r_max*1.2, r_max*1.2)
            ax_t.set_ylim(-0.1, h_total*1.1)
            ax_t.set_xticks([]); ax_t.set_ylabel("Nivel [m]")
            h_vis = h_corrida + (0.02 * np.sin(t_act * 4) if u_inst > 0.05 else 0)
            
            if geom_tanque == "Cilíndrico":
                ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, h_vis, color='#3498db', alpha=0.6))
                ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=3)
            elif geom_tanque == "Cónico":
                r_h = (r_max / h_total) * h_vis
                ax_t.add_patch(plt.Polygon([[-r_h, h_vis], [r_h, h_vis], [0, 0]], color='#3498db', alpha=0.6))
                ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=3)
            elif geom_tanque == "Esférico":
                ax_t.add_patch(plt.Circle((0, r_max), r_max, color='#2c3e50', fill=False, lw=3))
                if h_vis > 0:
                    ang_w = np.degrees(np.arccos(np.clip(1 - (h_vis/r_max), -1, 1)))
                    ax_t.add_patch(plt.matplotlib.patches.Wedge((0, r_max), r_max, 270-ang_w, 270+ang_w, color='#3498db', alpha=0.6))

            ax_t.axhline(y=sp_nivel, color='red', ls='--', label=f"SP: {sp_nivel}m")
            placeholder_tanque.pyplot(fig_t)
            plt.close(fig_t)

            # Gráfica PV vs SP
            fig_tr, ax_tr = plt.subplots(figsize=(8, 3.5))
            ax_tr.plot(vector_t[:len(h_log)], h_log, color='#2980b9', lw=2.5, label='Nivel PV')
            ax_tr.plot(vector_t[:len(sp_log)], sp_log, color='#c0392b', ls='--', lw=2, label='Setpoint SP')
            ax_tr.grid(True, alpha=0.3)
            ax_tr.legend(loc='upper right')
            placeholder_grafico.pyplot(fig_tr)
            plt.close(fig_tr)

            # Acción u
            fig_u, ax_u = plt.subplots(figsize=(8, 2.5))
            ax_u.step(vector_t[:len(u_log)], u_log, color='#e67e22', where='post')
            ax_u.set_ylim(0, 0.7)
            placeholder_u.pyplot(fig_u)
            plt.close(fig_u)
        
        time.sleep(0.01) 
        barra_p.progress((i+1)/len(vector_t))

    # --- RESULTADOS FINALES ---
    status_placeholder.empty()
    st.success(f"✅ Simulación del Tanque {geom_tanque} completada.")
    st.balloons()
    
     # Tabla resumen y descarga
    df_final = pd.DataFrame({"Tiempo [s]": vector_t, "Nivel [m]": h_log, "u [m3/s]": u_log})
    st.dataframe(df_final.tail(10).style.format("{:.4f}"), use_container_width=)
    
    df_final = pd.DataFrame({
        "Tiempo [s]": vector_t, 
        "Nivel [m]": h_log, 
        "u [m3/s]": u_log,
        "Error [m]": e_log
    })
    
    st.subheader("📝 Resumen de Estabilidad")
    err_f = abs(sp_nivel - h_log[-1])
    c1, c2, c3 = st.columns(3)
    c1.metric("IAE Final", f"{iae_acumulado:.2f}")
    c2.metric("ITAE Final", f"{itae_acumulado:.2f}")
    c3.metric("Error Residual", f"{err_f:.4f} m")

    area_descarga.download_button(
        "📥 Descargar Datos  (CSV)", 
        df_final.to_csv(index=False), 
        "resultados_simulacion_ucv.csv",
        use_container_width=True
    )

    # Análisis de Estabilidad
    st.markdown("---")
    st.subheader("Análisis de Estabilidad")
    error_final = abs(sp_nivel - h_log[-1])
    if error_final < 0.05:
        st.success(f"✅ Sistema Estabilizado en {h_log[-1]:.3f} m.")
    else:
        st.warning(f"⚠️ Desviación de {error_final:.3f} m. Ajuste Kp, Ki o Kd.")
