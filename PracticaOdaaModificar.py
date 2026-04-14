import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import base64
from sklearn.metrics import mean_squared_error

# =============================================================================
# --- FUNCIONES DE CÁLCULO ---
# =============================================================================

def calcular_pid_adaptativo(geom, r_max, h_total):
    """
    Calcula parámetros PID según geometría del tanque.
    NOTA: Esta función es la versión simple (no usa cd/area_orificio)
    """
    import math
    area_max = math.pi * (r_max ** 2)
    
    if geom == "Cilíndrico":
        kp = area_max * 2.5
        ki = kp / 20.0
        kd = kp * 0.1
    elif geom == "Cónico":
        kp = (area_max / 3.0) * 1.5
        ki = kp / 15.0
        kd = kp * 0.05
    else:  # Esférico
        kp = (area_max * 0.6) * 2.0
        ki = kp / 18.0
        kd = kp * 0.2
    
    return round(kp, 2), round(ki, 3), round(kd, 3)


def sintonizar_controlador_robusto(geom, r, h_t, cd_calculado, area_ori, op_tipo="Llenado"):
    """
    Sintonización robusta del PID para rechazo de perturbaciones.
    Basada en métodos Ziegler-Nichols adaptados para tanques.
    """
    import numpy as np
    
    # Calcular área transversal característica
    if geom == "Cilíndrico":
        area_t = np.pi * (r**2)
        tau = area_t / (cd_calculado * area_ori)  # Constante de tiempo
    elif geom == "Cónico":
        area_t = np.pi * (r/2)**2
        tau = area_t / (cd_calculado * area_ori * 0.8)
    else:  # Esférico
        area_t = (2/3) * np.pi * (r**2)
        tau = area_t / (cd_calculado * area_ori * 0.6)
    
    # Ganancia crítica (Ziegler-Nichols)
    Kc = (2 * area_t) / (cd_calculado * area_ori * np.sqrt(2 * 9.81 * h_t/2))
    Kc = np.clip(Kc, 1.0, 30.0)
    
    # Parámetros PID robustos para rechazo de perturbaciones
    if op_tipo == "Llenado":
        kp = Kc * 0.6
        ki = kp / (tau * 0.5)  # Integral fuerte
        kd = kp * tau * 0.125
    else:
        kp = Kc * 0.5
        ki = kp / (tau * 0.6)
        kd = kp * tau * 0.1
    
    # Ajustes adicionales para robustez
    kp = np.clip(kp, 3.0, 25.0)
    ki = np.clip(ki, 0.3, 5.0)
    kd = np.clip(kd, 0.05, 2.0)
    
    return round(kp, 2), round(ki, 3), round(kd, 3)


def calcular_cd_inteligente(df_usr, r, h_t, geom, area_ori):
    """Calcula el Coeficiente de Descarga (Cd) usando el balance de masa."""
    import numpy as np
    import pandas as pd
    
    df = pd.DataFrame(df_usr) if isinstance(df_usr, list) else df_usr
    
    if len(df) < 2:
        return 0.61
    
    try:
        t1, t2 = df["Tiempo (s)"].iloc[0], df["Tiempo (s)"].iloc[1]
        h1, h2 = df["Nivel Medido (m)"].iloc[0], df["Nivel Medido (m)"].iloc[1]
        dt = abs(t2 - t1)
        
        if dt == 0:
            return 0.61

        if geom == "Cilíndrico":
            v1, v2 = np.pi*(r**2)*h1, np.pi*(r**2)*h2
        elif geom == "Cónico":
            v1 = (1/3)*np.pi*((r/h_t)*h1)**2*h1
            v2 = (1/3)*np.pi*((r/h_t)*h2)**2*h2
        else:  # Esférico
            v1 = (np.pi*(h1**2)/3)*(3*r-h1)
            v2 = (np.pi*(h2**2)/3)*(3*r-h2)

        q_real = abs(v1 - v2) / dt
        h_prom = (h1 + h2) / 2
        q_teorico = area_ori * np.sqrt(2 * 9.81 * max(h_prom, 0.001))
        
        cd_result = q_real / q_teorico if q_teorico > 0 else 0.61
        return float(np.clip(cd_result, 0.4, 1.0))
    except:
        return 0.61


def resolver_sistema_robusto(dt, h_prev, sp, geom, r, h_t, q_p_val, e_sum, e_prev, 
                             modo_op, cd_val, kp, ki, kd, d_pulgadas):
    """
    Resuelve la dinámica del sistema con Anti-Windup.
    """
    import numpy as np
    
    # Calcular área transversal
    if geom == "Cilíndrico":
        area_h = np.pi * (r**2)
    elif geom == "Cónico":
        radio_actual = (r / h_t) * max(h_prev, 0.01)
        area_h = np.pi * (radio_actual**2)
    else:  # Esférico
        area_h = np.pi * (2 * r * max(h_prev, 0.01) - max(h_prev, 0.01)**2)
    
    area_h = max(area_h, 0.01)

    # Cálculo del error
    err = sp - h_prev
    
    # Área del orificio
    a_o = np.pi * ((d_pulgadas * 0.0254) / 2)**2
    
    # --- ANTI-WINDUP ---
    q_max = 2.0
    
    # Calcular acción de control sin límites
    u_sin_limite = (kp * err) + (ki * e_sum) + (kd * (err - e_prev) / dt if dt > 0 else 0)
    
    if modo_op == "Llenado":
        if (u_sin_limite > q_max and err > 0) or (u_sin_limite < 0 and err < 0):
            e_sum += err * dt * 0.1  # Back-calculation
        else:
            e_sum += err * dt
    else:
        if (u_sin_limite > q_max and err > 0) or (u_sin_limite < 0 and err < 0):
            e_sum += err * dt * 0.1
        else:
            e_sum += err * dt
    
    e_sum = np.clip(e_sum, -10.0, 10.0)
    
    # Derivativo limitado
    e_der = (err - e_prev) / dt if dt > 0 else 0
    e_der = np.clip(e_der, -5.0, 5.0)
    
    u_control = (kp * err) + (ki * e_sum) + (kd * e_der)

    # Lógica de operación
    if modo_op == "Llenado":
        q_entrada = np.clip(u_control, 0, q_max)
        q_salida = cd_val * a_o * np.sqrt(2 * 9.81 * max(h_prev, 0.005)) if h_prev > 0.005 else 0
        dh_dt = (q_entrada + q_p_val - q_salida) / area_h
        u_graficar = q_entrada
    else:  # Vaciado
        q_entrada = q_p_val
        q_salida = np.clip(u_control, 0, q_max)
        dh_dt = (q_entrada - q_salida) / area_h
        u_graficar = q_salida
    
    h_next = np.clip(h_prev + dh_dt * dt, 0, h_t)
    return h_next, u_graficar, err, e_sum, err


def get_base64(path):
    """Convierte una imagen a base64 para incrustar en HTML"""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


# =============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Lab Virtual - Simulación Dinámica",
    page_icon="🧪",
    layout="wide"
)

# Inicialización del estado
if 'ejecutando' not in st.session_state:
    st.session_state.ejecutando = False


# =============================================================================
# 2. ESTILOS CSS (CORREGIDOS)
# =============================================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f4f7f9 !important;
}

button, a, [data-testid="stHeaderActionElements"], .stSlider {
    cursor: pointer !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a5276 0%, #154360 100%) !important;
    border-right: 4px solid #f1c40f !important;
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span {
    color: #f0f4f8 !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f5f9fc 100%) !important;
    border-left: 5px solid #1a5276 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #154360 !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #1a5276, #2471a3) !important;
    color: white !important;
    border-radius: 25px !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(90deg, #7b241c, #943126) !important;
    color: white !important;
    border-radius: 25px !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1a5276, #3498db, #1a5276) !important;
    animation: pulso_azul 2s ease-in-out infinite !important;
}

@keyframes pulso_azul {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
}

div[data-baseweb="slider"] > div > div > div {
    background-color: #f39c12 !important;
}

.header-container {
    background: linear-gradient(135deg, #0d3251 0%, #1a5276 50%, #1f618d 100%);
    border-radius: 20px;
    padding: 20px 25px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. ENCABEZADO INSTITUCIONAL
# =============================================================================
logo_ucv_64 = get_base64("logo_ucv.png")
logo_eiq_64 = get_base64("logoquimicaborde.png")

st.markdown(f"""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="width: 120px;">
            {f'<img src="data:image/png;base64,{logo_ucv_64}" width="100">' if logo_ucv_64 else "UCV"}
        </div>
        <div>
            <h1 style="color: white !important; font-size: 2.2rem;">Práctica Virtual: Balance en estado no estacionario</h1>
            <p style="color: #d4e6f1 !important;">Escuela de Ingeniería Química | Facultad de Ingeniería - UCV</p>
        </div>
        <div style="width: 160px;">
            {f'<img src="data:image/png;base64,{logo_eiq_64}" width="150">' if logo_eiq_64 else "EIQ"}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# 4. MARCO TEÓRICO
# =============================================================================
col_teoria1, col_teoria2, col_teoria3 = st.columns(3)

with col_teoria1:
    with st.expander("📐 Fundamento Teórico", expanded=False):
        st.markdown(r"""
        **Balance Global de Masa:**
        $$ \frac{dV}{dt} = Q_{in} - Q_{out} \pm Q_{p} $$
        
        **Ecuación general:**
        $$ A(h) \frac{dh}{dt} = Q_{in} - (C_d \cdot a \cdot \sqrt{2gh}) \pm Q_{p} $$
        """)

with col_teoria2:
    with st.expander("🎛️ Control PID Robusto", expanded=False):
        st.markdown(r"""
        $$ u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt} $$
        
        **Características:**
        - Anti-Windup anti-saturación
        - Sintonía Ziegler-Nichols adaptada
        - Límites en acción derivativa
        """)

with col_teoria3:
    with st.expander("📊 Métricas de Desempeño", expanded=False):
        st.markdown(r"""
        **IAE:** $$IAE = \int_{0}^{t} |e(t)| dt$$
        
        **ITAE:** $$ITAE = \int_{0}^{t} t \cdot |e(t)| dt$$
        """)


# =============================================================================
# 5. BARRA LATERAL - PARÁMETROS
# =============================================================================
st.sidebar.header("⚙️ Configuración del Sistema")

with st.sidebar.container(border=True):
    op_tipo = st.sidebar.selectbox("Operación Principal", ["Llenado", "Vaciado"])
    geom_tanque = st.sidebar.selectbox("Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📏 Especificaciones del Tanque", expanded=True):
    r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
    h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
    h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
    sp_nivel = st.slider("Consigna (Setpoint) [m]", 0.1, float(h_total), float(h_total/2))

with st.sidebar.expander("💧 Dimensiones de Salida", expanded=True):
    d_pulgadas = st.number_input("Diámetro del Orificio (pulgadas)", value=1.0, min_value=0.1, step=0.1)
    d_metros = d_pulgadas * 0.0254
    area_orificio = np.pi * (d_metros / 2)**2
    st.caption(f"Área calculada: {area_orificio:.6f} m²")

with st.sidebar.expander("🛡️ Escenario de Perturbación", expanded=True):
    p_activa = st.toggle("Simular Falla/Fuga", value=True)
    
    if p_activa:
        p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.045, format="%.4f")
        p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 80)
        modo_estres = st.toggle("🔥 Modo Estrés", help="La perturbación cambia según el nivel")
    else:
        p_magnitud = 0.0
        p_tiempo = 0
        modo_estres = False

with st.sidebar.expander("🎛️ Controlador PID", expanded=True):
    kp_sug, ki_sug, kd_sug = calcular_pid_adaptativo(geom_tanque, r_max, h_total)
    
    modo_auto = st.checkbox("🎯 Modo Robusto (Auto-sintonía)", value=True)
    
    if modo_auto:
        st.success("💡 Usando sintonización robusta")
        kp_val = st.number_input("Kp", value=kp_sug, key="kp_asist")
        ki_val = st.number_input("Ki", value=ki_sug, format="%.3f", key="ki_asist")
        kd_val = st.number_input("Kd", value=kd_sug, format="%.3f", key="kd_asist")
    else:
        st.info("✍️ Modo Manual")
        kp_val = st.number_input("Kp", value=7.5, step=0.5, key="kp_man")
        ki_val = st.number_input("Ki", value=1.2, step=0.1, format="%.3f", key="ki_man")
        kd_val = st.number_input("Kd", value=0.4, step=0.1, format="%.3f", key="kd_man")
    
    tiempo_ensayo = st.slider("Tiempo de simulación [s]", 60, 600, 300)

with st.sidebar.expander("📊 Datos Experimentales", expanded=True):
    st.caption("Ingresar nivel en **centímetros (cm)**")
    
    df_exp_default = pd.DataFrame({
        "Tiempo (s)": [0, 60, 120, 180, 240, 300],
        "Nivel Medido (cm)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    datos_usr = st.data_editor(df_exp_default, num_rows="dynamic")
    mostrar_ref = st.checkbox("Mostrar datos experimentales", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Biblioteca")

with st.sidebar.container(border=True):
    if os.path.exists("Guia_Practica_UCV.pdf"):
        with open("Guia_Practica_UCV.pdf", "rb") as f:
            st.sidebar.download_button(
                label="📥 Descargar Guía (PDF)",
                data=f,
                file_name="Guia_Practica_EIQ_UCV.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.sidebar.warning("⚠️ Guía no encontrada")

st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    iniciar_sim = st.button("▶️ Iniciar", use_container_width=True, type="primary")
with col_btn2:
    btn_reset = st.button("🔄 Reset", use_container_width=True, type="secondary")

if btn_reset:
    st.session_state.ejecutando = False
    st.rerun()


# =============================================================================
# 6. DIAGRAMA DEL PROCESO
# =============================================================================
with st.expander("📐 Diagrama del Proceso", expanded=not st.session_state.get('ejecutando', False)):
    col_img = st.columns([1, 5, 1])[1]
    with col_img:
        if os.path.exists("Captura de pantalla 2026-03-29 163125.png"):
            st.image("Captura de pantalla 2026-03-29 163125.png", use_container_width=True)
        else:
            st.info("📍 El diagrama del sistema se mostrará aquí.")


# =============================================================================
# 7. INICIALIZACIÓN DE SIMULACIÓN
# =============================================================================
if iniciar_sim:
    st.session_state.ejecutando = True
    st.session_state['error_acumulado'] = 0.0
    st.session_state['ultimo_error'] = 0.0
    
    try:
        if not isinstance(datos_usr, pd.DataFrame):
            datos_usr = pd.DataFrame(datos_usr)
        
        if modo_auto:
            if "Nivel Medido (cm)" in datos_usr.columns and not datos_usr["Nivel Medido (cm)"].isnull().all():
                df_calib = datos_usr.copy()
                df_calib["Nivel Medido (m)"] = df_calib["Nivel Medido (cm)"] / 100
                cd_calc = calcular_cd_inteligente(
                    df_calib[["Tiempo (s)", "Nivel Medido (m)"]], 
                    r_max, h_total, geom_tanque, area_orificio
                )
                kp_a, ki_a, kd_a = sintonizar_controlador_robusto(
                    geom_tanque, r_max, h_total, cd_calc, area_orificio, op_tipo
                )
                
                st.session_state['kp_ejecucion'] = kp_a
                st.session_state['ki_ejecucion'] = ki_a
                st.session_state['kd_ejecucion'] = kd_a
                st.session_state['cd_final'] = cd_calc
                st.toast(f"🎯 Robusto: Cd={cd_calc:.2f} | Kp={kp_a} | Ki={ki_a}")
            else:
                st.session_state['kp_ejecucion'] = 8.0
                st.session_state['ki_ejecucion'] = 1.5
                st.session_state['kd_ejecucion'] = 0.5
                st.session_state['cd_final'] = 0.61
                st.info("💡 Usando valores robustos por defecto")
        else:
            st.session_state['kp_ejecucion'] = kp_val
            st.session_state['ki_ejecucion'] = ki_val
            st.session_state['kd_ejecucion'] = kd_val
            st.session_state['cd_final'] = 0.61
    except Exception as e:
        st.session_state['kp_ejecucion'] = 8.0
        st.session_state['ki_ejecucion'] = 1.5
        st.session_state['kd_ejecucion'] = 0.5
        st.session_state['cd_final'] = 0.61
        st.warning(f"⚠️ Usando valores de emergencia")


# =============================================================================
# 8. SIMULACIÓN PRINCIPAL
# =============================================================================
if not st.session_state.ejecutando:
    st.info("💡 Ajuste los parámetros y presione 'Iniciar'")
else:
    col_graf, col_met = st.columns([2, 1])

    with col_graf:
        st.subheader("📊 Monitor del Proceso")
        placeholder_tanque = st.empty()
        st.subheader("📈 Tendencia Temporal")
        placeholder_grafico = st.empty()
        st.subheader("⚙️ Acción del Controlador")
        placeholder_u = st.empty()
        st.subheader("🔧 Estado de la Válvula")
        placeholder_valvula = st.empty()
        st.subheader("📉 Comparativa vs Datos")
        placeholder_comparativa = st.empty()

    with col_met:
        st.subheader("📊 Métricas de Control")
        
        kp_show = st.session_state.get('kp_ejecucion', 8.0)
        ki_show = st.session_state.get('ki_ejecucion', 1.5)
        cd_show = st.session_state.get('cd_final', 0.61)
        
        st.metric("Kp (Robusto)", f"{kp_show:.2f}")
        st.metric("Ki (Robusto)", f"{ki_show:.3f}")
        st.metric("Cd (Descarga)", f"{cd_show:.3f}")
        st.markdown("---")
        
        placeholder_iae = st.empty()
        placeholder_itae = st.empty()
        placeholder_iae.metric("IAE", "0.00")
        placeholder_itae.metric("ITAE", "0.00")
        st.markdown("---")
        
        m_h = st.empty()
        m_e = st.empty()
        m_h.metric("Nivel PV [m]", "0.000")
        m_e.metric("Error [m]", "0.000")

    # Preparación de la simulación
    status_placeholder = st.empty()
    dt = 1.0
    vector_t = np.arange(0, tiempo_ensayo, dt)
    h_log, u_log, e_log = [], [], []

    if op_tipo == "Llenado":
        h_corrida = 0.0
    else:
        h_corrida = h_total * 0.9
    
    valor_presente = h_corrida
    error_presente = 0.0
    err_int, err_pasado = 0.0, 0.0
    iae_acumulado, itae_acumulado = 0.0, 0.0
    
    # Procesar datos experimentales
    if not isinstance(datos_usr, pd.DataFrame):
        datos_usr = pd.DataFrame(datos_usr)
    
    if "Nivel Medido (cm)" in datos_usr.columns and len(datos_usr) > 0:
        t_exp = datos_usr["Tiempo (s)"].values
        h_exp = [val / 100 for val in datos_usr["Nivel Medido (cm)"].values]
        tiene_datos_exp = True
    else:
        t_exp = []
        h_exp = []
        tiene_datos_exp = False
    
    barra_p = st.progress(0)
    cd_para_simular = st.session_state.get('cd_final', 0.61)
    
    # Bucle de simulación
    for i, t_act in enumerate(vector_t):
        status_placeholder.markdown("💧 **SIMULANDO...**")
        
        # Lógica de perturbación
        if p_activa and t_act >= p_tiempo:
            if modo_estres:
                factor = 1.5 if valor_presente < sp_nivel else 0.5
                q_p_inst = p_magnitud * factor
            else:
                q_p_inst = p_magnitud
        else:
            q_p_inst = 0.0
        
        k_p = st.session_state.get('kp_ejecucion', 8.0)
        k_i = st.session_state.get('ki_ejecucion', 1.5)
        k_d = st.session_state.get('kd_ejecucion', 0.5)
        
        h_corrida, u_inst, e_inst, err_int, err_pasado = resolver_sistema_robusto(
            dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst,
            err_int, err_pasado, op_tipo, cd_para_simular,
            k_p, k_i, k_d, d_pulgadas
        )
        
        valor_presente = h_corrida
        error_presente = e_inst
        iae_acumulado += abs(e_inst) * dt
        itae_acumulado += (t_act * abs(e_inst)) * dt
        
        h_log.append(h_corrida)
        u_log.append(u_inst)
        e_log.append(e_inst)
        
        # Actualizar métricas
        m_h.metric("Nivel PV [m]", f"{valor_presente:.3f}")
        m_e.metric("Error [m]", f"{error_presente:.4f}")
        placeholder_iae.metric("IAE", f"{iae_acumulado:.2f}")
        placeholder_itae.metric("ITAE", f"{itae_acumulado:.2f}")
        
        # Visualización simplificada del tanque
        fig_t, ax_t = plt.subplots(figsize=(7, 5))
        ax_t.set_axis_off()
        ax_t.set_xlim(-r_max*3, r_max*3)
        ax_t.set_ylim(-0.8, h_total*1.3)
        
        color_agua = '#27ae60' if abs(e_inst) < 0.05 else '#e74c3c'
        
        if geom_tanque == "Cilíndrico":
            ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, valor_presente, color=color_agua, alpha=0.6))
            ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=4)
        elif geom_tanque == "Cónico":
            ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=4)
            r_act = (r_max / h_total) * max(valor_presente, 0.01)
            ax_t.add_patch(plt.Polygon([[-r_act, valor_presente], [r_act, valor_presente], [0, 0]], color=color_agua, alpha=0.6))
        else:  # Esférico
            ax_t.add_patch(plt.Circle((0, r_max), r_max, color=color_agua, alpha=0.6))
            ax_t.add_patch(plt.Circle((0, r_max), r_max, color='#2c3e50', fill=False, lw=4))
        
        ax_t.axhline(y=sp_nivel, color='red', ls='--', lw=2)
        ax_t.text(0, h_total * 1.2, f"PV: {valor_presente:.3f} m", ha='center', fontweight='bold')
        
        placeholder_tanque.pyplot(fig_t)
        plt.close(fig_t)
        
        # Gráfico de tendencia
        fig_tr, ax_tr = plt.subplots(figsize=(8, 3.5))
        ax_tr.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2)
        ax_tr.axhline(y=sp_nivel, color='red', ls='--', alpha=0.5)
        if p_activa and p_tiempo > 0 and t_act >= p_tiempo:
            ax_tr.axvspan(p_tiempo, tiempo_ensayo, alpha=0.1, color='orange')
        ax_tr.set_xlabel('Tiempo [s]')
        ax_tr.set_ylabel('Altura [m]')
        ax_tr.grid(True, alpha=0.2)
        placeholder_grafico.pyplot(fig_tr)
        plt.close(fig_tr)
        
        # Gráfico de acción de control
        fig_u, ax_u = plt.subplots(figsize=(8, 2.5))
        ax_u.step(vector_t[:i+1], u_log, color='#e67e22', where='post')
        ax_u.set_xlim(0, tiempo_ensayo)
        ax_u.grid(True, alpha=0.2)
        placeholder_u.pyplot(fig_u)
        plt.close(fig_u)
        
        # Gráfico de válvula
        fig_v, ax_v = plt.subplots(figsize=(8, 3))
        ax_v.plot(vector_t[:i+1], u_log, color='#2ecc71', lw=2.5)
        ax_v.set_ylim(-0.1, 1.1)
        ax_v.set_yticks([0, 0.5, 1])
        ax_v.set_yticklabels(['CERRADA', '50%', 'ABIERTA'])
        placeholder_valvula.pyplot(fig_v)
        plt.close(fig_v)
        
        # Gráfico comparativo
        fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
        ax_comp.plot(vector_t[:i+1], h_log, color='#1f77b4', lw=2, label='Simulación')
        if mostrar_ref and tiene_datos_exp and len(t_exp) > 0:
            ax_comp.scatter(t_exp, h_exp, color='red', marker='x', s=100, label='Datos')
        ax_comp.set_xlabel("Tiempo [s]")
        ax_comp.set_ylabel("Nivel [m]")
        ax_comp.grid(True, alpha=0.3)
        ax_comp.legend()
        placeholder_comparativa.pyplot(fig_comp)
        plt.close(fig_comp)
        
        time.sleep(0.01)
        barra_p.progress((i+1)/len(vector_t))
    
    # Fin de simulación
    status_placeholder.empty()
    st.success(f"✅ Simulación completada")
    st.balloons()
    
    # Resultados finales
    st.markdown("---")
    st.subheader("📈 Resultados Finales")
    
    col_an1, col_an2 = st.columns([2, 1])
    
    with col_an1:
        fig_amp, ax_amp = plt.subplots(figsize=(10, 5))
        ax_amp.plot(vector_t, h_log, color='#1f77b4', lw=2.5, label='Respuesta del Sistema')
        ax_amp.axhline(y=sp_nivel, color='#d62728', linestyle='--', lw=2, label='Setpoint')
        if p_activa and p_tiempo > 0:
            ax_amp.axvline(x=p_tiempo, color='orange', linestyle='--', alpha=0.7, label='Perturbación')
        ax_amp.set_xlabel("Tiempo (s)")
        ax_amp.set_ylabel("Nivel (m)")
        ax_amp.grid(True, alpha=0.5)
        ax_amp.legend()
        st.pyplot(fig_amp)
        plt.close(fig_amp)
    
    with col_an2:
        sobrepico = ((max(h_log) - sp_nivel) / sp_nivel) * 100 if max(h_log) > sp_nivel else 0
        st.metric("Sobrepico", f"{sobrepico:.2f} %")
        st.metric("IAE Final", f"{iae_acumulado:.2f}")
        st.metric("ITAE Final", f"{itae_acumulado:.2f}")
    
    # Exportación
    df_final = pd.DataFrame({
        "Tiempo [s]": vector_t,
        "Nivel [m]": h_log,
        "Control [m3/s]": u_log,
        "Error [m]": e_log
    })
    
    st.download_button(
        label="📥 Descargar CSV",
        data=df_final.to_csv(index=False),
        file_name=f"resultados_{geom_tanque}.csv",
        mime="text/csv"
    )


# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown("""
<hr>
<div style="text-align: center; color: #5d6d7e; font-size: 0.8rem;">
    <p>Universidad Central de Venezuela - Escuela de Ingeniería Química</p>
</div>
""", unsafe_allow_html=True)
