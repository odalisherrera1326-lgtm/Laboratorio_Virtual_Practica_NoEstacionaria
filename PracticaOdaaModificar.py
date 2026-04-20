import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import base64

# =============================================================================
# --- VALORES POR DEFECTO ---
# =============================================================================
modo_auto = False
p_activa = True
p_magnitud = 0.045
p_tiempo = 80
p_tipo = "Entrada"  # Puede ser "Entrada" o "Salida"

# =============================================================================
# --- FUNCIONES DE CÁLCULO ---
# =============================================================================
def get_area_transversal(geom, r, h, h_total):
    """Calcula el área transversal para cualquier geometría en función de la altura actual"""
    h_efectiva = max(h, 0.001)
    
    if geom == "Cilíndrico":
        return np.pi * (r ** 2)
    elif geom == "Cónico":
        radio_actual = (r / h_total) * h_efectiva
        return np.pi * (radio_actual ** 2)
    else:  # Esférico
        if h_efectiva <= 2 * r:
            radio_corte = np.sqrt(r**2 - (h_efectiva - r)**2)
            return np.pi * (radio_corte ** 2)
        else:
            return np.pi * (r ** 2)


def calcular_cd_inteligente(df_usr, r, h_t, geom, area_ori):
    """Calcula el Coeficiente de Descarga (Cd) usando el balance de masa."""
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


def sintonizar_controlador_robusto(geom, r, h_t, cd_calculado=0.61, q_max=2.0):
    """
    Sintonización robusta del PID para control de nivel con dos válvulas.
    Incluye ajuste por Cd.
    """
    # Calcular área transversal característica
    if geom == "Cilíndrico":
        area_t = np.pi * (r**2)
    elif geom == "Cónico":
        area_t = np.pi * (r/2)**2
    else:  # Esférico
        area_t = (2/3) * np.pi * (r**2)
    
    # Constante de tiempo aproximada
    tau = area_t * h_t / q_max
    
    # Sintonización Ziegler-Nichols para sistemas de primer orden
    kp = 1.2 * tau / (area_t * 0.1)
    ki = kp / (tau * 0.5)
    kd = kp * tau * 0.1
    
    # Ajuste por Cd (mayor Cd = más flujo = necesita ganancias más altas)
    factor_cd = np.clip(cd_calculado / 0.61, 0.7, 1.5)
    kp = kp * factor_cd
    ki = ki * factor_cd
    
    # Límites razonables
    kp = np.clip(kp, 5.0, 30.0)
    ki = np.clip(ki, 0.5, 5.0)
    kd = np.clip(kd, 0.1, 2.0)
    
    return round(kp, 2), round(ki, 3), round(kd, 2)


def resolver_sistema_completo(dt, h_prev, sp, geom, r, h_t, q_p_val, p_tipo, e_sum, e_prev, kp, ki, kd, q_max=2.0):
    """
    Sistema con DOS VÁLVULAS DE CONTROL que trabajan juntas para regular el nivel.
    """
    
    # Área transversal en la altura actual
    area_h = get_area_transversal(geom, r, h_prev, h_t)
    area_h = max(area_h, 0.0001)
    
    # Error actual (SP - PV)
    err = sp - h_prev
    
    # =========================================================================
    # ACCIONES PID
    # =========================================================================
    
    # Proporcional
    P = kp * err
    
    # Integral con anti-windup
    e_sum += err * dt
    e_sum = np.clip(e_sum, -30.0, 30.0)
    I = ki * e_sum
    
    # Derivativo
    if dt > 0:
        D = kd * (err - e_prev) / dt
        D = np.clip(D, -3.0, 3.0)
    else:
        D = 0.0
    
    # Acción de control total
    u_control = P + I + D
    
    # =========================================================================
    # ESTRATEGIA DE CONTROL CON DOS VÁLVULAS
    # =========================================================================
    
    if u_control > 0:
        q_entrada = np.clip(u_control, 0, q_max)
        q_salida = 0.0
    else:
        q_entrada = 0.0
        q_salida = np.clip(-u_control, 0, q_max)
    
    # Agregar perturbación según su tipo
    if p_tipo == "Entrada":
        q_entrada_total = q_entrada + q_p_val
        q_salida_total = q_salida
    else:  # Salida (fuga)
        q_entrada_total = q_entrada
        q_salida_total = q_salida + q_p_val
    
    # Balance de masa
    dh_dt = (q_entrada_total - q_salida_total) / area_h
    
    # Actualizar nivel con límites físicos
    h_next = h_prev + dh_dt * dt
    h_next = np.clip(h_next, 0.0, h_t)
    
    return h_next, q_entrada, q_salida, err, e_sum, err


def get_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Lab Virtual - Control PID de Nivel",
    page_icon="🧪",
    layout="wide"
)

if 'ejecutando' not in st.session_state:
    st.session_state.ejecutando = False

# =============================================================================
# ESTILOS CSS
# =============================================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f4f7f9 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a5276 0%, #154360 100%) !important;
    border-right: 4px solid #f1c40f !important;
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label {
    color: #f0f4f8 !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #f1c40f !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f5f9fc 100%) !important;
    border-left: 5px solid #1a5276 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    padding: 15px !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12) !important;
}

div[data-testid="stMetric"] label {
    color: #1a5276 !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #1a5276, #2471a3) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(90deg, #2471a3, #2e86c1) !important;
    transform: scale(1.02);
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(90deg, #7b241c, #943126) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
}

.header-container {
    background: linear-gradient(135deg, #0d3251 0%, #1a5276 50%, #1f618d 100%);
    border-radius: 20px;
    padding: 20px 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

.flow-indicator {
    font-size: 1.2rem;
    font-weight: bold;
    color: #1a5276;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1a5276, #3498db, #1a5276) !important;
    background-size: 200% 100% !important;
    animation: gradientMove 1.5s ease infinite;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

.streamlit-expanderHeader {
    background-color: #e8f0f7 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #1a5276 !important;
    border-left: 4px solid #f1c40f !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ENCABEZADO
# =============================================================================
st.markdown("""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="width: 120px;">
            <h3 style="color: white;">UCV</h3>
        </div>
        <div>
            <h1 style="color: white !important; font-size: 2.2rem;">Control PID de Nivel - Dos Válvulas</h1>
            <p style="color: #d4e6f1 !important; margin: 0;">Escuela de Ingeniería Química | Facultad de Ingeniería - UCV</p>
        </div>
        <div style="width: 120px;">
            <h3 style="color: white;">EIQ</h3>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# BARRA LATERAL
# =============================================================================
st.sidebar.header("⚙️ Configuración del Sistema")

with st.sidebar.container(border=True):
    geom_tanque = st.sidebar.selectbox("Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("Especificaciones del Tanque", expanded=True):
    r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
    h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
    h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
    sp_nivel = st.slider("Setpoint [m]", 0.2, float(h_total)-0.2, float(h_total/2))

with st.sidebar.expander("Válvulas de Control", expanded=True):
    q_max = st.number_input("Flujo máximo por válvula [m³/s]", value=2.0, min_value=0.5, max_value=5.0, step=0.5)

with st.sidebar.expander("🛡️ Escenario de Perturbación", expanded=True):
    p_activa = st.toggle("Activar Perturbación", value=True)
    
    if p_activa:
        p_tipo = st.selectbox("Tipo de Perturbación", ["Entrada", "Salida (Fuga)"], 
                              help="Entrada: flujo adicional que entra. Salida: fuga que sale.")
        p_tipo = "Entrada" if p_tipo == "Entrada" else "Salida"
        p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.5, min_value=0.1, max_value=3.0, step=0.1, format="%.2f")
        p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 100)
    else:
        p_magnitud = 0.0
        p_tiempo = 0
        p_tipo = "Entrada"

# =============================================================================
# DATOS EXPERIMENTALES Y CÁLCULO DE Cd
# =============================================================================
with st.sidebar.expander("📊 Datos Experimentales y Cd", expanded=True):
    st.write("Ingresa los datos medidos en laboratorio:")
    st.caption("⚠️ El nivel debe ingresarse en **centímetros (cm)**")
    
    df_exp_default = pd.DataFrame({
        "Tiempo (s)": [0, 60, 120, 180, 240, 300],
        "Nivel Medido (cm)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    datos_usr = st.data_editor(df_exp_default, num_rows="dynamic", key="datos_exp")
    mostrar_ref = st.checkbox("Mostrar referencia en gráfica", value=True)
    
    # Calcular Cd automáticamente
    if st.button("🧮 Calcular Cd", use_container_width=True):
        if not isinstance(datos_usr, pd.DataFrame):
            datos_usr = pd.DataFrame(datos_usr)
        
        if "Nivel Medido (cm)" in datos_usr.columns and len(datos_usr) >= 2:
            df_calib = datos_usr.copy()
            df_calib["Nivel Medido (m)"] = df_calib["Nivel Medido (cm)"] / 100
            area_ori_temp = np.pi * ((0.0254) / 2)**2  # 1 pulgada
            cd_calculado = calcular_cd_inteligente(
                df_calib[["Tiempo (s)", "Nivel Medido (m)"]], 
                r_max, h_total, geom_tanque, area_ori_temp
            )
            st.session_state['cd_calculado'] = cd_calculado
            st.success(f"✅ Cd calculado: {cd_calculado:.4f}")
        else:
            st.warning("⚠️ Ingresa al menos 2 datos para calcular Cd")
            st.session_state['cd_calculado'] = 0.61

with st.sidebar.expander("Parámetros PID", expanded=True):
    cd_actual = st.session_state.get('cd_calculado', 0.61)
    kp_sug, ki_sug, kd_sug = sintonizar_controlador_robusto(geom_tanque, r_max, h_total, cd_actual, q_max)
    
    modo_auto = st.checkbox("🎯 Modo Auto-sintonía", value=True)
    
    if modo_auto:
        st.success(f"💡 PID optimizado (Cd={cd_actual:.3f}):")
        st.caption(f"Kp={kp_sug} | Ki={ki_sug} | Kd={kd_sug}")
        kp_val = st.number_input("Kp", value=kp_sug, key="kp_auto")
        ki_val = st.number_input("Ki", value=ki_sug, format="%.3f", key="ki_auto")
        kd_val = st.number_input("Kd", value=kd_sug, format="%.3f", key="kd_auto")
    else:
        kp_val = st.number_input("Kp", value=10.0, step=1.0, key="kp_man")
        ki_val = st.number_input("Ki", value=2.0, step=0.5, format="%.3f", key="ki_man")
        kd_val = st.number_input("Kd", value=0.5, step=0.1, format="%.3f", key="kd_man")
    
    tiempo_ensayo = st.slider("Tiempo de simulación [s]", 60, 600, 300)

# Botones
st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    iniciar_sim = st.button("▶️ Iniciar Simulación", use_container_width=True, type="primary")
with col_btn2:
    btn_reset = st.button("🔄 Reset", use_container_width=True, type="secondary")

if btn_reset:
    st.session_state.ejecutando = False
    st.session_state['cd_calculado'] = 0.61
    st.rerun()


# =============================================================================
# SIMULACIÓN PRINCIPAL
# =============================================================================
if not st.session_state.ejecutando:
    st.info("💡 Configure los parámetros, calcule Cd (opcional) y presione 'Iniciar Simulación'")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 🎯 Estrategia de Control con Dos Válvulas
        
        **Lógica de control:**
        - Si **h < SP**: Abrir V-01, cerrar V-02 → Sube nivel
        - Si **h > SP**: Cerrar V-01, abrir V-02 → Baja nivel
        - Si hay **perturbación**, el PID compensa automáticamente
        """)
else:
    # Inicializar simulación
    if iniciar_sim:
        st.session_state['error_acumulado'] = 0.0
        st.session_state['ultimo_error'] = 0.0
        if 'cd_calculado' not in st.session_state:
            st.session_state['cd_calculado'] = 0.61
    
    col_graf, col_met = st.columns([2, 1])
    
    with col_graf:
        st.subheader("🎮 Monitor del Proceso")
        placeholder_tanque = st.empty()
        st.subheader("📈 Tendencia Temporal")
        placeholder_grafico = st.empty()
        st.subheader("🔧 Acción de las Válvulas")
        placeholder_valvulas = st.empty()
        st.markdown("---")
        st.subheader("📊 Comparativa: Simulación vs Datos Experimentales")
        placeholder_comparativa = st.empty()
    
    with col_met:
        st.subheader("📊 Métricas de Control")
        
        kp_show = kp_val if not modo_auto else kp_sug
        ki_show = ki_val if not modo_auto else ki_sug
        cd_show = st.session_state.get('cd_calculado', 0.61)
        
        st.write(f"**Parámetros Activos:**")
        st.caption(f"Kp: {kp_show} | Ki: {ki_show} | Kd: {kd_val if not modo_auto else kd_sug}")
        st.caption(f"Cd: {cd_show:.3f} | Qmax: {q_max} m³/s")
        st.markdown("---")
        
        placeholder_iae = st.empty()
        placeholder_itae = st.empty()
        placeholder_iae.metric("IAE (Error Acumulado)", "0.00")
        placeholder_itae.metric("ITAE", "0.00")
        
        st.markdown("---")
        m_h = st.empty()
        m_e = st.empty()
        m_qin = st.empty()
        m_qout = st.empty()
        m_h.metric("Nivel [m]", "0.000")
        m_e.metric("Error [m]", "0.000")
        m_qin.metric("Flujo Entrada [m³/s]", "0.000")
        m_qout.metric("Flujo Salida [m³/s]", "0.000")
    
    # Preparar simulación
    status_placeholder = st.empty()
    dt = 1.0
    vector_t = np.arange(0, tiempo_ensayo, dt)
    h_log, qin_log, qout_log, e_log = [], [], [], []
    
    h_corrida = 0.5  # Nivel inicial
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
    
    # Usar PID seleccionado
    k_p = kp_sug if modo_auto else kp_val
    k_i = ki_sug if modo_auto else ki_val
    k_d = kd_sug if modo_auto else kd_val
    
    # Bucle de simulación
    for i, t_act in enumerate(vector_t):
        status_placeholder.markdown("<div class='flow-indicator'>⚡ PID ACTIVO - CONTROLANDO...</div>", unsafe_allow_html=True)
        
        # Perturbación
        if p_activa and t_act >= p_tiempo:
            q_p_inst = p_magnitud
        else:
            q_p_inst = 0.0
        
        # Resolver sistema
        h_corrida, q_entrada, q_salida, e_inst, err_int, err_pasado = resolver_sistema_completo(
            dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst, p_tipo,
            err_int, err_pasado, k_p, k_i, k_d, q_max
        )
        
        # Acumular métricas
        iae_acumulado += abs(e_inst) * dt
        itae_acumulado += (t_act * abs(e_inst)) * dt
        
        # Guardar logs
        h_log.append(h_corrida)
        qin_log.append(q_entrada)
        qout_log.append(q_salida)
        e_log.append(e_inst)
        
        # Actualizar métricas
        m_h.metric("Nivel [m]", f"{h_corrida:.3f}")
        m_e.metric("Error [m]", f"{e_inst:.4f}")
        m_qin.metric("Flujo Entrada [m³/s]", f"{q_entrada:.3f}")
        m_qout.metric("Flujo Salida [m³/s]", f"{q_salida:.3f}")
        placeholder_iae.metric("IAE", f"{iae_acumulado:.2f}")
        placeholder_itae.metric("ITAE", f"{itae_acumulado:.2f}")
        
        # Visualización del tanque
        fig_t, ax_t = plt.subplots(figsize=(7, 5))
        ax_t.set_axis_off()
        ax_t.set_xlim(-r_max*3, r_max*3)
        ax_t.set_ylim(-0.8, h_total*1.3)
        
        # Color según error
        if abs(e_inst) < 0.05:
            color_agua = '#27ae60'
        elif abs(e_inst) < 0.15:
            color_agua = '#f39c12'
        else:
            color_agua = '#e74c3c'
        
        if geom_tanque == "Cilíndrico":
            ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=5, zorder=2)
            ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, h_corrida, color=color_agua, alpha=0.7, zorder=1))
            if q_entrada > 0:
                ax_t.annotate('', xy=(-r_max-1.5, h_corrida*0.7), xytext=(-r_max-0.3, h_corrida*0.7),
                            arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
            if q_salida > 0:
                ax_t.annotate('', xy=(r_max+1.5, 0.3), xytext=(r_max+0.3, 0.3),
                            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
            
        elif geom_tanque == "Cónico":
            ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=5, zorder=2)
            if h_corrida > 0:
                radio_h = (r_max / h_total) * h_corrida
                vertices = [[-radio_h, h_corrida], [radio_h, h_corrida], [0, 0]]
                ax_t.add_patch(plt.Polygon(vertices, color=color_agua, alpha=0.7, zorder=1))
            
        else:  # Esférico
            import math
            ax_t.add_patch(plt.Circle((0, r_max), r_max, color='#2c3e50', fill=False, lw=5, zorder=2))
            agua = plt.Circle((0, r_max), r_max, color=color_agua, alpha=0.7, zorder=1)
            ax_t.add_patch(agua)
            recorte = plt.Rectangle((-r_max, 0), 2*r_max, h_corrida, transform=ax_t.transData)
            agua.set_clip_path(recorte)
        
        # Setpoint
        ax_t.axhline(y=sp_nivel, color='red', ls='--', lw=2, zorder=3, alpha=0.8)
        ax_t.text(-r_max*2.5, sp_nivel + 0.05, f"SP: {sp_nivel:.2f}m", color='red', fontweight='bold')
        ax_t.text(0, h_total * 1.2, f"PV: {h_corrida:.3f} m", ha='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='#1a5276', boxstyle='round'))
        
        if p_activa and t_act >= p_tiempo:
            ax_t.text(0, -0.5, f"⚠️ PERTURBACIÓN ACTIVA", ha='center', color='orange', fontweight='bold')
        
        placeholder_tanque.pyplot(fig_t)
        plt.close(fig_t)
        
        # Gráfico de tendencia
        fig_tr, ax_tr = plt.subplots(figsize=(8, 3.5))
        ax_tr.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2.5, label='Simulación')
        ax_tr.axhline(y=sp_nivel, color='red', ls='--', alpha=0.5, label='Setpoint')
        if p_activa and t_act >= p_tiempo:
            ax_tr.axvline(x=p_tiempo, color='orange', ls='--', alpha=0.7)
            ax_tr.axvspan(p_tiempo, tiempo_ensayo, alpha=0.1, color='orange')
        ax_tr.set_xlabel('Tiempo [s]')
        ax_tr.set_ylabel('Nivel [m]')
        ax_tr.legend()
        ax_tr.set_xlim(0, tiempo_ensayo)
        ax_tr.set_ylim(0, h_total * 1.1)
        ax_tr.grid(True, alpha=0.3)
        placeholder_grafico.pyplot(fig_tr)
        plt.close(fig_tr)
        
        # Gráfico de válvulas
        fig_v, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.step(vector_t[:i+1], qin_log, where='post', color='blue', lw=2)
        ax1.set_ylabel('Q entrada [m³/s]')
        ax1.set_ylim(0, q_max * 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Válvula de Entrada (V-01)')
        
        ax2.step(vector_t[:i+1], qout_log, where='post', color='red', lw=2)
        ax2.set_ylabel('Q salida [m³/s]')
        ax2.set_xlabel('Tiempo [s]')
        ax2.set_ylim(0, q_max * 1.1)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Válvula de Salida (V-02)')
        
        plt.tight_layout()
        placeholder_valvulas.pyplot(fig_v)
        plt.close(fig_v)
        
        # Gráfico comparativo
        fig_comp, ax_comp = plt.subplots(figsize=(8, 3))
        ax_comp.plot(vector_t[:i+1], h_log, color='#1f77b4', lw=2, label='Simulación')
        if mostrar_ref and tiene_datos_exp and len(t_exp) > 0:
            ax_comp.scatter(t_exp, h_exp, color='red', marker='x', s=80, label='Datos Experimentales')
            ax_comp.plot(t_exp, h_exp, color='red', linestyle='--', alpha=0.3)
        ax_comp.axhline(y=sp_nivel, color='green', ls='--', alpha=0.3, label='Setpoint')
        ax_comp.set_xlabel("Tiempo [s]")
        ax_comp.set_ylabel("Nivel [m]")
        ax_comp.set_ylim(0, h_total * 1.1)
        ax_comp.grid(True, alpha=0.3)
        ax_comp.legend(loc='lower right', fontsize='x-small')
        placeholder_comparativa.pyplot(fig_comp)
        plt.close(fig_comp)
        
        time.sleep(0.02)
        barra_p.progress((i+1)/len(vector_t))
    
    status_placeholder.empty()
    st.success("✅ Simulación completada - Control PID exitoso")
    st.balloons()
    
    # Análisis final
    st.markdown("---")
    st.subheader("📈 Análisis de Respuesta")
    
    col_an1, col_an2 = st.columns([2, 1])
    
    with col_an1:
        fig_amp, ax_amp = plt.subplots(figsize=(10, 5))
        ax_amp.plot(vector_t, h_log, color='#1f77b4', lw=2.5, label='Respuesta del Sistema')
        ax_amp.axhline(y=sp_nivel, color='#d62728', linestyle='--', lw=2, label='Setpoint')
        if p_activa and p_tiempo > 0:
            ax_amp.axvline(x=p_tiempo, color='orange', linestyle='--', alpha=0.7, label='Perturbación')
            ax_amp.axvspan(p_tiempo, tiempo_ensayo, alpha=0.08, color='orange')
        ax_amp.set_title("Respuesta del Control PID con Dos Válvulas", fontsize=12)
        ax_amp.set_xlabel("Tiempo [s]")
        ax_amp.set_ylabel("Nivel [m]")
        ax_amp.grid(True, alpha=0.3)
        ax_amp.legend()
        st.pyplot(fig_amp)
        plt.close(fig_amp)
    
    with col_an2:
        sobrepico = ((max(h_log) - sp_nivel) / sp_nivel) * 100 if max(h_log) > sp_nivel else 0
        st.metric("Sobrepico Máximo", f"{sobrepico:.2f} %")
        st.metric("IAE Final", f"{iae_acumulado:.2f}")
        st.metric("ITAE Final", f"{itae_acumulado:.2f}")
        
        err_final = abs(h_log[-1] - sp_nivel) if h_log else 0
        st.metric("Error Final", f"{err_final:.4f} m")
        
        if err_final < 0.02:
            st.success("✅ Excelente control - Error < 2%")
        elif err_final < 0.05:
            st.info("👍 Buen control")
        else:
            st.warning("⚠️ Ajustar PID para mejor precisión")
    
    # Tabla de resumen
    st.markdown("---")
    st.subheader("📋 Resumen de Datos y Estabilidad del Control")
    
    col_tab, col_res = st.columns([2, 1])
    
    with col_tab:
        df_resumen = pd.DataFrame({
            "Tiempo [s]": vector_t[-10:] if len(vector_t) >= 10 else vector_t,
            "Nivel [m]": h_log[-10:] if len(h_log) >= 10 else h_log,
            "Q_entrada [m³/s]": qin_log[-10:] if len(qin_log) >= 10 else qin_log,
            "Q_salida [m³/s]": qout_log[-10:] if len(qout_log) >= 10 else qout_log,
            "Error [m]": e_log[-10:] if len(e_log) >= 10 else e_log
        })
        
        st.dataframe(df_resumen.style.format({
            "Tiempo [s]": "{:.0f}",
            "Nivel [m]": "{:.4f}",
            "Q_entrada [m³/s]": "{:.4f}",
            "Q_salida [m³/s]": "{:.4f}",
            "Error [m]": "{:.4f}"
        }), use_container_width=True)
        
        st.caption("📊 Últimos 10 datos de la simulación")
    
    with col_res:
        err_f = abs(sp_nivel - h_log[-1]) if len(h_log) > 0 else 0
        st.metric("Error Residual Final (Offset)", f"{err_f:.4f} m")
        
        if err_f < 0.01:
            st.success("✅ Error residual nulo")
        elif err_f < 0.05:
            st.info("👍 Error residual aceptable")
        else:
            st.warning("⚠️ Aumentar Ki")
    
    # Exportar datos
    st.markdown("---")
    
    df_final = pd.DataFrame({
        "Tiempo [s]": vector_t,
        "Nivel [m]": h_log,
        "Q_entrada [m3/s]": qin_log,
        "Q_salida [m3/s]": qout_log,
        "Error [m]": e_log,
        "Kp_Usado": [k_p] * len(vector_t),
        "Ki_Usado": [k_i] * len(vector_t),
        "Kd_Usado": [k_d] * len(vector_t),
        "Cd_Usado": [st.session_state.get('cd_calculado', 0.61)] * len(vector_t)
    })
    
    col_down1, col_down2, col_down3 = st.columns([1, 2, 1])
    with col_down2:
        st.download_button(
            label="📥 Descargar Reporte Completo (CSV)",
            data=df_final.to_csv(index=False),
            file_name=f"control_pid_{geom_tanque.lower()}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("""
<hr style="margin: 2rem 0 1rem 0; border-color: #1a5276;">
<div style="text-align: center; color: #5d6d7e; font-size: 0.8rem;">
    <p>Universidad Central de Venezuela - Escuela de Ingeniería Química</p>
    <p>Control PID de Nivel con Dos Válvulas | Laboratorio Virtual | © 2025</p>
</div>
""", unsafe_allow_html=True)
