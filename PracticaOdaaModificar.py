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
p_tipo = "Entrada"

# =============================================================================
# --- FUNCIONES DE CÁLCULO (NUEVA LÓGICA CON DOS VÁLVULAS) ---
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
    """Sintonización robusta del PID para control de nivel con dos válvulas."""
    if geom == "Cilíndrico":
        area_t = np.pi * (r**2)
    elif geom == "Cónico":
        area_t = np.pi * (r/2)**2
    else:
        area_t = (2/3) * np.pi * (r**2)
    
    tau = area_t * h_t / q_max
    
    kp = 1.2 * tau / (area_t * 0.1)
    ki = kp / (tau * 0.5)
    kd = kp * tau * 0.1
    
    factor_cd = np.clip(cd_calculado / 0.61, 0.7, 1.5)
    kp = kp * factor_cd
    ki = ki * factor_cd
    
    kp = np.clip(kp, 5.0, 30.0)
    ki = np.clip(ki, 0.5, 5.0)
    kd = np.clip(kd, 0.1, 2.0)
    
    return round(kp, 2), round(ki, 3), round(kd, 2)


def resolver_sistema_dos_valvulas(dt, h_prev, sp, geom, r, h_t, q_p_val, p_tipo, e_sum, e_prev, kp, ki, kd, q_max=2.0):
    """Sistema con DOS VÁLVULAS DE CONTROL que trabajan juntas para regular el nivel."""
    
    area_h = get_area_transversal(geom, r, h_prev, h_t)
    area_h = max(area_h, 0.0001)
    
    err = sp - h_prev
    
    P = kp * err
    e_sum += err * dt
    e_sum = np.clip(e_sum, -30.0, 30.0)
    I = ki * e_sum
    
    if dt > 0:
        D = kd * (err - e_prev) / dt
        D = np.clip(D, -3.0, 3.0)
    else:
        D = 0.0
    
    u_control = P + I + D
    
    if u_control > 0:
        q_entrada = np.clip(u_control, 0, q_max)
        q_salida = 0.0
    else:
        q_entrada = 0.0
        q_salida = np.clip(-u_control, 0, q_max)
    
    if p_tipo == "Entrada":
        q_entrada_total = q_entrada + q_p_val
        q_salida_total = q_salida
    else:
        q_entrada_total = q_entrada
        q_salida_total = q_salida + q_p_val
    
    dh_dt = (q_entrada_total - q_salida_total) / area_h
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
    page_title="Lab Virtual - Simulación Dinámica -",
    page_icon="🧪",
    layout="wide"
)

if 'ejecutando' not in st.session_state:
    st.session_state.ejecutando = False


# =============================================================================
# ESTILOS CSS (ORIGINAL CONSERVADO)
# =============================================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f4f7f9 !important;
    cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='%23333' stroke-width='1.5'><circle cx='12' cy='12' r='3'/><path d='M12 1v3M12 20v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M1 12h3M20 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12'/></svg>") 12 12, auto !important;
}

button, a, [data-testid="stHeaderActionElements"], .stSlider {
    cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='%23333' stroke-width='1.5'><circle cx='12' cy='12' r='3'/><path d='M12 1v3M12 20v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M1 12h3M20 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12'/></svg>") 12 12, pointer !important;
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
    font-weight: 400 !important;
    line-height: 1.4 !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #f1c40f !important;
    border-bottom: 1px solid #f1c40f80;
    padding-bottom: 5px;
    margin-top: 10px;
}

.streamlit-expanderHeader {
    background-color: #e8f0f7 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    color: #1a5276 !important;
    border-left: 4px solid #f1c40f !important;
    transition: all 0.2s ease !important;
}

.streamlit-expanderHeader:hover {
    background-color: #d4e6f1 !important;
    transform: translateX(3px);
}

.streamlit-expanderContent {
    background-color: #ffffff !important;
    border-radius: 0 0 10px 10px !important;
    border: 1px solid #e0e8f0 !important;
    border-top: none !important;
    padding: 15px !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f5f9fc 100%) !important;
    border: none !important;
    border-left: 5px solid #1a5276 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    padding: 15px !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12) !important;
}

div[data-testid="stMetric"] label {
    color: #1a5276 !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
    text-transform: uppercase !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #154360 !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
}

.stButton > button[kind="primary"], 
.stButton > button:first-child:not([kind="secondary"]) {
    background: linear-gradient(90deg, #1a5276, #2471a3) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button:first-child:not([kind="secondary"]):hover {
    background: linear-gradient(90deg, #2471a3, #2e86c1) !important;
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(26,82,118,0.4) !important;
}

.stButton > button[kind="secondary"] {
    background: linear-gradient(90deg, #7b241c, #943126) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    transition: all 0.3s ease !important;
}

.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(90deg, #943126, #a93226) !important;
    transform: scale(1.02);
    box-shadow: 0 4px 12px rgba(148,49,38,0.4) !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1a5276, #3498db, #1a5276) !important;
    background-size: 200% 100% !important;
    animation: gradientMove 1.5s ease infinite, pulso_azul 2s ease-in-out infinite !important;
    border-radius: 10px !important;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

@keyframes pulso_azul {
    0% { opacity: 0.7; }
    50% { opacity: 1; }
    100% { opacity: 0.7; }
}

div[data-baseweb="slider"] > div > div > div {
    background-color: #f39c12 !important;
}

div[role="slider"] {
    background-color: #f39c12 !important;
    border: 2px solid white !important;
}

.header-container {
    background: linear-gradient(135deg, #0d3251 0%, #1a5276 50%, #1f618d 100%);
    background-size: 200% 200%;
    animation: gradientBG 8s ease infinite;
    border-radius: 20px;
    padding: 20px 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@media (max-width: 768px) {
    .header-container h1 {
        font-size: 1.2rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
}

div[data-testid="stInfo"] {
    background-color: #e8f4fd !important;
    border-left: 5px solid #1a5276 !important;
    border-radius: 10px !important;
}

div[data-testid="stSuccess"] {
    background-color: #e8f8e8 !important;
    border-left: 5px solid #27ae60 !important;
    border-radius: 10px !important;
}

div[data-testid="stWarning"] {
    background-color: #fef5e7 !important;
    border-left: 5px solid #f39c12 !important;
    border-radius: 10px !important;
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
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ENCABEZADO INSTITUCIONAL
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
            <p style="color: #d4e6f1 !important; margin: 0;">Escuela de Ingeniería Química | Facultad de Ingeniería - UCV</p>
        </div>
        <div style="width: 160px;">
            {f'<img src="data:image/png;base64,{logo_eiq_64}" width="150">' if logo_eiq_64 else "EIQ"}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MARCO TEÓRICO
# =============================================================================
col_teoria1, col_teoria2, col_teoria3 = st.columns(3)

with col_teoria1:
    with st.expander("📚 Fundamento teórico: Ecuaciones de Conservación y Descarga", expanded=False):
        st.markdown(r"""
        La dinámica del sistema se describe mediante el **Balance Global de Masa** para un volumen de control con densidad constante ($\rho$):
        
        $$ \frac{dV}{dt} = Q_{in} - Q_{out} \pm Q_{p} $$
        
        Considerando que el volumen es función del nivel ($V = \int A(h)dh$), aplicamos la regla de la cadena para obtener la ecuación general de vaciado/llenado válida para **cualquier área transversal $A(h)$**:
        
        $$ A(h) \frac{dh}{dt} = Q_{in} - Q_{out} \pm Q_{p} $$
        
        Donde:
        * **$A(h)$**: Área de la sección transversal en función de la altura (m²).
        * **$Q_{in}$**: Flujo de entrada controlado (m³/s).
        * **$Q_{out}$**: Flujo de salida controlado (m³/s).
        * **$Q_{p}$**: Flujo de perturbación o falla (m³/s).
        """)

with col_teoria2:
    with st.expander("🎮 Teoría: Estrategia de control PID con Dos Válvulas", expanded=False):
        st.markdown(r"""
        El sistema utiliza un controlador **PID** que actúa sobre **DOS VÁLVULAS**:
        
        $$ u(t) = K_p e(t) + K_i \int_{0}^{t} e(\tau) d\tau + K_d \frac{de(t)}{dt} $$
        
        **Estrategia de control bidireccional:**
        * **V-01 (Entrada):** Controla $Q_{in}$ (0 a $Q_{max}$)
        * **V-02 (Salida):** Controla $Q_{out}$ (0 a $Q_{max}$)
        
        **Lógica de operación:**
        * Si $h < SP$ ($u > 0$): Abrir V-01, cerrar V-02 → Sube nivel
        * Si $h > SP$ ($u < 0$): Cerrar V-01, abrir V-02 → Baja nivel
        
        **Ventajas:**
        * Control completo en ambos sentidos
        * Rechazo activo de perturbaciones
        * Sin zonas muertas de control
        """)

with col_teoria3:
    with st.expander("📊 Criterios de Desempeño (IAE/ITAE)", expanded=False):
        st.markdown(r"""
        Para evaluar la eficiencia del control, se utilizan métricas integrales del error $e(t) = SP - PV$:

        1. **IAE (Integral del Error Absoluto):**
        $$IAE = \int_{0}^{t} |e(t)| dt$$
        Mide el rendimiento acumulado.

        2. **ITAE (Integral del Tiempo por el Error Absoluto):**
        $$ITAE = \int_{0}^{t} t \cdot |e(t)| dt$$
        **Penaliza errores que duran mucho tiempo.**
        """)


# =============================================================================
# DIAGRAMA DEL PROCESO
# =============================================================================
estado_expander = not st.session_state.get('ejecutando', False)

with st.expander("🔧 Diagrama del Proceso", expanded=estado_expander):
    col_img = st.columns([1, 5, 1])[1]
    with col_img:
        if os.path.exists("Captura de pantalla 2026-03-29 163125.png"):
            st.image("Captura de pantalla 2026-03-29 163125.png", use_container_width=True)
        else:
            st.info("📍 Diagrama del sistema de control con dos válvulas")
            st.markdown("""
""")


# =============================================================================
# BARRA LATERAL - PARÁMETROS
# =============================================================================
st.sidebar.header("⚙️ Configuración del Sistema")

with st.sidebar.container(border=True):
geom_tanque = st.sidebar.selectbox("Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📐 Especificaciones del Tanque", expanded=True):
r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
sp_nivel = st.slider("Consigna de Nivel (Setpoint) [m]", 0.2, float(h_total)-0.2, float(h_total/2))

with st.sidebar.expander("🚰 Válvulas de Control", expanded=True):
q_max = st.number_input("Flujo máximo por válvula [m³/s]", value=2.0, min_value=0.5, max_value=5.0, step=0.5)

with st.sidebar.expander("🛡️ Escenario de Perturbación ($Q_p$)", expanded=True):
p_activa = st.toggle("Simular Falla/Fuga Externas", value=True)

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
with st.sidebar.expander("📊 Cargar Datos Experimentales", expanded=False):
st.write("Ingresa los datos medidos en el laboratorio:")
st.caption("⚠️ Nota: El nivel debe ingresarse en **centímetros (cm)**")

df_exp_default = pd.DataFrame({
"Tiempo (s)": [0, 60, 120, 180, 240, 300],
"Nivel Medido (cm)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
})

datos_usr = st.data_editor(df_exp_default, num_rows="dynamic", key="datos_exp")
mostrar_ref = st.checkbox("Mostrar referencia en gráfica", value=True)

if st.button("🧮 Calcular Cd", use_container_width=True):
if not isinstance(datos_usr, pd.DataFrame):
datos_usr = pd.DataFrame(datos_usr)

if "Nivel Medido (cm)" in datos_usr.columns and len(datos_usr) >= 2:
df_calib = datos_usr.copy()
df_calib["Nivel Medido (m)"] = df_calib["Nivel Medido (cm)"] / 100
area_ori_temp = np.pi * ((0.0254) / 2)**2
cd_calculado = calcular_cd_inteligente(
    df_calib[["Tiempo (s)", "Nivel Medido (m)"]], 
    r_max, h_total, geom_tanque, area_ori_temp
)
st.session_state['cd_calculado'] = cd_calculado
st.success(f"✅ Cd calculado: {cd_calculado:.4f}")
else:
st.warning("⚠️ Ingresa al menos 2 datos para calcular Cd")
st.session_state['cd_calculado'] = 0.61

with st.sidebar.expander("🎛️ Parámetros del Controlador PID", expanded=True):
cd_actual = st.session_state.get('cd_calculado', 0.61)
kp_sug, ki_sug, kd_sug = sintonizar_controlador_robusto(geom_tanque, r_max, h_total, cd_actual, q_max)

modo_auto = st.checkbox("🎯 Modo Robusto (Auto-sintonía optimizada)", value=True)

st.markdown("---")
st.subheader("🎛️ Configuración")

if modo_auto:
st.success(f"💡 PID optimizado (Cd={cd_actual:.3f})")
st.caption(f"Kp={kp_sug} | Ki={ki_sug} | Kd={kd_sug}")
kp_val = st.number_input("Kp (robusto)", value=kp_sug, key="kp_auto")
ki_val = st.number_input("Ki (robusto)", value=ki_sug, format="%.3f", key="ki_auto")
kd_val = st.number_input("Kd (robusto)", value=kd_sug, format="%.3f", key="kd_auto")
st.caption("✅ Parámetros optimizados para rechazar perturbaciones")
else:
st.info("✍️ Modo Manual - Valores recomendados: Kp=10, Ki=2.0, Kd=0.5")
kp_val = st.number_input("Kp", value=10.0, step=1.0, key="kp_man")
ki_val = st.number_input("Ki", value=2.0, step=0.5, format="%.3f", key="ki_man")
kd_val = st.number_input("Kd", value=0.5, step=0.1, format="%.3f", key="kd_man")

tiempo_ensayo = st.slider("Tiempo de simulación [s]", 60, 600, 300)


# =============================================================================
# BIBLIOTECA Y BOTONES
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Biblioteca Técnica")

with st.sidebar.container(border=True):
nombre_pdf = "Guia_Practica_UCV.pdf"
if os.path.exists(nombre_pdf):
with open(nombre_pdf, "rb") as f:
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
iniciar_sim = st.button("▶️ Iniciar Simulación Robusta", use_container_width=True, type="primary")
with col_btn2:
btn_reset = st.button("🔄 Reset", use_container_width=True, type="secondary")

if btn_reset:
st.session_state.ejecutando = False
st.session_state['cd_calculado'] = 0.61
st.rerun()


# =============================================================================
# INICIALIZACIÓN DE SIMULACIÓN
# =============================================================================
if iniciar_sim:
st.session_state.ejecutando = True

st.session_state['error_acumulado'] = 0.0
st.session_state['ultimo_error'] = 0.0

try:
if not isinstance(datos_usr, pd.DataFrame):
datos_usr = pd.DataFrame(datos_usr)

cd_para_usar = st.session_state.get('cd_calculado', 0.61)

if modo_auto:
kp_a, ki_a, kd_a = sintonizar_controlador_robusto(
    geom_tanque, r_max, h_total, cd_para_usar, q_max
)
st.session_state['kp_ejecucion'] = kp_a
st.session_state['ki_ejecucion'] = ki_a
st.session_state['kd_ejecucion'] = kd_a
st.session_state['cd_final'] = cd_para_usar
st.toast(f"🎯 Control Robusto Activado: Cd={cd_para_usar:.2f} | Kp={kp_a} | Ki={ki_a} | Kd={kd_a}")
else:
st.session_state['kp_ejecucion'] = kp_val
st.session_state['ki_ejecucion'] = ki_val
st.session_state['kd_ejecucion'] = kd_val
st.session_state['cd_final'] = cd_para_usar
st.info(f"✍️ Modo Manual: Kp={kp_val}, Ki={ki_val}, Kd={kd_val}")

except Exception as e:
st.session_state['kp_ejecucion'] = 12.0
st.session_state['ki_ejecucion'] = 2.5
st.session_state['kd_ejecucion'] = 0.8
st.session_state['cd_final'] = 0.61
st.warning(f"⚠️ Usando configuración robusta de emergencia")


# =============================================================================
# SIMULACIÓN PRINCIPAL
# =============================================================================
if not st.session_state.ejecutando:
st.info("💡 Ajuste los parámetros en la barra lateral y presione 'Iniciar Simulación Robusta' para comenzar.")
else:
col_graf, col_met = st.columns([2, 1])

with col_graf:
st.subheader("🎮 Monitor del Proceso - Control PID con Dos Válvulas")
placeholder_tanque = st.empty()
st.subheader("📈 Tendencia Temporal")
placeholder_grafico = st.empty()
st.subheader("🔧 Acción de las Válvulas de Control")
placeholder_valvulas = st.empty()
st.markdown("---")
st.subheader("📊 Comparativa: Modelo Teórico vs Datos Experimentales")
placeholder_comparativa = st.empty()

with col_met:
st.subheader("📊 Métricas de Control Robusto")

kp_show = st.session_state.get('kp_ejecucion', 12.0)
ki_show = st.session_state.get('ki_ejecucion', 2.5)
cd_show = st.session_state.get('cd_final', 0.61)

st.write(f"**Parámetros Activos (Robustos):**")
st.caption(f"Kp: {kp_show} | Ki: {ki_show} | Kd: {st.session_state.get('kd_ejecucion', 0.8)}")
st.caption(f"Cd: {cd_show:.3f} | Qmax: {q_max} m³/s | {p_tipo}")
st.markdown("---")

placeholder_iae = st.empty()
placeholder_itae = st.empty()

placeholder_iae.metric("IAE (Error Acumulado)", "0.00")
placeholder_itae.metric("ITAE (Criterio Tesis)", "0.00")

st.markdown("---")

m_h = st.empty()
m_e = st.empty()
m_qin = st.empty()
m_qout = st.empty()
m_h.metric("Nivel PV [m]", "0.000")
m_e.metric("Error [m]", "0.000")
m_qin.metric("Flujo Entrada [m³/s]", "0.000")
m_qout.metric("Flujo Salida [m³/s]", "0.000")

st.markdown("---")
st.caption("💡 El controlador regula el nivel usando ambas válvulas")

# Preparación
status_placeholder = st.empty()
dt = 1.0
vector_t = np.arange(0, tiempo_ensayo, dt)
h_log, qin_log, qout_log, e_log = [], [], [], []

h_corrida = 0.5  # Nivel inicial

valor_presente = h_corrida
error_presente = 0.0
err_int, err_pasado = 0.0, 0.0
iae_acumulado, itae_acumulado = 0.0, 0.0

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

k_p = st.session_state.get('kp_ejecucion', 12.0)
k_i = st.session_state.get('ki_ejecucion', 2.5)
k_d = st.session_state.get('kd_ejecucion', 0.8)

# Bucle de simulación
for i, t_act in enumerate(vector_t):
status_placeholder.markdown("<div class='flow-indicator'>💧 CONTROL PID ACTIVO - PROCESANDO...</div>", unsafe_allow_html=True)

if p_activa and t_act >= p_tiempo:
q_p_inst = p_magnitud
else:
q_p_inst = 0.0

h_corrida, q_entrada, q_salida, e_inst, err_int, err_pasado = resolver_sistema_dos_valvulas(
dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst, p_tipo,
err_int, err_pasado, k_p, k_i, k_d, q_max
)

valor_presente = h_corrida
error_presente = e_inst
iae_acumulado += abs(e_inst) * dt
itae_acumulado += (t_act * abs(e_inst)) * dt

h_log.append(h_corrida)
qin_log.append(q_entrada)
qout_log.append(q_salida)
e_log.append(e_inst)

m_h.metric("Nivel PV [m]", f"{valor_presente:.3f}")
m_e.metric("Error [m]", f"{error_presente:.4f}")
m_qin.metric("Flujo Entrada [m³/s]", f"{q_entrada:.3f}")
m_qout.metric("Flujo Salida [m³/s]", f"{q_salida:.3f}")
placeholder_iae.metric("IAE (Error Acumulado)", f"{iae_acumulado:.2f}")
placeholder_itae.metric("ITAE (Criterio Tesis)", f"{itae_acumulado:.2f}")

# Visualización del tanque (estética original conservada)
fig_t, ax_t = plt.subplots(figsize=(7, 5))
ax_t.set_axis_off()
ax_t.set_xlim(-r_max*3, r_max*3)
ax_t.set_ylim(-0.8, h_total*1.3)

if abs(e_inst) < 0.05:
color_agua = '#27ae60'
elif abs(e_inst) < 0.15:
color_agua = '#f39c12'
else:
color_agua = '#e74c3c'

if geom_tanque == "Cilíndrico":
c_in_x, c_in_y = -r_max, h_total*0.8
c_out_x, c_out_y = r_max, 0.1
ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=5, zorder=2)
ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, valor_presente, color=color_agua, alpha=0.85, zorder=1, edgecolor='#2980b9', linewidth=1.5))
if valor_presente > 0 and valor_presente < h_total:
    ax_t.axhline(y=valor_presente, color='white', linestyle='-', linewidth=2, alpha=0.8, zorder=3)
if q_entrada > 0:
    ax_t.annotate('', xy=(-r_max-1.5, valor_presente*0.7), xytext=(-r_max-0.3, valor_presente*0.7),
                arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
if q_salida > 0:
    ax_t.annotate('', xy=(r_max+1.5, 0.3), xytext=(r_max+0.3, 0.3),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))

elif geom_tanque == "Cónico":
c_in_x, c_in_y = -(r_max/h_total)*(h_total*0.8), h_total*0.8
c_out_x, c_out_y = 0, 0
ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=5, zorder=2)
if valor_presente > 0:
    radio_superficie = (r_max / h_total) * valor_presente
    vertices = [[-radio_superficie, valor_presente], [radio_superficie, valor_presente], [0, 0]]
    ax_t.add_patch(plt.Polygon(vertices, color=color_agua, alpha=0.85, zorder=1, edgecolor='#2980b9', linewidth=1.5))
    ax_t.plot([-radio_superficie, radio_superficie], [valor_presente, valor_presente], color='white', linewidth=2, alpha=0.8, zorder=3)

else:  # Esférico
import math
c_in_y = h_total * 0.7
c_in_x = -math.sqrt(abs(r_max**2 - (c_in_y - r_max)**2))
c_out_x, c_out_y = 0, 0
agua_esf = plt.Circle((0, r_max), r_max, color=color_agua, alpha=0.85, zorder=1, edgecolor='#2980b9', linewidth=1.5)
ax_t.add_patch(agua_esf)
recorte_nivel = plt.Rectangle((-r_max, 0), 2*r_max, valor_presente, transform=ax_t.transData)
agua_esf.set_clip_path(recorte_nivel)
ax_t.add_patch(plt.Circle((0, r_max), r_max, color='#2c3e50', fill=False, lw=5, zorder=2))
if valor_presente > 0 and valor_presente < 2*r_max:
    radio_nivel = math.sqrt(r_max**2 - (valor_presente - r_max)**2)
    ax_t.plot([-radio_nivel, radio_nivel], [valor_presente, valor_presente], color='white', linewidth=2, alpha=0.8, zorder=3)

# Tuberías y válvulas (estética original)
ax_t.add_patch(plt.Rectangle((c_in_x - 1.5, c_in_y - 0.1), 1.5, 0.2, color='silver', zorder=0))
ax_t.add_patch(plt.Polygon([[c_in_x-1, c_in_y+0.2], [c_in_x-1, c_in_y-0.2], [c_in_x-0.6, c_in_y]], color='#2c3e50', zorder=2))
ax_t.add_patch(plt.Polygon([[c_in_x-0.2, c_in_y+0.2], [c_in_x-0.2, c_in_y-0.2], [c_in_x-0.6, c_in_y]], color='#2c3e50', zorder=2))
ax_t.text(c_in_x-0.6, c_in_y+0.4, "V-01", ha='center', fontsize=9, fontweight='bold', color='blue')

if geom_tanque == "Cilíndrico":
ax_t.add_patch(plt.Rectangle((c_out_x, c_out_y - 0.1), 1.5, 0.2, color='silver', zorder=0))
vs_x, vs_y = c_out_x + 0.8, c_out_y
else:
ax_t.add_patch(plt.Rectangle((c_out_x - 0.1, -0.6), 0.2, 0.6, color='silver', zorder=0))
vs_x, vs_y = c_out_x, -0.4

ax_t.add_patch(plt.Polygon([[vs_x-0.25, vs_y+0.2], [vs_x-0.25, vs_y-0.2], [vs_x, vs_y]], color='#2c3e50', zorder=2))
ax_t.add_patch(plt.Polygon([[vs_x+0.25, vs_y+0.2], [vs_x+0.25, vs_y-0.2], [vs_x, vs_y]], color='#2c3e50', zorder=2))
offset_t = 0.4 if geom_tanque == "Cilíndrico" else 0
ax_t.text(vs_x + offset_t, vs_y - 0.5, "V-02", ha='center', fontsize=9, fontweight='bold', color='red')

ax_t.axhline(y=sp_nivel, color='red', ls='--', lw=2, zorder=3, alpha=0.8)
ax_t.text(-r_max*2.8, sp_nivel + 0.05, f"SETPOINT: {sp_nivel:.2f}m", color='red', fontweight='bold', fontsize=9)
ax_t.text(0, h_total * 1.2, f"PV: {valor_presente:.3f} m", 
     ha='center', va='center', fontsize=11, fontweight='bold',
     bbox=dict(facecolor='white', alpha=0.9, edgecolor='#1a5276', boxstyle='round,pad=0.5', lw=2))

if p_activa and t_act >= p_tiempo:
ax_t.text(0, -0.5, f"⚠️ PERTURBACIÓN ACTIVA ({p_tipo})", ha='center', color='orange', fontweight='bold', fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange', boxstyle='round'))

placeholder_tanque.pyplot(fig_t)
plt.close(fig_t)

# Gráfico de tendencia
fig_tr, ax_tr = plt.subplots(figsize=(8, 3.5))
ax_tr.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2, label='Nivel (h) - Control PID')
ax_tr.axhline(y=sp_nivel, color='red', ls='--', alpha=0.5, label='Setpoint')
if p_activa and p_tiempo > 0 and t_act >= p_tiempo:
ax_tr.axvspan(p_tiempo, tiempo_ensayo, alpha=0.1, color='orange', label='Zona con Perturbación')
ax_tr.set_xlabel('Tiempo [s]', fontsize=10, fontweight='bold')
ax_tr.set_ylabel('Altura [m]', fontsize=10, fontweight='bold')
ax_tr.legend(loc='upper right', fontsize='x-small')
ax_tr.set_xlim(0, tiempo_ensayo)
ax_tr.set_ylim(0, h_total * 1.1)
ax_tr.grid(True, alpha=0.2)
placeholder_grafico.pyplot(fig_tr)
plt.close(fig_tr)

# Gráfico de válvulas
fig_v, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
ax1.step(vector_t[:i+1], qin_log, where='post', color='blue', lw=2)
ax1.set_ylabel('Q entrada [m³/s]', fontsize=9)
ax1.set_ylim(0, q_max * 1.1)
ax1.grid(True, alpha=0.2)
ax1.set_title('Válvula de Entrada (V-01)', fontsize=10, fontweight='bold')

ax2.step(vector_t[:i+1], qout_log, where='post', color='red', lw=2)
ax2.set_ylabel('Q salida [m³/s]', fontsize=9)
ax2.set_xlabel('Tiempo [s]', fontsize=10, fontweight='bold')
ax2.set_ylim(0, q_max * 1.1)
ax2.grid(True, alpha=0.2)
ax2.set_title('Válvula de Salida (V-02)', fontsize=10, fontweight='bold')

plt.tight_layout()
placeholder_valvulas.pyplot(fig_v)
plt.close(fig_v)

# Gráfico comparativo
fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
ax_comp.plot(vector_t[:i+1], h_log, color='#1f77b4', lw=2, label='Simulación Robusta')
if mostrar_ref and tiene_datos_exp and len(t_exp) > 0:
ax_comp.scatter(t_exp, h_exp, color='red', marker='x', s=100, label='Datos Experimentales')
ax_comp.plot(t_exp, h_exp, color='red', linestyle='--', alpha=0.3)
ax_comp.set_title("Validación de Resultados - Control PID con Dos Válvulas", fontsize=10, fontweight='bold')
ax_comp.set_xlabel("Tiempo [s]")
ax_comp.set_ylabel("Nivel [m]")
ax_comp.set_ylim(0, h_total * 1.1)
ax_comp.grid(True, alpha=0.3)
ax_comp.legend(loc='lower right')
placeholder_comparativa.pyplot(fig_comp)
plt.close(fig_comp)

time.sleep(0.02)
barra_p.progress((i+1)/len(vector_t))

status_placeholder.empty()
st.success(f"✅ Simulación Robusta completada - El controlador mantuvo el nivel ante las perturbaciones")
st.balloons()

# Análisis final
st.markdown("---")
st.subheader("📈 Análisis de Respuesta - Control PID con Dos Válvulas")

col_an1, col_an2 = st.columns([2, 1])

with col_an1:
fig_amp, ax_amp = plt.subplots(figsize=(10, 5))
ax_amp.plot(vector_t, h_log, color='#1f77b4', lw=2.5, label='Respuesta del Sistema (PV)')
ax_amp.axhline(y=sp_nivel, color='#d62728', linestyle='--', lw=2, label='Referencia (SP)')
if p_activa and p_tiempo > 0:
ax_amp.axvline(x=p_tiempo, color='orange', linestyle='--', alpha=0.7, label='Inicio Perturbación')
ax_amp.axvspan(p_tiempo, tiempo_ensayo, alpha=0.08, color='orange', label='Zona con Perturbación Activa')
ax_amp.set_title("Respuesta Transitoria del Lazo de Control", fontsize=12)
ax_amp.set_xlabel("Tiempo (s)")
ax_amp.set_ylabel("Amplitud (m)")
ax_amp.grid(True, which='both', linestyle='--', alpha=0.5)
ax_amp.legend(loc='lower right')

if len(h_log) > 0:
error_f_val = abs(h_log[-1] - sp_nivel)
if error_f_val < 0.05:
    ax_amp.axhspan(sp_nivel-0.05, sp_nivel+0.05, color='green', alpha=0.1, label='Banda de Estabilidad (±5%)')

st.pyplot(fig_amp)
plt.close(fig_amp)

with col_an2:
st.info("**Interpretación del Control:**")
sobrepico = ((max(h_log) - sp_nivel) / sp_nivel) * 100 if max(h_log) > sp_nivel else 0
st.metric("Sobrepico Máximo", f"{sobrepico:.2f} %")
st.metric("IAE Final", f"{iae_acumulado:.2f}")
st.metric("ITAE Final", f"{itae_acumulado:.2f}")

if p_activa and p_tiempo > 0:
idx_pert = int(p_tiempo / dt) if p_tiempo < len(h_log) else 0
if idx_pert < len(h_log) - 10:
    error_post_pert = max([abs(h_log[j] - sp_nivel) for j in range(idx_pert, min(idx_pert+50, len(h_log)))])
    st.metric("Máximo Error tras Perturbación", f"{error_post_pert:.4f} m")
    if error_post_pert < 0.05:
        st.success("✅ Excelente rechazo a perturbaciones")
    elif error_post_pert < 0.1:
        st.info("👍 Buen rechazo a perturbaciones")
    else:
        st.warning("⚠️ El rechazo puede mejorar ajustando Ki")

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
st.success("✅ Error residual prácticamente nulo - Control excelente")
elif err_f < 0.05:
st.info("👍 Error residual aceptable")
else:
st.warning("⚠️ Aumentar Ki para eliminar el offset")

# Exportar datos
df_final = pd.DataFrame({
"Tiempo [s]": vector_t,
"Nivel [m]": h_log,
"Q_entrada [m3/s]": qin_log,
"Q_salida [m3/s]": qout_log,
"Error [m]": e_log,
"Kp_Usado": [k_p] * len(vector_t),
"Ki_Usado": [k_i] * len(vector_t),
"Kd_Usado": [k_d] * len(vector_t),
"Cd_Usado": [cd_para_simular] * len(vector_t)
})

col_down1, col_down2, col_down3 = st.columns([1, 2, 1])
with col_down2:
st.download_button(
label="📥 Descargar Reporte de Simulación Robusta (CSV)",
data=df_final.to_csv(index=False),
file_name=f"resultados_robustos_{geom_tanque}.csv",
mime="text/csv",
use_container_width=True
)

if err_f < 0.05:
st.success(f"✅ El controlador mantuvo el sistema en {h_log[-1]:.3f} m (error < 5%)")
else:
st.warning(f"⚠️ Error residual de {err_f:.3f} m. Aumente Ki para eliminarlo.")


# =============================================================================
# FOOTER INSTITUCIONAL
# =============================================================================
st.markdown("""
<hr style="margin: 2rem 0 1rem 0; border-color: #1a5276;">
<div style="text-align: center; color: #5d6d7e; font-size: 0.8rem;">
<p>Universidad Central de Venezuela - Escuela de Ingeniería Química</p>
<p>Simulador de Control PID con Dos Válvulas | Anti-Perturbaciones | © 2025</p>
<p style="font-size: 0.7rem;">Sintonía optimizada para rechazo de fugas y cambios de carga</p>
</div>
""", unsafe_allow_html=True)
