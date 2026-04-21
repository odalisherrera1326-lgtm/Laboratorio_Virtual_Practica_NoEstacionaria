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


def calcular_q_max_automatico(geom, r, h_t, d_orificio_pulg):
    """
    Calcula automáticamente el flujo máximo basado en el volumen del tanque 
    y el diámetro del orificio de salida.
    """
    if geom == "Cilíndrico":
        volumen = np.pi * (r**2) * h_t
    elif geom == "Cónico":
        volumen = (1/3) * np.pi * (r**2) * h_t
    else:  # Esférico
        volumen = (4/3) * np.pi * (r**3)
    
    # Área del orificio
    d_metros = d_orificio_pulg * 0.0254
    area_orificio = np.pi * (d_metros / 2)**2
    
    # Flujo máximo proporcional al volumen y al área del orificio
    q_max = volumen * area_orificio * 500
    q_max = np.clip(q_max, 0.5, 5.0)
    
    return round(float(q_max), 2)
def calcular_q_max_salida(d_orificio_pulg, cd=0.61, h_max=10.0):
    """
    Calcula el caudal máximo de salida basado en el orificio.
    Usa la Ley de Torricelli: Q = Cd * A * sqrt(2 * g * h)
    """
    g = 9.81  # m/s²
    d_metros = d_orificio_pulg * 0.0254
    area_orificio = np.pi * (d_metros / 2)**2
    
    # Caudal máximo teórico (con altura máxima)
    q_max_salida = cd * area_orificio * np.sqrt(2 * g * h_max)
    
    return round(float(q_max_salida), 4)

def calcular_cd_automatico(geom, d_orificio_pulg):
    """
    Calcula un Cd automático basado en la geometría y el diámetro del orificio.
    Valores típicos de Cd para orificios:
    - Orificio pequeño (borde afilado): 0.60 - 0.62
    - Orificio grande: 0.65 - 0.70
    - Cono/Esfera: ligeramente menor por resistencia
    """
    # Cd base según geometría
    if geom == "Cilíndrico":
        cd_base = 0.61
    elif geom == "Cónico":
        cd_base = 0.58
    else:  # Esférico
        cd_base = 0.55
    
    # Ajuste por diámetro del orificio (orificio más grande = mayor Cd)
    factor_diametro = np.clip(d_orificio_pulg / 1.0, 0.9, 1.1)
    
    cd_final = cd_base * factor_diametro
    
    return round(float(np.clip(cd_final, 0.45, 0.75)), 4)

def calcular_cd_desde_datos(df_usr, r, h_t, geom, d_orificio_pulg):
    """
    Calcula el Coeficiente de Descarga (Cd) usando datos experimentales.
    Usa el balance de masa: Q = A * dh/dt
    """
    df = pd.DataFrame(df_usr) if isinstance(df_usr, list) else df_usr
    
    if len(df) < 2:
        return 0.61
    
    try:
        t1, t2 = df["Tiempo (s)"].iloc[0], df["Tiempo (s)"].iloc[1]
        h1, h2 = df["Nivel Medido (cm)"].iloc[0] / 100, df["Nivel Medido (cm)"].iloc[1] / 100
        dt = abs(t2 - t1)
        
        if dt == 0:
            return 0.61
        
        if geom == "Cilíndrico":
            v1, v2 = np.pi * (r**2) * h1, np.pi * (r**2) * h2
        elif geom == "Cónico":
            v1 = (1/3) * np.pi * ((r/h_t) * h1)**2 * h1
            v2 = (1/3) * np.pi * ((r/h_t) * h2)**2 * h2
        else:
            v1 = (np.pi * (h1**2) / 3) * (3*r - h1)
            v2 = (np.pi * (h2**2) / 3) * (3*r - h2)
        
        q_real = abs(v1 - v2) / dt
        h_prom = (h1 + h2) / 2
        
        d_metros = d_orificio_pulg * 0.0254
        area_ori = np.pi * (d_metros / 2)**2
        q_teorico = area_ori * np.sqrt(2 * 9.81 * max(h_prom, 0.001))
        
        cd_result = q_real / q_teorico if q_teorico > 0 else 0.61
        return float(np.clip(cd_result, 0.4, 1.0))
    except:
        return 0.61
def sintonizar_controlador_robusto(geom, r, h_t, cd_calculado=0.61, q_max_bomba=2.0, tipo_proceso="Llenado"):
    """Sintonización robusta del PID CORREGIDA con ganancias más altas"""
    if geom == "Cilíndrico":
        area_t = np.pi * (r**2)
    elif geom == "Cónico":
        area_t = np.pi * (r/2)**2
    else:
        area_t = (2/3) * np.pi * (r**2)
    
    # Ganancias base MÁS ALTAS
    if tipo_proceso == "Llenado":
        kp = 25.0 * (area_t / 3.0)
        ki = 5.0 * (area_t / 3.0)
        kd = 2.0 * (area_t / 3.0)
    else:
        kp = 20.0 * (area_t / 3.0)
        ki = 4.0 * (area_t / 3.0)
        kd = 1.5 * (area_t / 3.0)
    
    factor_cd = np.clip(cd_calculado / 0.61, 0.8, 1.3)
    kp = kp * factor_cd
    ki = ki * factor_cd
    
    kp = np.clip(kp, 15.0, 50.0)
    ki = np.clip(ki, 3.0, 10.0)
    kd = np.clip(kd, 1.0, 3.0)
    
    return round(kp, 2), round(ki, 3), round(kd, 2)

def resolver_sistema_dos_valvulas(dt, h_prev, sp, geom, r, h_t, q_p_val, p_tipo, e_sum, e_prev, kp, ki, kd, q_max_bomba, q_max_salida, cd_val, d_orificio_pulg):
    """
    Sistema CORREGIDO - Físicamente correcto:
    - V-01 (Entrada): Controla flujo de bomba (0 a Qmax_bomba)
    - V-02 (Salida): Controla flujo por orificio (Ley de Torricelli)
    """
    
    area_h = get_area_transversal(geom, r, h_prev, h_t)
    area_h = max(area_h, 0.0001)
    
    err = sp - h_prev
    
    # Acciones PID
    P = kp * err
    e_sum += err * dt
    I = ki * e_sum
    
    if dt > 0:
        D = kd * (err - e_prev) / dt
    else:
        D = 0.0
    
    u_control = P + I + D
    
    # =========================================================================
    # CONTROL DE VÁLVULA DE ENTRADA (BOMBA)
    # =========================================================================
    flujo_base_bomba = q_max_bomba * 0.15
    
    if err > 0.01:  # Nivel BAJO - Necesito SUBIR
        q_entrada = flujo_base_bomba + np.clip(u_control, 0, q_max_bomba - flujo_base_bomba)
        apertura_salida = 0.3  # Válvula de salida casi cerrada (30% abierta)
    elif err < -0.01:  # Nivel ALTO - Necesito BAJAR
        q_entrada = flujo_base_bomba * 0.3  # Bomba al 30%
        apertura_salida = 1.0 + np.clip(-u_control / q_max_bomba, 0, 1.0)
        apertura_salida = np.clip(apertura_salida, 0.3, 1.0)
    else:  # En el setpoint - Equilibrio
        q_entrada = flujo_base_bomba
        apertura_salida = 0.5  # Válvula al 50%
    
    q_entrada = np.clip(q_entrada, 0, q_max_bomba)
    
    # =========================================================================
    # CONTROL DE VÁLVULA DE SALIDA (ORIFICIO - LEY DE TORRICELLI)
    # =========================================================================
    g = 9.81
    d_metros = d_orificio_pulg * 0.0254
    area_orificio = np.pi * (d_metros / 2)**2
    
    if h_prev > 0.001:
        q_salida_teorico = cd_val * area_orificio * np.sqrt(2 * g * h_prev)
    else:
        q_salida_teorico = 0.0
    
    q_salida = apertura_salida * q_salida_teorico
    
    # =========================================================================
    # AGREGAR PERTURBACIÓN
    # =========================================================================
    if p_tipo == "Entrada":
        q_entrada_total = q_entrada + q_p_val
        q_salida_total = q_salida
    else:
        q_entrada_total = q_entrada
        q_salida_total = q_salida + q_p_val
    
    # Balance de masa
    dh_dt = (q_entrada_total - q_salida_total) / area_h
    h_next = h_prev + dh_dt * dt
    h_next = np.clip(h_next, 0.0, h_t)
    
    return h_next, q_entrada, q_salida, err, e_sum, err
 


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

if 'cd_calculado' not in st.session_state:
    st.session_state['cd_calculado'] = 0.61

# Inicializar datos_usr si no existe
if 'datos_usr' not in st.session_state:
    st.session_state['datos_usr'] = pd.DataFrame({
        "Tiempo (s)": [0, 60, 120, 180, 240, 300],
        "Nivel Medido (cm)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })

# =============================================================================
# --- ESTILOS CSS MEJORADOS ---
# =============================================================================
st.markdown("""
<style>
/* Estilos generales */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4f8 0%, #e8edf2 100%);
    cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='%23333' stroke-width='1.5'><circle cx='12' cy='12' r='3'/><path d='M12 1v3M12 20v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M1 12h3M20 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12'/></svg>") 12 12, auto !important;
}

/* Tabs personalizadas */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background: linear-gradient(180deg, #1a5276 0%, #154360 100%);
    padding: 0.5rem 2rem;
    border-radius: 15px 15px 0 0;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 10px;
    padding: 0.5rem 2rem;
    font-size: 1.2rem;
    font-weight: bold;
    color: #f0f4f8 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(241, 196, 15, 0.2);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #f1c40f, #e67e22);
    color: #1a5276 !important;
}

/* Tarjetas de prácticas */
.practica-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 6px solid #f1c40f;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    cursor: pointer;
}

.practica-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 30px rgba(0,0,0,0.15);
}

.practica-card h3 {
    color: #1a5276;
    margin-bottom: 0.5rem;
}

.practica-card p {
    color: #5d6d7e;
    font-size: 0.9rem;
}

.practica-card .badge {
    display: inline-block;
    background: linear-gradient(90deg, #1a5276, #2471a3);
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    margin-top: 0.5rem;
}

/* Header institucional */
.header-container {
    background: linear-gradient(135deg, #0d3251 0%, #1a5276 50%, #1f618d 100%);
    background-size: 200% 200%;
    animation: gradientBG 8s ease infinite;
    border-radius: 20px;
    padding: 20px 25px;
    margin-bottom: 30px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Botones */
.stButton > button {
    background: linear-gradient(90deg, #1a5276, #2471a3) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #2471a3, #2e86c1) !important;
    transform: scale(1.02);
}

/* Sidebar cuando está visible */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a5276 0%, #154360 100%) !important;
    border-right: 4px solid #f1c40f !important;
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] p {
    color: #f0f4f8 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #5d6d7e;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #d4e6f1;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ENCABEZADO INSTITUCIONAL
# =============================================================================
def get_base64(path):
    # Intentar primero con la ruta directa
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    # Si no existe, intentar con rutas alternativas (para Streamlit Cloud)
    posibles_rutas = [
        path,
        os.path.join(os.path.dirname(__file__), path),
        os.path.join("/mount/src/laboratorio_virtual_practica_noestacionaria", path),
        os.path.join(os.getcwd(), path)
    ]
    
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            with open(ruta, "rb") as f:
                return base64.b64encode(f.read()).decode()
    
    # Si no se encuentra en ninguna ruta, retornar None
    return None
logo_ucv_64 = get_base64("logo_ucv.png")
logo_eiq_64 = get_base64("logoquimicaborde.png")

# Si no se encuentran las imágenes, mostrar texto
logo_ucv_html = f'<img src="data:image/png;base64,{logo_ucv_64}" width="100">' if logo_ucv_64 else '<h3 style="color: white;">UCV</h3>'
logo_eiq_html = f'<img src="data:image/png;base64,{logo_eiq_64}" width="150">' if logo_eiq_64 else '<h3 style="color: white;">EIQ</h3>'

st.markdown(f"""
<div class="header-container">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="width: 120px; text-align: center;">
            {logo_ucv_html}
        </div>
        <div style="text-align: center;">
            <h1 style="color: white !important; font-size: 2.2rem;">Práctica Virtual: Balance en estado no estacionario</h1>
            <p style="color: #d4e6f1 !important; margin: 0;">Escuela de Ingeniería Química | Facultad de Ingeniería - UCV</p>
        </div>
        <div style="width: 160px; text-align: center;">
            {logo_eiq_html}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MARCO TEÓRICO
# =============================================================================
col_teoria1, col_teoria2, col_teoria3 = st.columns(3)

with col_teoria1:
    with st.expander("📚 Fundamento teórico: Ecuaciones de Conservación", expanded=False):
        st.markdown(r"""
        La dinámica del sistema se describe mediante el **Balance Global de Masa**:
        
        $$ \frac{dV}{dt} = Q_{in} - Q_{out} \pm Q_{p} $$
        
        $$ A(h) \frac{dh}{dt} = Q_{in} - Q_{out} \pm Q_{p} $$
        
        Donde:
        * **$A(h)$**: Área transversal (m²)
        * **$Q_{in}$**: Flujo de entrada (m³/s)
        * **$Q_{out}$**: Flujo de salida (m³/s)
        * **$Q_{p}$**: Flujo de perturbación (m³/s)
        """)

with col_teoria2:
    with st.expander("🎮 Teoría: Control PID con Dos Válvulas", expanded=False):
        st.markdown(r"""
        El sistema utiliza un controlador **PID** que actúa sobre **DOS VÁLVULAS**:
        
        $$ u(t) = K_p e(t) + K_i \int e(t)dt + K_d \frac{de}{dt} $$
        
        **Estrategia bidireccional:**
        * **V-01 (Entrada):** $Q_{in}$ (0 a $Q_{max}$)
        * **V-02 (Salida):** $Q_{out}$ (0 a $Q_{max}$)
        
        **Lógica:**
        * $h < SP$ → Abrir V-01, cerrar V-02
        * $h > SP$ → Cerrar V-01, abrir V-02
        """)

with col_teoria3:
    with st.expander("📊 Criterios de Desempeño (IAE/ITAE)", expanded=False):
        st.markdown(r"""
        Métricas integrales del error $e(t) = SP - PV$:

        1. **IAE (Integral del Error Absoluto):**
        $$IAE = \int_{0}^{t} |e(t)| dt$$

        2. **ITAE (Integral del Tiempo por Error Absoluto):**
        $$ITAE = \int_{0}^{t} t \cdot |e(t)| dt$$
        Penaliza errores que duran mucho tiempo.
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


# =============================================================================
# BARRA LATERAL - PARÁMETROS
# =============================================================================
st.sidebar.header("⚙️ Configuración del Sistema")

with st.sidebar.container(border=True):
    tipo_proceso = st.sidebar.selectbox("Tipo de Proceso", ["Llenado", "Vaciado"])
    geom_tanque = st.sidebar.selectbox("Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander("📐 Especificaciones del Tanque", expanded=True):
    r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
    h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
    h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
    sp_nivel = st.slider("Consigna de Nivel (Setpoint) [m]", 0.2, float(h_total)-0.2, float(h_total/2))

with st.sidebar.expander("🚰 Bomba de Alimentación", expanded=True):
    q_max_bomba = st.number_input("Caudal máximo de bomba [m³/s]", value=2.0, min_value=0.5, max_value=5.0, step=0.5)
    st.caption("💡 Capacidad máxima de la bomba de entrada")

with st.sidebar.expander("🔧 Orificio de Salida", expanded=True):
    d_pulgadas = st.number_input("Diámetro del Orificio (pulgadas)", value=1.0, min_value=0.1, step=0.1)
    d_metros = d_pulgadas * 0.0254
    area_orificio = np.pi * (d_metros / 2)**2
    st.caption(f"Área calculada: {area_orificio:.6f} m²")

# Cálculo de Cd y Qmax_salida (FUERA del with, SIN indentación extra)
cd_automatico = calcular_cd_automatico(geom_tanque, d_pulgadas)
q_max_salida = calcular_q_max_salida(d_pulgadas, cd_automatico, h_total)
st.session_state['cd_calculado'] = cd_automatico
with st.sidebar.expander("📊 Datos Experimentales", expanded=False):
    st.write("Ingresa los datos medidos en laboratorio:")
    st.caption("⚠️ El nivel debe ingresarse en **centímetros (cm)**")
    
    df_exp_default = pd.DataFrame({
        "Tiempo (s)": [0, 60, 120, 180, 240, 300],
        "Nivel Medido (cm)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    datos_usr = st.data_editor(df_exp_default, num_rows="dynamic", key="datos_exp")
    mostrar_ref = st.checkbox("Mostrar referencia experimental en gráfica", value=True)
    
    col_cd1, col_cd2 = st.columns(2)
    with col_cd1:
        if st.button("🧮 Calcular Cd desde datos", use_container_width=True):
            if not isinstance(datos_usr, pd.DataFrame):
                datos_usr = pd.DataFrame(datos_usr)
            
            if "Nivel Medido (cm)" in datos_usr.columns and len(datos_usr) >= 2:
                cd_calculado = calcular_cd_desde_datos(
                    datos_usr, r_max, h_total, geom_tanque, d_pulgadas
                )
                st.session_state['cd_calculado'] = cd_calculado
                st.success(f"✅ Cd calculado: {cd_calculado:.4f}")
            else:
                st.warning("⚠️ Ingresa al menos 2 datos")
    with col_cd2:
        if st.button("🔄 Usar Cd teórico", use_container_width=True):
            st.session_state['cd_calculado'] = cd_automatico
            st.success(f"✅ Cd teórico: {cd_automatico:.4f}")

with st.sidebar.expander("📊 Parámetros Calculados Automáticamente", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Qmax Bomba", f"{q_max_bomba:.2f}")
    with col2:
        st.metric("Qmax Salida", f"{q_max_salida:.4f}")
    with col3:
        st.metric("Cd", f"{cd_automatico:.4f}")
    
    ajuste_manual = st.checkbox("Ajuste manual de parámetros", value=False)
    if ajuste_manual:
        q_max_bomba = st.number_input("Qmax Bomba Manual [m³/s]", value=q_max_bomba, min_value=0.5, max_value=5.0, step=0.5)
        cd_manual = st.number_input("Cd Manual", value=cd_automatico, min_value=0.30, max_value=0.90, step=0.01, format="%.4f")
        st.session_state['cd_calculado'] = cd_manual

with st.sidebar.expander("🛡️ Escenario de Perturbación ($Q_p$)", expanded=True):
    
    p_activa = st.toggle("Simular Falla/Fuga Externas", value=True)
    
    if p_activa:
        p_tipo = st.selectbox("Tipo de Perturbación", ["Entrada", "Salida (Fuga)"])
        p_tipo = "Entrada" if p_tipo == "Entrada" else "Salida"
        p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.5, min_value=0.1, max_value=3.0, step=0.1, format="%.2f")
        p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 100)
    else:
        p_magnitud = 0.0
        p_tiempo = 0
        p_tipo = "Entrada"

with st.sidebar.expander("🎛️ Parámetros del Controlador PID", expanded=True):
    cd_actual = st.session_state.get('cd_calculado', 0.61)
    kp_sug, ki_sug, kd_sug = sintonizar_controlador_robusto(
        geom_tanque, r_max, h_total, cd_actual, q_max_bomba, tipo_proceso
    )
    
    modo_auto = st.checkbox("🎯 Modo Robusto (Auto-sintonía)", value=True)
    
    st.markdown("---")
    
    if modo_auto:
        st.success(f"💡 PID optimizado para {tipo_proceso}")
        st.caption(f"Kp={kp_sug} | Ki={ki_sug} | Kd={kd_sug}")
        kp_val = st.number_input("Kp", value=kp_sug, key="kp_auto")
        ki_val = st.number_input("Ki", value=ki_sug, format="%.3f", key="ki_auto")
        kd_val = st.number_input("Kd", value=kd_sug, format="%.3f", key="kd_auto")
    else:
        st.info(f"✍️ Modo Manual - {tipo_proceso}")
        if tipo_proceso == "Llenado":
            kp_default, ki_default, kd_default = 12.0, 2.5, 0.8
        else:
            kp_default, ki_default, kd_default = 8.0, 1.5, 0.5
        kp_val = st.number_input("Kp", value=kp_default, step=1.0, key="kp_man")
        ki_val = st.number_input("Ki", value=ki_default, step=0.5, format="%.3f", key="ki_man")
        kd_val = st.number_input("Kd", value=kd_default, step=0.1, format="%.3f", key="kd_man")
    
    tiempo_ensayo = st.slider("Tiempo de simulación [s]", 60, 600, 300)


# =============================================================================
# BOTONES
# =============================================================================
st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    iniciar_sim = st.button("▶️ Iniciar Simulación", use_container_width=True, type="primary")
with col_btn2:
    btn_reset = st.button("🔄 Reset", use_container_width=True, type="secondary")

if btn_reset:
    st.session_state.ejecutando = False
    st.rerun()


# =============================================================================
# INICIALIZACIÓN DE SIMULACIÓN
# =============================================================================
if iniciar_sim:
    st.session_state.ejecutando = True
    st.session_state['error_acumulado'] = 0.0
    st.session_state['ultimo_error'] = 0.0
    
    cd_para_usar = st.session_state.get('cd_calculado', 0.61)
    
    if modo_auto:
        kp_a, ki_a, kd_a = sintonizar_controlador_robusto(
            geom_tanque, r_max, h_total, cd_para_usar, q_max_bomba, tipo_proceso
        )
        st.session_state['kp_ejecucion'] = kp_a
        st.session_state['ki_ejecucion'] = ki_a
        st.session_state['kd_ejecucion'] = kd_a
        st.session_state['cd_final'] = cd_para_usar
        st.toast(f"🎯 Control Robusto ({tipo_proceso}) | Qmax={q_max_bomba:.2f} | Cd={cd_para_usar:.4f}")
    else:
        st.session_state['kp_ejecucion'] = kp_val
        st.session_state['ki_ejecucion'] = ki_val
        st.session_state['kd_ejecucion'] = kd_val
        st.session_state['cd_final'] = cd_para_usar
        st.info(f"✍️ Modo Manual ({tipo_proceso}) | Qmax={q_max_bomba:.2f} | Cd={cd_para_usar:.4f}")


# =============================================================================
# SIMULACIÓN PRINCIPAL
# =============================================================================
if not st.session_state.ejecutando:
    st.info("💡 Configure los parámetros y presione 'Iniciar Simulación'")
else:
    col_graf, col_met = st.columns([2, 1])

    with col_graf:
        st.subheader(f"🎮 Monitor del Proceso - Control PID ({tipo_proceso})")
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
        
        kp_show = st.session_state.get('kp_ejecucion', 12.0)
        ki_show = st.session_state.get('ki_ejecucion', 2.5)
        cd_show = st.session_state.get('cd_final', 0.61)
        
        st.write(f"**Parámetros Activos:**")
        st.caption(f"Proceso: {tipo_proceso} | Qmax: {q_max_bomba:.2f} m³/s | Cd: {cd_show:.4f}")
        st.caption(f"Kp: {kp_show} | Ki: {ki_show} | Kd: {st.session_state.get('kd_ejecucion', 0.8)}")
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

    # Preparación
    status_placeholder = st.empty()
    dt = 1.0
    vector_t = np.arange(0, tiempo_ensayo, dt)
    h_log, qin_log, qout_log, e_log = [], [], [], []

    if tipo_proceso == "Llenado":
        h_corrida = 0.2
    else:
        h_corrida = h_total * 0.9
    
    err_int, err_pasado = 0.0, 0.0
    iae_acumulado, itae_acumulado = 0.0, 0.0
    
    barra_p = st.progress(0)
    cd_para_simular = st.session_state.get('cd_final', 0.61)
    
    k_p = st.session_state.get('kp_ejecucion', 12.0)
    k_i = st.session_state.get('ki_ejecucion', 2.5)
    k_d = st.session_state.get('kd_ejecucion', 0.8)
    
    # Bucle de simulación
    for i, t_act in enumerate(vector_t):
        status_placeholder.markdown("<div class='flow-indicator'>💧 CONTROL PID ACTIVO</div>", unsafe_allow_html=True)
        
        if p_activa and t_act >= p_tiempo:
            q_p_inst = p_magnitud
        else:
            q_p_inst = 0.0
            
        h_corrida, q_entrada, q_salida, e_inst, err_int, err_pasado = resolver_sistema_dos_valvulas(
    dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst, p_tipo,
    err_int, err_pasado, k_p, k_i, k_d, q_max_bomba, q_max_salida, cd_para_simular, d_pulgadas
)
 
        
        iae_acumulado += abs(e_inst) * dt
        itae_acumulado += (t_act * abs(e_inst)) * dt
        
        h_log.append(h_corrida)
        qin_log.append(q_entrada)
        qout_log.append(q_salida)
        e_log.append(e_inst)
        
        m_h.metric("Nivel PV [m]", f"{h_corrida:.3f}")
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
            ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, h_corrida, color=color_agua, alpha=0.85, zorder=1))
            if q_entrada > 0:
                ax_t.annotate('', xy=(-r_max-1.5, h_corrida*0.7), xytext=(-r_max-0.3, h_corrida*0.7),
                            arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
            if q_salida > 0:
                ax_t.annotate('', xy=(r_max+1.5, 0.3), xytext=(r_max+0.3, 0.3),
                            arrowprops=dict(arrowstyle='->', lw=3, color='red'))
            
        elif geom_tanque == "Cónico":
            c_in_x, c_in_y = -(r_max/h_total)*(h_total*0.8), h_total*0.8
            c_out_x, c_out_y = 0, 0
            ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=5, zorder=2)
            if h_corrida > 0:
                radio_h = (r_max / h_total) * h_corrida
                vertices = [[-radio_h, h_corrida], [radio_h, h_corrida], [0, 0]]
                ax_t.add_patch(plt.Polygon(vertices, color=color_agua, alpha=0.85, zorder=1))
            
        else:
            import math
            c_in_y = h_total * 0.7
            c_in_x = -math.sqrt(abs(r_max**2 - (c_in_y - r_max)**2))
            c_out_x, c_out_y = 0, 0
            agua = plt.Circle((0, r_max), r_max, color=color_agua, alpha=0.85, zorder=1)
            ax_t.add_patch(agua)
            recorte = plt.Rectangle((-r_max, 0), 2*r_max, h_corrida, transform=ax_t.transData)
            agua.set_clip_path(recorte)
            ax_t.add_patch(plt.Circle((0, r_max), r_max, color='#2c3e50', fill=False, lw=5, zorder=2))
        
        # Tuberías y válvulas
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
        ax_t.text(-r_max*2.8, sp_nivel + 0.05, f"SP: {sp_nivel:.2f}m", color='red', fontweight='bold')
        ax_t.text(0, h_total * 1.2, f"PV: {h_corrida:.3f} m", ha='center', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='#1a5276', boxstyle='round'))
        
        if p_activa and t_act >= p_tiempo:
            ax_t.text(0, -0.5, f"⚠️ PERTURBACIÓN ACTIVA", ha='center', color='orange', fontweight='bold')
        
        placeholder_tanque.pyplot(fig_t)
        plt.close(fig_t)
        
        # Gráfico de tendencia
        fig_tr, ax_tr = plt.subplots(figsize=(8, 3.5))
        ax_tr.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2, label='Nivel')
        ax_tr.axhline(y=sp_nivel, color='red', ls='--', alpha=0.5, label='Setpoint')
        if p_activa and t_act >= p_tiempo:
            ax_tr.axvspan(p_tiempo, tiempo_ensayo, alpha=0.1, color='orange', label='Perturbación')
        ax_tr.set_xlabel('Tiempo [s]')
        ax_tr.set_ylabel('Altura [m]')
        ax_tr.legend()
        ax_tr.set_xlim(0, tiempo_ensayo)
        ax_tr.set_ylim(0, h_total * 1.1)
        ax_tr.grid(True, alpha=0.2)
        placeholder_grafico.pyplot(fig_tr)
        plt.close(fig_tr)
        
        # Gráfico de válvulas
        fig_v, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3))
        ax1.step(vector_t[:i+1], qin_log, where='post', color='blue', lw=2)
        ax1.set_ylabel('Q entrada [m³/s]')
        ax1.set_ylim(0, q_max_bomba * 1.1)
        ax1.grid(True, alpha=0.2)
        ax1.set_title('V-01 (Entrada)')
        
        ax2.step(vector_t[:i+1], qout_log, where='post', color='red', lw=2)
        ax2.set_ylabel('Q salida [m³/s]')
        ax2.set_xlabel('Tiempo [s]')
        ax2.set_ylim(0, q_max_bomba * 1.1)
        ax2.grid(True, alpha=0.2)
        ax2.set_title('V-02 (Salida)')
        plt.tight_layout()
        placeholder_valvulas.pyplot(fig_v)
        plt.close(fig_v)
        
        
        # Gráfico comparativo con datos experimentales (DENTRO DEL BUCLE)
        if mostrar_ref:
            datos_usr_df = st.session_state.get('datos_usr', pd.DataFrame())
            if not isinstance(datos_usr_df, pd.DataFrame):
                datos_usr_df = pd.DataFrame(datos_usr_df)
            
            if "Nivel Medido (cm)" in datos_usr_df.columns and len(datos_usr_df) > 0:
                t_exp = datos_usr_df["Tiempo (s)"].values
                h_exp = [val / 100 for val in datos_usr_df["Nivel Medido (cm)"].values]
                
                fig_comp, ax_comp = plt.subplots(figsize=(8, 3.5))
                ax_comp.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2.5, label='Simulación PID')
                ax_comp.scatter(t_exp, h_exp, color='red', marker='x', s=100, label='Datos Experimentales')
                ax_comp.plot(t_exp, h_exp, color='red', linestyle='--', alpha=0.3)
                ax_comp.axhline(y=sp_nivel, color='green', ls='--', alpha=0.3, label='Setpoint')
                ax_comp.set_xlabel("Tiempo [s]")
                ax_comp.set_ylabel("Nivel [m]")
                ax_comp.set_xlim(0, tiempo_ensayo)
                ax_comp.set_ylim(0, h_total * 1.1)
                ax_comp.grid(True, alpha=0.3)
                ax_comp.legend(loc='lower right', fontsize='x-small')
                placeholder_comparativa.pyplot(fig_comp)
                plt.close(fig_comp)
        
        
       
        time.sleep(0.02)
        barra_p.progress((i+1)/len(vector_t))
    
    status_placeholder.empty()
    st.success(f"✅ Simulación completada - Proceso: {tipo_proceso}")
    st.balloons()
    
    # Análisis final
    st.markdown("---")
    st.subheader("📈 Análisis de Respuesta")
    
    col_an1, col_an2 = st.columns([2, 1])
    
    with col_an1:
        fig_amp, ax_amp = plt.subplots(figsize=(10, 5))
        ax_amp.plot(vector_t, h_log, color='#1f77b4', lw=2.5, label='Respuesta')
        ax_amp.axhline(y=sp_nivel, color='#d62728', linestyle='--', lw=2, label='Setpoint')
        if p_activa and p_tiempo > 0:
            ax_amp.axvline(x=p_tiempo, color='orange', linestyle='--', alpha=0.7)
            ax_amp.axvspan(p_tiempo, tiempo_ensayo, alpha=0.08, color='orange')
        ax_amp.set_title(f"Respuesta del Control PID - {tipo_proceso}")
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
            st.success("✅ Excelente control")
        elif err_final < 0.05:
            st.info("👍 Buen control")
        else:
            st.warning("⚠️ Ajustar PID")
    
    # Tabla de resumen
    st.markdown("---")
    st.subheader("📋 Resumen de Datos")
    
    col_tab, col_res = st.columns([2, 1])
    
    with col_tab:
        df_resumen = pd.DataFrame({
            "Tiempo [s]": vector_t[-10:] if len(vector_t) >= 10 else vector_t,
            "Nivel [m]": h_log[-10:] if len(h_log) >= 10 else h_log,
            "Q_entrada": qin_log[-10:] if len(qin_log) >= 10 else qin_log,
            "Q_salida": qout_log[-10:] if len(qout_log) >= 10 else qout_log,
            "Error [m]": e_log[-10:] if len(e_log) >= 10 else e_log
        })
        st.dataframe(df_resumen.style.format("{:.4f}"), use_container_width=True)
    
    with col_res:
        err_f = abs(sp_nivel - h_log[-1]) if len(h_log) > 0 else 0
        st.metric("Error Residual", f"{err_f:.4f} m")
    
    # Exportar
    df_final = pd.DataFrame({
        "Tiempo [s]": vector_t,
        "Nivel [m]": h_log,
        "Q_entrada [m3/s]": qin_log,
        "Q_salida [m3/s]": qout_log,
        "Error [m]": e_log,
        "Kp_Usado": [k_p] * len(vector_t),
        "Ki_Usado": [k_i] * len(vector_t),
        "Kd_Usado": [k_d] * len(vector_t),
        "Qmax_Usado": [q_max_bomba] * len(vector_t),
        "Cd_Usado": [cd_para_simular] * len(vector_t),
        "Diametro_orificio_pulg": [d_pulgadas] * len(vector_t),
        "Tipo_Proceso": [tipo_proceso] * len(vector_t)
    })
    
    st.download_button(
        label="📥 Descargar Datos (CSV)",
        data=df_final.to_csv(index=False),
        file_name=f"simulacion_{tipo_proceso.lower()}_{geom_tanque.lower()}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("""
<hr style="margin: 2rem 0 1rem 0; border-color: #1a5276;">
<div style="text-align: center; color: #5d6d7e; font-size: 0.8rem;">
    <p>Universidad Central de Venezuela - Escuela de Ingeniería Química</p>
    <p>Simulador de Control PID con Dos Válvulas | © 2025</p>
</div>
""", unsafe_allow_html=True)
