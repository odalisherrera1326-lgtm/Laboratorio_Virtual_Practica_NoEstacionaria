import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_squared_error
# =============================================================================
# --- 1. FUNCIÓN PARA AUTO-SINTONIZACIÓN ---
# =============================================================================
def calcular_pid_adaptativo(geom, r_max, h_total):
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
    else: # Esférico
        kp = (area_max * 0.6) * 2.0
        ki = kp / 18.0
        kd = kp * 0.2
    return round(kp, 2), round(ki, 3), round(kd, 3)


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

# =============================================================================
# 1. CONFIGURACIÓN E IDENTIDAD INSTITUCIONAL UCV
# =============================================================================
st.set_page_config(
    page_title="Tesis UCV - Simulación Dinámica",
    page_icon="🛠️",
    layout="wide"
)

# --- PEGA EL BLOQUE DE COLORES AQUÍ ---
st.markdown("""
    <style>
    /* 1. CONFIGURACIÓN GLOBAL Y CURSOR DE ENGRANAJE ⚙️ */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f4f7f9 !important; 
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' style='font-size: 24px;'><text y='20'>⚙️</text></svg>") 16 16, auto !important;
    }

    button, a, [data-testid="stHeaderActionElements"], .stSlider {
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32' style='font-size: 24px;'><text y='20'>⚙️</text><text x='10' y='28' style='font-size: 14px;'>👆</text></svg>") 16 16, pointer !important;
    }

    /* 2. BARRAS DE INTERFAZ (AZUL UCV) */
    header[data-testid="stHeader"] {
        background-color: #1a5276 !important;
        color: white !important;
    }

    [data-testid="stSidebar"] {
        background-color: #1a5276 !important;
        border-right: 4px solid #154360 !important;
    }

    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important;
        font-weight: 500;
    }

    /* 3. RECUADROS Y NÚMEROS DE MÉTRICAS  */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2px solid #1a5276 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
        text-align: center !important;
    }
    div[data-testid="stMetric"] label {
        color: #1a5276 !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stMetric"] div[data-testid Baseweb] {
        color: #154360 !important; 
    }

    /* 4. PERSONALIZACIÓN DE SLIDERS (NARANJA PARA CONTRASTE) */
    div[data-baseweb="slider"] > div > div > div {
        background-color: #FFA500 !important;
    }
    div[role="slider"] {
        background-color: #FFA500 !important;
        border: 2px solid white !important;
    }
    div[data-testid="stTickBar"] div {
        color: white !important; /* Cambiado a blanco para que se vea en el azul */
    }

    /* 5. INPUTS Y BOTONES */
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #ffffff !important;
        color: #1a5276 !important;
        border: 2px solid #2980b9 !important;
        border-radius: 8px !important;
    }

    .stButton>button {
        background-color: #1a5276 !important; 
        color: white !important; 
        border: 2px solid white !important;
        border-radius: 12px !important;
        font-weight: bold !important;
    }
    
    .stButton>button:hover {
        border: 2px solid #f1c40f !important;
        box-shadow: 0 0 25px #f1c40f !important;
    }

    /* 6. BANNER DE ENCABEZADO UCV */
    .header-container {
        background: linear-gradient(-45deg, #154360, #1a5276, #21618c, #1a5276);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 25px; border-radius: 15px; border-bottom: 6px solid #f1c40f;
        color: white; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# ENCABEZADO INSTITUCIONAL CON FONDO 
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
    with st.expander(" Criterios de Desempeño (IAE/ITAE)", expanded=False):
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
    op_tipo = st.sidebar.selectbox(" Operación Principal", ["Llenado", "Vaciado"])
    geom_tanque = st.sidebar.selectbox(" Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])

with st.sidebar.expander(" Especificaciones del Tanque", expanded=True):
    r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
    h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
    h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
    sp_nivel = st.slider("Consigna de Nivel (Setpoint) [m]", 0.1, float(h_total), float(h_total/2))


with st.sidebar.expander(" Dimensiones de Salida", expanded=True):
    # Entrada única en pulgadas
    d_pulgadas = st.number_input("Diámetro del Orificio (pulgadas)", value=1.0, min_value=0.1, step=0.1)
    
    # Conversión a metros
    d_metros = d_pulgadas * 0.0254
    area_orificio = np.pi * (d_metros / 2)**2
    st.caption(f"Área calculada: {area_orificio:.6f} m²")

with st.sidebar.expander("🛡️ Escenario de Perturbación ($Q_p$)"):
            p_activa = st.toggle("Simular Falla/Fuga Externas", value=True)
            
            if p_activa:
                p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.045, format="%.4f")
                p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 80)
                
                # --- NUEVO: Interruptor de Inteligencia de Perturbación ---
                modo_estres = st.toggle("🔥 Activar Modo Estrés", 
                                       help="La perturbación cambiará según el nivel para desafiar al PID.")
            else:
                p_magnitud = 0.0
                p_tiempo = 0
                modo_estres = False

with st.sidebar.expander("Parámetros del Controlador PID"):
    # Primero calculamos los sugeridos
    kp_sug, ki_sug, kd_sug = calcular_pid_adaptativo(geom_tanque, r_max, h_total)

    st.markdown("---")
    st.subheader("🎛️ Configuración")
    
    # Selector de modo
    metodo_control = st.radio("Método de Sintonización", 
                            ["Manual (Usuario)", "Asistida (Sugerida)"],
                            index=0, horizontal=True)

    if metodo_control == "Asistida (Sugerida)":
        st.success("💡 Usando parámetros optimizados")
        kp_val = st.number_input("Kp", value=kp_sug, key="kp_asist")
        ki_val = st.number_input("Ki", value=ki_sug, format="%.3f", key="ki_asist")
        kd_val = st.number_input("Kd", value=kd_sug, format="%.3f", key="kd_asist")
    else:
        st.info("✍️ Ingrese sus propios parámetros")
        kp_val = st.number_input("Kp", value=kp_sug, step=0.1, key="kp_man")
        ki_val = st.number_input("Ki", value=ki_sug, step=0.001, format="%.3f", key="ki_man")
        kd_val = st.number_input("Kd", value=kd_sug, step=0.001, format="%.3f", key="kd_man")
    
    tiempo_ensayo = st.sidebar.slider("Tiempo de simulación [s]", 60, 600, 300)
with st.sidebar.expander("📊 Cargar Datos Experimentales"):
    st.write("Ingresa los datos medidos en el laboratorio:")
    # Tabla interactiva para el usuario
    datos_usr = st.data_editor({
        "Tiempo (s)": [0, 60, 120, 180, 240, 300],
        "Nivel Medido (m)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }, num_rows="dynamic")
    
    mostrar_ref = st.checkbox("Mostrar referencia en gráfica", value=True)

# =============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Biblioteca Técnica")

with st.sidebar.container(border=True):
    st.sidebar.write("Descargue el material de apoyo:")
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
def sintonizar_controlador_dinamico(geom, r, h_t, cd_calculado, area_ori):
    # Calculamos el área transversal según la geometría seleccionada
    if geom == "Cilíndrico":
        area_t = np.pi * (r**2)
    elif geom == "Cónico":
        area_t = np.pi * (r/2)**2 
    else: 
        area_t = (2/3) * np.pi * (r**2)

    # Kp: Proporcional a la inercia del tanque. 
    # Si el tanque es grande (area_t alta), necesitamos más fuerza.
    kp_sintonizado = np.clip(area_t / (cd_calculado * area_ori * 1.5), 3.0, 30.0)
    
    # Ki: Acción Integral reforzada. 
    # Es la que elimina el error estacionario y asegura llegar al Setpoint.
    ki_sintonizado = np.clip(1.2 * (1 / cd_calculado), 0.5, 5.0) 
    
    # Kd: Amortiguación para evitar que el nivel "rebote" mucho al llegar.
    kd_sintonizado = 0.15 
    
    return round(kp_sintonizado, 2), round(ki_sintonizado, 2), kd_sintonizado
def calcular_cd_inteligente(df_usr, r, h_t, geom, area_ori):
    """
    Calcula el Coeficiente de Descarga (Cd) usando el balance de masa.
    """
    import pandas as pd
    # Convertimos a DataFrame para evitar el AttributeError de la lista
    df = pd.DataFrame(df_usr) if isinstance(df_usr, list) else df_usr
    
    # Validamos que existan datos suficientes
    if len(df) < 2:
        return 0.61
    
    try:
        # Extraemos los datos de tiempo y nivel
        t1, t2 = df["Tiempo (s)"].iloc[0], df["Tiempo (s)"].iloc[1]
        h1, h2 = df["Nivel Medido (m)"].iloc[0], df["Nivel Medido (m)"].iloc[1]
        dt = abs(t2 - t1)
        
        if dt == 0: return 0.61

        # Cálculo de volúmenes según la geometría del tanque
        if geom == "Cilíndrico":
            v1, v2 = np.pi*(r**2)*h1, np.pi*(r**2)*h2
        elif geom == "Cónico":
            v1 = (1/3)*np.pi*((r/h_t)*h1)**2*h1
            v2 = (1/3)*np.pi*((r/h_t)*h2)**2*h2
        else: # Esférico
            v1 = (np.pi*(h1**2)/3)*(3*r-h1)
            v2 = (np.pi*(h2**2)/3)*(3*r-h2)

        q_real = abs(v1 - v2) / dt
        h_prom = (h1 + h2) / 2
        q_teorico = area_ori * np.sqrt(2 * 9.81 * max(h_prom, 0.001))
        
        # El Cd real suele estar entre 0.6 y 0.85
        return float(np.clip(q_real / q_teorico, 0.4, 1.0))
    except:
        return 0.61
    
    # Limitar el resultado a valores físicamente lógicos (0.4 a 1.0)
    return float(np.clip(cd_result, 0.4, 1.0))

def resolver_sistema(dt, h_prev, sp, geom, r, h_t, q_p_val, e_sum, e_prev, modo_op, cd_val,kp,ki,kd):
    # 1. Cálculo de área según geometría
    if geom == "Cilíndrico":
        area_h = np.pi * (r**2)
    elif geom == "Cónico":
        area_h = np.pi * ((r/h_t) * max(h_prev, 0.01))**2
    else: # Esférico
        area_h = np.pi * (2 * r * max(h_prev, 0.01) - max(h_prev, 0.01)**2)
    
    area_h = max(area_h, 0.01) 

   # 2. Algoritmo PID (Ahora usa las variables kp, ki, kd pasadas por argumento)
    err = sp - h_prev
    e_sum += err * dt
    e_der = (err - e_prev) / dt
    u_control = (kp * err) + (ki * e_sum) + (kd * e_der)
    
    # 3. Lógica de Operación y Balance de Masa
    # Usamos el área del orificio calculada en el sidebar (0.0254 es ejemplo, usa la real)
    # Para ser exactos, pasamos el área_orificio global o la calculamos aquí
    a_o = np.pi * ((d_pulgadas * 0.0254) / 2)**2 

    if modo_op == "Llenado":
        q_entrada = np.clip(u_control, 0, 2)
        # USAMOS EL CD CALCULADO (cd_val)
        q_salida = cd_val * a_o * np.sqrt(2 * 9.81 * h_prev) if h_prev > 0.005 else 0
        dh_dt = (q_entrada + q_p_val - q_salida) / area_h
        u_graficar = q_entrada
    else:
        q_entrada = q_p_val  
        q_salida = np.clip(-u_control, 0, 2) 
        dh_dt = (q_entrada - q_salida) / area_h
        u_graficar = q_salida
    
    h_next = np.clip(h_prev + dh_dt * dt, 0, h_t)
    return h_next, u_graficar, err, e_sum, err

# =============================================================================
# 5. LÓGICA DE VISUALIZACIÓN Y SIMULACIÓN UNIFICADA
# =============================================================================
if iniciar_sim:
    st.session_state.ejecutando = True
    
    # --- REINICIO DE MEMORIA DE CONTROL ---
    st.session_state['error_acumulado'] = 0.0
    st.session_state['ultimo_error'] = 0.0
    
    try:
        df_calibracion = pd.DataFrame(datos_usr)
        
        # LÓGICA DE DECISIÓN: ¿Automático o Manual?
        if modo_auto:
            if "Nivel Medido (m)" in df_calibracion.columns and not df_calibracion["Nivel Medido (m)"].isnull().all():
                # Sintonía automática de la tesis
                cd_calc = calcular_cd_inteligente(df_calibracion, r_max, h_total, geom_tanque, area_orificio)
                kp_a, ki_a, kd_a = sintonizar_controlador_dinamico(geom_tanque, r_max, h_total, cd_calc, area_orificio)
                
                st.session_state['kp_ejecucion'] = kp_a
                st.session_state['ki_ejecucion'] = ki_a
                st.session_state['kd_ejecucion'] = kd_a
                st.session_state['cd_final'] = cd_calc
                st.toast(f"🎯 Control Adaptativo: Cd={cd_calc:.2f} | Kp={kp_a}")
            else:
                raise ValueError("Tabla vacía")
        else:
            # Modo Manual (Sidebar)
            st.session_state['kp_ejecucion'] = kp_val
            st.session_state['ki_ejecucion'] = ki_val
            st.session_state['kd_ejecucion'] = kd_val
            st.session_state['cd_final'] = 0.61

    except Exception as e:
        st.session_state['kp_ejecucion'] = 5.0
        st.session_state['ki_ejecucion'] = 1.2
        st.session_state['kd_ejecucion'] = 0.1
        st.session_state['cd_final'] = 0.61
        st.warning("⚠️ Usando valores de respaldo.")

# Esta línea va fuera de los bloques 'if' y corregida (sin el paréntesis extra del error 616)
estado_expander = not st.session_state.get('ejecutando', False)



# --- PESTAÑA DEL DIAGRAMA ---
with st.expander("Diagrama del Proceso", expanded=estado_expander):
    col_img = st.columns([1, 5, 1])[1]
    with col_img:
        # Asegúrate de que el nombre del archivo coincida exactamente con el de tu PC
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
        st.subheader("Monitor del Proceso")
        placeholder_tanque = st.empty()
        st.subheader("Tendencia Temporal")
        placeholder_grafico = st.empty()
        st.subheader("⚙️ Acción del Controlador")
        placeholder_u = st.empty()
        # --- NUEVO: Espacio para el estado de la válvula ---
        st.markdown("---")
        st.subheader("⚙️ Estado de Operación: Válvula V-02")
        placeholder_valvula = st.empty()
        # --- NUEVO: Gráfica Independiente de Validación ---
        st.markdown("---")
        st.subheader("📊 Comparativa: Modelo Teórico vs Datos Experimentales")
        placeholder_comparativa = st.empty()
       

    with col_met:
        st.subheader("Métricas de Control")
        
        # 1. Parámetros calculados o manuales que se están usando
        kp_show = st.session_state.get('kp_ejecucion', kp_val)
        cd_show = st.session_state.get('cd_final', 0.61)
        
        st.write(f"**Parámetros Activos:**")
        st.caption(f"Kp: {kp_show} | Cd: {cd_show}")
        st.markdown("---")
        
        # 2. Espacios reservados para métricas dinámicas (placeholders)
        placeholder_iae = st.empty()
        placeholder_itae = st.empty()
        
        # Inicialización visual de las tarjetas
        placeholder_iae.metric("IAE (Error Acumulado)", "0.00")
        placeholder_itae.metric("ITAE (Criterio Tesis)", "0.00")
        
        st.markdown("---")
        
        # 3. Monitoreo instantáneo del nivel y error
        m_h = st.empty()
        m_e = st.empty()
        m_h.metric("Nivel PV [m]", "0.000")
        m_e.metric("Error [m]", "0.000")
        
        st.markdown("---")
        area_descarga = st.empty()
        
        
        
   # --- 2. Preparación de datos (Corregido para Vaciado) ---
    status_placeholder = st.empty()
    dt = 1.0 
    vector_t = np.arange(0, tiempo_ensayo, dt)
    h_log, u_log, e_log = [], [], []

    # Ajuste de condición inicial según la operación
    if op_tipo == "Llenado":
        h_corrida = 0.0
    else:
        # Para vaciado, empezamos con el tanque al 90% de su capacidad
        h_corrida = h_total * 0.9
    
    # Sincronización inicial de variables
    valor_presente = h_corrida  
    error_presente = 0.0        
    err_int, err_pasado = 0.0, 0.0
    iae_acumulado, itae_acumulado = 0.0, 0.0
   
    t_exp = datos_usr["Tiempo (s)"]
    # Asegúrate de que la división /100 sea necesaria (si tus datos están en cm)
    h_exp = [val / 100 for val in datos_usr["Nivel Medido (m)"]]
    barra_p = st.progress(0)
   
    # 3. Bucle de Simulación
    cd_para_simular = st.session_state.get('cd_final', 0.61)
    for i, t_act in enumerate(vector_t):
        status_placeholder.markdown("<div class='flow-indicator'>💧 PROCESANDO...</div>", unsafe_allow_html=True)
        
        # --- Lógica de Perturbación Inteligente ---
        if 'p_activa' in locals() and p_activa and t_act >= p_tiempo:
            if modo_estres:
                # Si el nivel < Setpoint (está subiendo), le metemos más flujo para sabotear
                factor = 1.5 if valor_presente < sp_nivel else 0.5
                q_p_inst = p_magnitud * factor
            else:
                q_p_inst = p_magnitud
        else:
            q_p_inst = 0.0
    
        k_p = st.session_state.get('kp_ejecucion', kp_val)
        k_i = st.session_state.get('ki_ejecucion', ki_val)
        k_d = st.session_state.get('kd_ejecucion', kd_val)

        h_corrida, u_inst, e_inst, err_int, err_pasado = resolver_sistema(
            dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst, 
            err_int, err_pasado, op_tipo, cd_para_simular,
            k_p, k_i, k_d
        )
        
        valor_presente = h_corrida 
        error_presente = e_inst
        iae_acumulado += abs(e_inst) * dt
        itae_acumulado += (t_act * abs(e_inst)) * dt
        
        h_log.append(h_corrida)
        u_log.append(u_inst)
        e_log.append(e_inst)
        
        # --- A. ACTUALIZACIÓN DE MÉTRICAS ---
        m_h.metric("Nivel PV [m]", f"{valor_presente:.3f}")
        m_e.metric("Error [m]", f"{error_presente:.4f}")
        placeholder_iae.metric("IAE (Error Acumulado)", f"{iae_acumulado:.2f}")
        placeholder_itae.metric("ITAE (Criterio Tesis)", f"{itae_acumulado:.2f}")
        
        # --- B. MONITOR DEL PROCESO (DINÁMICO) ---
        fig_t, ax_t = plt.subplots(figsize=(7, 5))
        ax_t.set_axis_off() 
        ax_t.set_xlim(-r_max*3, r_max*3) 
        ax_t.set_ylim(-0.8, h_total*1.3)
        color_agua = '#3498db' if abs(e_inst) < 0.1 else '#e74c3c'
        
        if geom_tanque == "Cilíndrico":
            c_in_x, c_in_y = -r_max, h_total*0.8
            c_out_x, c_out_y = r_max, 0.1
            ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, valor_presente, color=color_agua, alpha=0.6, zorder=1))
            ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=4, zorder=2)

        elif geom_tanque == "Cónico":
            c_in_x, c_in_y = -(r_max/h_total)*(h_total*0.8), h_total*0.8
            c_out_x, c_out_y = 0, 0
            ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=4, zorder=2)
            r_act_cono = (r_max / h_total) * valor_presente
            ax_t.add_patch(plt.Polygon([[-r_act_cono, valor_presente], [r_act_cono, valor_presente], [0, 0]], color=color_agua, alpha=0.6, zorder=1))

        else: # Esférico
            import math
            c_in_y = h_total * 0.7
            c_in_x = -math.sqrt(abs(r_max**2 - (c_in_y - r_max)**2))
            c_out_x, c_out_y = 0, 0
            agua_esf = plt.Circle((0, r_max), r_max, color=color_agua, alpha=0.6, zorder=1)
            ax_t.add_patch(agua_esf)
            recorte_nivel = plt.Rectangle((-r_max, 0), 2*r_max, valor_presente, transform=ax_t.transData)
            agua_esf.set_clip_path(recorte_nivel)
            ax_t.add_patch(plt.Circle((0, r_max), r_max, color='#2c3e50', fill=False, lw=4, zorder=2))

        # --- 2. INFRAESTRUCTURA DE ENTRADA (V-01) ---
        # Tubo de entrada gris
        ax_t.add_patch(plt.Rectangle((c_in_x - 1.5, c_in_y - 0.1), 1.5, 0.2, color='silver', zorder=0))
        # Válvula V-01 (Símbolo moño completo)
        ax_t.add_patch(plt.Polygon([[c_in_x-1, c_in_y+0.2], [c_in_x-1, c_in_y-0.2], [c_in_x-0.6, c_in_y]], color='#2c3e50', zorder=2))
        ax_t.add_patch(plt.Polygon([[c_in_x-0.2, c_in_y+0.2], [c_in_x-0.2, c_in_y-0.2], [c_in_x-0.6, c_in_y]], color='#2c3e50', zorder=2))
        ax_t.text(c_in_x-0.6, c_in_y+0.4, "V-01", ha='center', fontsize=9, fontweight='bold')

        # --- 3. INFRAESTRUCTURA DE SALIDA (V-02 CV) ---
        t_ancho = 0.2
        if geom_tanque == "Cilíndrico":
            ax_t.add_patch(plt.Rectangle((c_out_x, c_out_y - t_ancho/2), 1.5, t_ancho, color='silver', zorder=0))
            vs_x, vs_y = c_out_x + 0.8, c_out_y
        else:
            ax_t.add_patch(plt.Rectangle((c_out_x - t_ancho/2, -0.6), t_ancho, 0.6, color='silver', zorder=0))
            vs_x, vs_y = c_out_x, -0.4

        # Válvula V-02
        ax_t.add_patch(plt.Polygon([[vs_x-0.25, vs_y+0.2], [vs_x-0.25, vs_y-0.2], [vs_x, vs_y]], color='#2c3e50', zorder=2))
        ax_t.add_patch(plt.Polygon([[vs_x+0.25, vs_y+0.2], [vs_x+0.25, vs_y-0.2], [vs_x, vs_y]], color='#2c3e50', zorder=2))
        offset_t = 0.4 if geom_tanque == "Cilíndrico" else 0
        ax_t.text(vs_x + offset_t, vs_y - 0.5, "V-02 (CV)", ha='center', fontsize=9, fontweight='bold')

        # --- 4. INDICADORES DINÁMICOS Y SETPOINT ---
        ax_t.axhline(y=sp_nivel, color='red', ls='--', lw=2, zorder=3)
        ax_t.text(-r_max*2.8, sp_nivel + 0.05, f"SETPOINT: {sp_nivel:.2f}m", color='red', fontweight='bold', fontsize=9)

        # Burbuja de PV
        ax_t.text(0, h_total * 1.2, f"PV: {valor_presente:.3f} m", 
                 ha='center', va='center', fontsize=11, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='#1a5276', boxstyle='round,pad=0.5', lw=2))

        # Renderizados dentro del bucle para animación en vivo
        placeholder_tanque.pyplot(fig_t)
        plt.close(fig_t)

        # --- C. TENDENCIA DE NIVEL --- 
        fig_tr, ax_tr = plt.subplots(figsize=(8, 3.5))
        ax_tr.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2, label='Nivel (h)')
        ax_tr.axhline(y=sp_nivel, color='red', ls='--', alpha=0.5, label='Setpoint')
        ax_tr.set_xlabel('Tiempo [s]', fontsize=10, fontweight='bold')
        ax_tr.set_ylabel('Altura [m]', fontsize=10, fontweight='bold')
        ax_tr.legend(loc='upper right', fontsize='x-small')
        ax_tr.set_xlim(0, tiempo_ensayo)
        ax_tr.set_ylim(0, h_total * 1.1)
        ax_tr.grid(True, alpha=0.2)
        placeholder_grafico.pyplot(fig_tr)
        plt.close(fig_tr)

        # --- D. ACCIÓN DE CONTROL ---
        fig_u, ax_u = plt.subplots(figsize=(8, 2.5))
        ax_u.step(vector_t[:i+1], u_log, color='#e67e22', where='post')
        ax_u.set_xlim(0, tiempo_ensayo)
        techo_dinamico = max(max(u_log), 0.1) * 1.2 if u_log else 0.7
        ax_u.set_ylim(0, techo_dinamico)
        ax_u.grid(True, alpha=0.2)
        ax_u.set_xlabel('Tiempo [s]', fontsize=10, fontweight='bold')
        ax_u.set_ylabel('Flujo [m3/s]', fontsize=10, fontweight='bold')
        placeholder_u.pyplot(fig_u)
        plt.close(fig_u)

        # --- E. LÓGICA DE LA VÁLVULA ---
        fig_v, ax_v = plt.subplots(figsize=(8, 3))
        ax_v.plot(vector_t[:i+1], u_log, color='#2ecc71', lw=2.5)
        ax_v.fill_between(vector_t[:i+1], u_log, color='#2ecc71', alpha=0.15)
        ax_v.set_ylim(-0.1, 1.1) 
        ax_v.set_yticks([0, 0.5, 1])
        ax_v.set_yticklabels(['CERRADA', '50%', 'ABIERTA'])
        ax_v.set_title("Apertura de Válvula de Control", fontsize=10, fontweight='bold')
        placeholder_valvula.pyplot(fig_v)
        plt.close(fig_v)

        # --- F. GRÁFICA COMPARATIVA ---
        fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
        ax_comp.plot(vector_t[:i+1], h_log, color='#1f77b4', lw=2, label='Simulación')
        if mostrar_ref:
            ax_comp.scatter(t_exp, h_exp, color='red', marker='x', s=100, label='Datos UCV')
            ax_comp.plot(t_exp, h_exp, color='red', linestyle='--', alpha=0.3)
        ax_comp.set_title("Validación de Resultados", fontsize=10, fontweight='bold')
        ax_comp.set_xlabel("Tiempo [s]")
        ax_comp.set_ylabel("Nivel [m]")
        ax_comp.set_ylim(0, h_total * 1.1)
        ax_comp.grid(True, alpha=0.3)
        ax_comp.legend(loc='lower right')
        placeholder_comparativa.pyplot(fig_comp)
        plt.close(fig_comp)

        # Control de velocidad y progreso
        time.sleep(0.01) 
        barra_p.progress((i+1)/len(vector_t))

    # =============================================================================
    # FIN DEL BUCLE FOR - INICIO DE RESULTADOS FINALES
    # =============================================================================
    status_placeholder.empty()
    st.success(f"✅ Simulación del Tanque {geom_tanque} completada exitosamente.")
    st.balloons()

    # --- 7. ANÁLISIS DE RESPUESTA TRANSITORIA (AMPLITUD VS TIEMPO) ---
    st.markdown("---")
    st.subheader("📈 Análisis de Respuesta al Escalón (Amplitud vs. Tiempo)")

    col_an1, col_an2 = st.columns([2, 1])

    with col_an1:
        fig_amp, ax_amp = plt.subplots(figsize=(10, 5))
        ax_amp.plot(vector_t, h_log, color='#1f77b4', lw=2.5, label='Respuesta del Sistema (PV)')
        
        # Corregido: Línea horizontal para evitar el error de sp_log
        ax_amp.axhline(y=sp_nivel, color='#d62728', linestyle='--', lw=2, label='Referencia (SP)')
        
        ax_amp.set_title("Respuesta Transitoria del Lazo de Control (MatLab Style)", fontsize=12)
        ax_amp.set_xlabel("Tiempo (s)")
        ax_amp.set_ylabel("Amplitud (m)")
        ax_amp.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_amp.legend(loc='lower right')
        
        # Banda de estabilidad técnica
        if len(h_log) > 0:
            error_f_val = abs(h_log[-1] - sp_nivel)
            if error_f_val < 0.05:
                ax_amp.axhspan(sp_nivel-0.05, sp_nivel+0.05, color='green', alpha=0.1, label='Banda de Estabilidad')
        
        st.pyplot(fig_amp)
        plt.close(fig_amp)

    with col_an2:
        st.info("**Interpretación Técnica:**")
        sobrepico = ((max(h_log) - sp_nivel) / sp_nivel) * 100 if max(h_log) > sp_nivel else 0
        st.metric("Sobrepico Máximo", f"{sobrepico:.2f} %")
        st.metric("IAE Final", f"{iae_acumulado:.2f}")
        st.metric("ITAE Final", f"{itae_acumulado:.2f}")

    # --- RESULTADOS FINALES Y DESCARGA ---
    df_final = pd.DataFrame({
        "Tiempo [s]": vector_t, 
        "Nivel [m]": h_log, 
        "Control [m3/s]": u_log,
        "Error [m]": e_log,
        "Kp_Sintonizado": [st.session_state.get('kp_ejecucion')] * len(vector_t),
        "Ki_Sintonizado": [st.session_state.get('ki_ejecucion')] * len(vector_t)
    })
    
    st.subheader("📋 Resumen de Datos y Estabilidad")
    
    col_tab, col_res = st.columns([2, 1])
    
    with col_tab:
        st.dataframe(df_final.tail(10).style.format("{:.4f}"), use_container_width=True)
    
    with col_res:
        err_f = abs(sp_nivel - h_log[-1]) if len(h_log) > 0 else 0
        st.metric("Error Residual Final", f"{err_f:.4f} m")
        
        st.download_button(
            label="📥 Descargar Reporte de datos (CSV)", 
            data=df_final.to_csv(index=False), 
            file_name=f"resultados_tesis_{geom_tanque}.csv",
            mime="text/csv",
            use_container_width=True
        )

    if err_f < 0.05:
        st.success(f"✅ El sistema alcanzó el estado estacionario en {h_log[-1]:.3f} m.")
    else:
        st.warning(f"⚠️ El sistema presenta un error residual de {err_f:.3f} m. Se sugiere ajustar Kp/Ki.")
