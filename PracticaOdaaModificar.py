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

# =============================================================================
# INTERFAZ TESIS 
# =============================================================================
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
    /* Color de los números: Azul oscuro para legibilidad */
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #154360 !important; 
        font-size: 2rem !important;
        font-weight: 800 !important;
    }

    /* 4. BARRA DE PROGRESO "PROCESANDO" (AHORA AZUL) */
    .stProgress > div > div > div > div {
        background-color: #2980b9 !important;
        animation: pulso_azul 2s ease-in-out infinite;
        box-shadow: 0 0 12px rgba(41, 128, 185, 0.6);
    }
    
    @keyframes pulso_azul {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
//* 1. COLOR DE LA BARRA ACTIVA  */
    div[data-baseweb="slider"] > div > div > div {
        background-color: #FFA500 !important;
    }

    /* 2. COLOR DEL BOTÓN CIRCULAR */
    div[role="slider"] {
        background-color: #FFA500 !important;
        border: 2px solid white !important;
    }

    /* 3. COLOR DEL NÚMERO QUE FLOTA (EL VALOR ACTUAL) */
    /* Este selector apunta específicamente al valor dinámico */
    div[data-baseweb="slider"] div[data-testid="stTickBar"] + div,
    div[data-baseweb="slider"] ~ div span {
        color: #FFA500 !important;
        font-weight: bold !important;
    }
    
    /* 4. COLOR DE LOS NÚMEROS DE LOS EXTREMOS (MIN Y MAX) */
    div[data-testid="stTickBar"] div {
        color: #FFA500 !important;
        font-size: 0.8rem !important;
    }

    /* 5. RESTABLECER EL COLOR DE LAS ETIQUETAS (PARA QUE NO SEAN NARANJAS) */
    /* Esto asegura que "Simular Falla/Fuga" y "Consigna de Nivel" vuelvan a ser blancos o negros */
    div[data-testid="stWidgetLabel"] p {
        color: white !important; /* O el color que prefieras para tus títulos */
        font-weight: normal !important;
    }
    }

    /* 6. INPUTS Y BOTONES */
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #ffffff !important;
        color: #1a5276 !important;
        border: 2px solid #2980b9 !important;
        border-radius: 8px !important;
    }

    [data-testid="stSidebar"] .stDownloadButton button {
        background-color: #27ae60 !important;
        color: white !important;
        border-radius: 12px !important;
    }

    .stButton>button {
        background-color: #1a5276 !important; 
        color: white !important; 
        border: 2px solid white !important;
        border-radius: 12px !important;
    }
    
    .stButton>button:hover {
        border: 2px solid #2980b9 !important;
        box-shadow: 0 0 20px rgba(41, 128, 185, 0.5) !important;
    }

    /* 7. BANNER DE ENCABEZADO */
    .header-container {
        background: linear-gradient(-45deg, #154360, #1a5276, #21618c, #1a5276);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        padding: 25px; border-radius: 15px; border-bottom: 6px solid #1a5276;
        color: white; text-align: center;
    }

    /* 8. BOTONES DE CONTROL (INICIAR/RESET) CON BRILLO */
    .stButton>button {
        background-color: #1a5276 !important; 
        color: white !important; 
        border: 2px solid white !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        border: 2px solid #f1c40f !important;
        box-shadow: 0 0 25px #f1c40f !important;
        transform: translateY(-2px);
    }

    div.stButton > button:first-child[kind="secondary"] {
        background-color: #943126 !important;
    }
    div.stButton > button:first-child[kind="secondary"]:hover {
        box-shadow: 0 0 25px #cb4335 !important;
    }

    /* 9. BANNER DE ENCABEZADO UCV */
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

with st.sidebar.expander(" Escenario de Perturbación ($Q_p$)"):
    p_activa = st.toggle("Simular Falla/Fuga Externas", value=True)
    p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.045, format="%.4f") if p_activa else 0.0
    p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 80) if p_activa else 0

with st.sidebar.expander("Parámetros del Controlador PID"):
    c1, c2, c3 = st.columns(3)
    kp_val = c1.number_input("Kp", value=2.6)
    ki_val = c2.number_input("Ki", value=0.5)
    kd_val = c3.number_input("Kd", value=0.1)
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
def resolver_sistema(dt, h_prev, sp, geom, r, h_t, q_p_val, e_sum, e_prev, modo_op):
    # 1. Cálculo de área según geometría
    if geom == "Cilíndrico":
        area_h = np.pi * (r**2)
    elif geom == "Cónico":
        area_h = np.pi * ((r/h_t) * max(h_prev, 0.01))**2
    else: # Esférico
        area_h = np.pi * (2 * r * max(h_prev, 0.01) - max(h_prev, 0.01)**2)
    
    area_h = max(area_h, 0.01) 

    # 2. Algoritmo PID
    err = sp - h_prev
    e_sum += err * dt
    e_der = (err - e_prev) / dt
    u_control = (kp_val * err) + (ki_val * e_sum) + (kd_val * e_der)
    
    # 3. Lógica de Operación y Balance de Masa
    if modo_op == "Llenado":
        # Controlamos la ENTRADA. La perturbación es una fuga o entrada extra.
        q_entrada = np.clip(u_control, 0, 0.6)
        q_salida = 0.61 * 0.04 * np.sqrt(2 * 9.81 * h_prev) if h_prev > 0.005 else 0
        
        # dh/dt = (Entrada_Control + Perturbación - Salida_Natural) / Área
        dh_dt = (q_entrada + q_p_val - q_salida) / area_h
        u_graficar = q_entrada
    else:
        # VACIADO: Controlamos la SALIDA. La perturbación es el FLUJO DE ENTRADA.
        q_entrada = q_p_val  # La perturbación entra al tanque
        q_salida = np.clip(-u_control, 0, 0.6) # PID abre la válvula de salida
        
        # dh/dt = (Entrada_Perturbación - Salida_Controlada) / Área
        dh_dt = (q_entrada - q_salida) / area_h
        u_graficar = q_salida
    
    h_next = np.clip(h_prev + dh_dt * dt, 0, h_t)
    
    return h_next, u_graficar, err, e_sum, err


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
        st.subheader("📊 Comparativa: Modelo Teórico vs Planta Real")
        placeholder_comparativa = st.empty()
       

    with col_met:
        st.subheader("Métricas de Control")
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
   
    t_exp = datos_usr["Tiempo (s)"]
    h_exp = [val / 100 for val in datos_usr["Nivel Medido (m)"]]
    barra_p = st.progress(0)
   

    # 3. Bucle de Simulación
    for i, t_act in enumerate(vector_t):
        status_placeholder.markdown("<div class='flow-indicator'>💧 PROCESANDO...</div>", unsafe_allow_html=True)
        
        # Lógica de perturbación
        q_p_inst = p_magnitud if ('p_activa' in locals() and p_activa and t_act >= p_tiempo) else 0.0
        
        h_corrida, u_inst, e_inst, err_int, err_pasado = resolver_sistema(
            dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst, err_int, err_pasado, op_tipo
        )
        
        # Cálculo de métricas integrales
        iae_acumulado += abs(e_inst) * dt
        itae_acumulado += (t_act * abs(e_inst)) * dt
        
        h_log.append(h_corrida)
        u_log.append(u_inst)
        sp_log.append(sp_nivel) 
        e_log.append(e_inst)
        
        if i % 2 == 0:
            # A. Actualización de métricas
            m_h.metric("Nivel PV [m]", f"{h_corrida:.3f}")
            m_e.metric("Error [m]", f"{e_inst:.4f}")
            placeholder_iae.metric("IAE (Error Acumulado)", f"{iae_acumulado:.2f}")
            placeholder_itae.metric("ITAE (Criterio Tesis)", f"{itae_acumulado:.2f}")

           # --- B. MONITOR DEL PROCESO (DINÁMICO) ---
         
            fig_t, ax_t = plt.subplots(figsize=(7, 5))
            ax_t.set_axis_off() 
            ax_t.set_xlim(-r_max*3, r_max*3) 
            ax_t.set_ylim(-0.8, h_total*1.3)
            color_agua = '#3498db' if abs(e_inst) < 0.1 else '#e74c3c'
            
            # --- 1. LÓGICA DE GEOMETRÍA Y PUNTOS DE CONEXIÓN ---
            if geom_tanque == "Cilíndrico":
                c_in_x, c_in_y = -r_max, h_total*0.8
                c_out_x, c_out_y = r_max, 0.1
                # Dibujo del agua y cuerpo
                ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, h_corrida, color=color_agua, alpha=0.6, zorder=1))
                ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=4, zorder=2)

            elif geom_tanque == "Cónico":
                c_in_x, c_in_y = -(r_max/h_total)*(h_total*0.8), h_total*0.8
                c_out_x, c_out_y = 0, 0  # Conexión pegada a la punta
                # Cuerpo del tanque
                ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=4, zorder=2)
                # Dibujo del agua (Triángulo invertido dinámico)
                r_act_cono = (r_max / h_total) * h_corrida
                ax_t.add_patch(plt.Polygon([[-r_act_cono, h_corrida], [r_act_cono, h_corrida], [0, 0]], color=color_agua, alpha=0.6, zorder=1))

            else: # Esférico
                import math
                c_in_y = h_total * 0.7
                c_in_x = -math.sqrt(abs(r_max**2 - (c_in_y - r_max)**2))
                c_out_x, c_out_y = 0, 0 # Conexión pegada a la base
                
                # Dibujo del agua con técnica de recorte (clipping)
                agua_esf = plt.Circle((0, r_max), r_max, color=color_agua, alpha=0.6, zorder=1)
                ax_t.add_patch(agua_esf)
                
                # Recorte dinámico según el nivel h_corrida
                recorte_nivel = plt.Rectangle((-r_max, 0), 2*r_max, h_corrida, transform=ax_t.transData)
                agua_esf.set_clip_path(recorte_nivel)
                
                # Borde del tanque esférico
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
                # Salida lateral para el cilindro
                ax_t.add_patch(plt.Rectangle((c_out_x, c_out_y - t_ancho/2), 1.5, t_ancho, color='silver', zorder=0))
                vs_x, vs_y = c_out_x + 0.8, c_out_y
            else:
                # Salida inferior vertical pegada al tanque (y=0)
                ax_t.add_patch(plt.Rectangle((c_out_x - t_ancho/2, -0.6), t_ancho, 0.6, color='silver', zorder=0))
                vs_x, vs_y = c_out_x, -0.4

            # Válvula V-02 (Símbolo moño corregido)
            ax_t.add_patch(plt.Polygon([[vs_x-0.25, vs_y+0.2], [vs_x-0.25, vs_y-0.2], [vs_x, vs_y]], color='#2c3e50', zorder=2))
            ax_t.add_patch(plt.Polygon([[vs_x+0.25, vs_y+0.2], [vs_x+0.25, vs_y-0.2], [vs_x, vs_y]], color='#2c3e50', zorder=2))
            
            offset_t = 0.4 if geom_tanque == "Cilíndrico" else 0
            ax_t.text(vs_x + offset_t, vs_y - 0.5, "V-02 (CV)", ha='center', fontsize=9, fontweight='bold')

            # --- 4. INDICADORES DINÁMICOS Y SETPOINT ---
            # Línea de Setpoint roja
            ax_t.axhline(y=sp_nivel, color='red', ls='--', lw=2, zorder=3)
            ax_t.text(-r_max*2.8, sp_nivel + 0.05, f"SETPOINT: {sp_nivel:.2f}m", color='red', fontweight='bold', fontsize=9)

            # Burbuja de Nivel Actual superior
            ax_t.text(0, h_total * 1.2, f"NIVEL ACTUAL: {h_corrida:.3f} m", 
                     ha='center', va='center', fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='#1a5276', boxstyle='round,pad=0.5', lw=2))

            # Renderizado final
            placeholder_tanque.pyplot(fig_t)
            plt.close(fig_t)
            # C. Tendencia de Nivel 
            fig_tr, ax_tr = plt.subplots(figsize=(8, 3.5))
            
            # Graficamos el nivel con etiqueta para la leyenda
            ax_tr.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2, label='Nivel del Tanque (h)')
            
            # Graficamos la consigna (Setpoint) con etiqueta
            ax_tr.axhline(y=sp_nivel, color='red', ls='--', alpha=0.5, label='Consigna (Setpoint)')
            
            # --- CONFIGURACIÓN DE LEYENDA Y EJES ---
            ax_tr.set_xlabel('Tiempo [s]', fontsize=10, fontweight='bold')
            ax_tr.set_ylabel('Altura [m]', fontsize=10, fontweight='bold')
            # --- COMPARACIÓN CON DATOS EXPERIMENTALES (UCV) ---
        # --- COMPARACIÓN CON DATOS EXPERIMENTALES (UCV) ---
        if mostrar_ref:
            t_usr = datos_usr["Tiempo (s)"]
            # La conversión ocurre aquí internamente:
            h_usr = [x / 100 for x in datos_usr["Nivel Medido (m)"]]
            
            # Cambiamos el label para que sea más limpio
            #ax_tr.scatter(t_usr, h_usr, color='red', marker='x', s=100, 
                          label='Datos Experimentales', zorder=5)
            
            # Quitamos el label de la línea para no repetir en la leyenda
            #ax_tr.plot(t_usr, h_usr, color='red', linestyle='--', alpha=0.3) 

        # Configuración final de la leyenda
            ax_tr.legend(loc='upper right', frameon=True, fontsize='x-small')
            # Línea punteada de referencia
            #ax_tr.plot(t_usr, h_usr, color='red', linestyle='--', alpha=0.3)
            #ax_tr.legend(loc='upper right', frameon=True, fontsize='x-small')
            
            ax_tr.set_xlim(0, tiempo_ensayo)
            ax_tr.set_ylim(0, h_total*1.1)
            ax_tr.grid(True, alpha=0.2)
            
            placeholder_grafico.pyplot(fig_tr)
            plt.close(fig_tr)
            
            # D. Acción de Control
            fig_u, ax_u = plt.subplots(figsize=(8, 2.5))
            ax_u.step(vector_t[:i+1], u_log, color='#e67e22', where='post')
            ax_u.set_xlim(0, tiempo_ensayo)
            # El eje Y se ajusta al valor máximo de flujo detectado + un margen del 20%
            techo_dinamico = max(max(u_log), 0.1) * 1.2 if u_log else 0.7
            ax_u.set_ylim(0, techo_dinamico)
            ax_u.grid(True, alpha=0.2)
            ax_u.set_xlabel('Tiempo [s]', fontsize=10, fontweight='bold')
            ax_u.set_ylabel('Flujo [m3/s]', fontsize=10, fontweight='bold')
            placeholder_u.pyplot(fig_u)
            # --- LÓGICA DE LA VÁLVULA ---
            fig_v, ax_v = plt.subplots(figsize=(8, 3))
            
            # Dibujamos la apertura (u_log) en color verde
            ax_v.plot(vector_t[:i+1], u_log, color='#2ecc71', lw=2.5, label='Apertura Real')
            ax_v.fill_between(vector_t[:i+1], u_log, color='#2ecc71', alpha=0.15)
            
            # Configuramos los límites para que se vea claro el On/Off
            ax_v.set_ylim(-0.1, 1.1) 
            ax_v.set_yticks([0, 0.5, 1])
            ax_v.set_yticklabels(['CERRADA (0%)', '50%', 'ABIERTA (100%)'])
            
            # Estética profesional para la UCV
            ax_v.set_title("Comportamiento Dinámico de la Válvula de Control", fontsize=10, fontweight='bold')
            ax_v.grid(True, axis='y', ls='--', alpha=0.5)
            ax_v.set_xlabel("Tiempo de simulación [s]")
            
            # Mostramos en el espacio creado
            placeholder_valvula.pyplot(fig_v)
            # --- PEGAR AQUÍ: GRÁFICA COMPARATIVA ---
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
            plt.close(fig_v) # Importante cerrar para no saturar la memoria
            
           
           
            
            plt.close(fig_u)
        
        time.sleep(0.01) 
        barra_p.progress((i+1)/len(vector_t))

    status_placeholder.empty()
    st.success(f"✅ Simulación del Tanque {geom_tanque} completada.")
    st.balloons()

  # =============================================================================
    # 7. ANÁLISIS DE RESPUESTA TRANSITORIA (AMPLITUD VS TIEMPO)
    # =============================================================================
    st.markdown("---")
    st.subheader("📈 Análisis de Respuesta al Escalón (Amplitud vs. Tiempo)")

    col_an1, col_an2 = st.columns([2, 1])

    with col_an1:
        fig_amp, ax_amp = plt.subplots(figsize=(10, 5))
        # Usamos los datos recolectados en la simulación
        ax_amp.plot(vector_t, h_log, color='#1f77b4', lw=2.5, label='Respuesta del Sistema (PV)')
        ax_amp.step(vector_t, sp_log, color='#d62728', linestyle='--', lw=2, label='Referencia (SP)')
        
        ax_amp.set_title("Respuesta Transitoria del Lazo de Control (MatLab Style)", fontsize=12)
        ax_amp.set_xlabel("Tiempo (s)")
        ax_amp.set_ylabel("Amplitud (m)")
        ax_amp.grid(True, which='both', linestyle='--', alpha=0.5)
        ax_amp.legend(loc='lower right')
        
        # Banda de estabilidad técnica (±5%)
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
    st.markdown("---")
    st.success(f"✅ Simulación del Tanque {geom_tanque} completada exitosamente.")
    st.balloons()
    
    # 1. Crear el DataFrame único
    df_final = pd.DataFrame({
        "Tiempo [s]": vector_t, 
        "Nivel [m]": h_log, 
        "Control [m3/s]": u_log,
        "Error [m]": e_log
    })
    
    # 2. Mostrar la tabla y métricas de cierre
    st.subheader("📋 Resumen de Datos y Estabilidad")
    
    col_tab, col_res = st.columns([2, 1])
    
    with col_tab:
        st.dataframe(df_final.tail(10).style.format("{:.4f}"), use_container_width=True)
    
    with col_res:
        err_f = abs(sp_nivel - h_log[-1])
        st.metric("Error Residual Final", f"{err_f:.4f} m")
        
        # El botón de descarga ahora usa el DataFrame ya creado
        st.download_button(
            label="📥 Descargar Reporte (CSV)", 
            data=df_final.to_csv(index=False), 
            file_name=f"resultados_tesis_{geom_tanque}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Validación final de estado estacionario
    if err_f < 0.05:
        st.success(f"✅ El sistema alcanzó el estado estacionario en {h_log[-1]:.3f} m.")
    else:
        st.warning(f"⚠️ El sistema presenta un error residual de {err_f:.3f} m. Se sugiere ajustar Kp/Ki.")
