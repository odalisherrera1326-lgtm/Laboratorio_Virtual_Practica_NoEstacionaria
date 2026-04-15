import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import base64
from sklearn.metrics import mean_squared_error

# =============================================================================
# --- CONFIGURACIÓN DE LA PÁGINA (debe ir primero) ---
# =============================================================================
st.set_page_config(
    page_title="Lab Virtual - LOU I y LOU II - EIQ UCV",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# --- ESTILOS CSS ---
# =============================================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4f8 0%, #e8edf2 100%);
}

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

.practica-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 6px solid #f1c40f;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.practica-card:hover {
    transform: translateY(-5px);
}

.practica-card h3 {
    color: #1a5276;
}

.practica-card .badge {
    display: inline-block;
    background: linear-gradient(90deg, #1a5276, #2471a3);
    color: white;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
}

.header-container {
    background: linear-gradient(135deg, #0d3251 0%, #1a5276 50%, #1f618d 100%);
    border-radius: 20px;
    padding: 20px 25px;
    margin-bottom: 30px;
}

.stButton > button {
    background: linear-gradient(90deg, #1a5276, #2471a3) !important;
    color: white !important;
    border-radius: 25px !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a5276 0%, #154360 100%) !important;
    border-right: 4px solid #f1c40f !important;
}

[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] p {
    color: #f0f4f8 !important;
}

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
# --- FUNCIONES DE UTILIDAD ---
# =============================================================================
def get_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def mostrar_encabezado():
    logo_ucv_64 = get_base64("logo_ucv.png")
    logo_eiq_64 = get_base64("logoquimicaborde.png")
    
    st.markdown(f"""
    <div class="header-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="width: 120px;">
                {f'<img src="data:image/png;base64,{logo_ucv_64}" width="100">' if logo_ucv_64 else "UCV"}
            </div>
            <div>
                <h1 style="color: white !important; font-size: 1.8rem; margin: 0;">Laboratorio de Operaciones Unitarias</h1>
                <p style="color: #d4e6f1 !important; margin: 0;">Escuela de Ingeniería Química | UCV</p>
            </div>
            <div style="width: 160px;">
                {f'<img src="data:image/png;base64,{logo_eiq_64}" width="150">' if logo_eiq_64 else "EIQ"}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# --- TUS FUNCIONES ORIGINALES (copia y pega todo tu código aquí) ---
# =============================================================================

def get_area_transversal(geom, r, h, h_total):
    h_efectiva = max(h, 0.001)
    if geom == "Cilíndrico":
        return np.pi * (r ** 2)
    elif geom == "Cónico":
        radio_actual = (r / h_total) * h_efectiva
        return np.pi * (radio_actual ** 2)
    else:
        if h_efectiva <= 2 * r:
            radio_corte = np.sqrt(r**2 - (h_efectiva - r)**2)
            return np.pi * (radio_corte ** 2)
        else:
            return np.pi * (r ** 2)


def calcular_pid_adaptativo(geom, r_max, h_total, cd=0.61, area_orificio=0.0005):
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
    else:
        kp = (area_max * 0.6) * 2.0
        ki = kp / 18.0
        kd = kp * 0.2
    factor_ajuste = np.clip(1.0 / (cd * area_orificio * 1000), 0.5, 2.0)
    kp = kp * factor_ajuste
    ki = ki * factor_ajuste * 0.8
    return round(kp, 2), round(ki, 3), round(kd, 3)


def sintonizar_controlador_robusto(geom, r, h_t, cd_calculado, area_ori, op_tipo="Llenado"):
    if geom == "Cilíndrico":
        area_t = np.pi * (r**2)
    elif geom == "Cónico":
        area_t = np.pi * (r/2)**2
    else:
        area_t = (2/3) * np.pi * (r**2)
    
    Kc = 10.0 * area_t
    Kc = np.clip(Kc, 8.0, 25.0)
    
    if op_tipo == "Llenado":
        kp = Kc * 1.2
        ki = kp / 5.0
        kd = kp * 0.15
    else:
        kp = Kc * 1.0
        ki = kp / 6.0
        kd = kp * 0.12
    
    kp = np.clip(kp, 12.0, 30.0)
    ki = np.clip(ki, 2.5, 8.0)
    kd = np.clip(kd, 0.5, 2.5)
    return round(kp, 2), round(ki, 3), round(kd, 2)


def calcular_cd_inteligente(df_usr, r, h_t, geom, area_ori):
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
        else:
            v1 = (np.pi*(h1**2)/3)*(3*r-h1)
            v2 = (np.pi*(h2**2)/3)*(3*r-h2)
        q_real = abs(v1 - v2) / dt
        h_prom = (h1 + h2) / 2
        q_teorico = area_ori * np.sqrt(2 * 9.81 * max(h_prom, 0.001))
        cd_result = q_real / q_teorico if q_teorico > 0 else 0.61
        return float(np.clip(cd_result, 0.4, 1.0))
    except:
        return 0.61


def resolver_sistema_robusto(dt, h_prev, sp, geom, r, h_t, q_p_val, e_sum, e_prev, modo_op, cd_val, kp, ki, kd, d_pulgadas):
    area_h = get_area_transversal(geom, r, h_prev, h_t)
    area_h = max(area_h, 0.0001)
    err = sp - h_prev
    a_o = np.pi * ((d_pulgadas * 0.0254) / 2)**2
    q_max = 2.0
    
    P = kp * err
    e_sum += err * dt
    e_sum = np.clip(e_sum, -50.0, 50.0)
    I = ki * e_sum
    D = kd * (err - e_prev) / dt if dt > 0 else 0
    D = np.clip(D, -5.0, 5.0)
    u_control = P + I + D
    
    if modo_op == "Llenado":
        q_entrada = np.clip(u_control, 0, q_max)
        q_salida = cd_val * a_o * np.sqrt(2 * 9.81 * max(h_prev, 0.005)) if h_prev > 0.005 else 0
        dh_dt = (q_entrada + q_p_val - q_salida) / area_h
        u_graficar = q_entrada
    else:
        q_salida = np.clip(-u_control, 0, q_max)
        q_entrada = q_p_val
        dh_dt = (q_entrada - q_salida) / area_h
        u_graficar = q_salida
    
    h_next = np.clip(h_prev + dh_dt * dt, 0, h_t)
    return h_next, u_graficar, err, e_sum, err


# =============================================================================
# --- FUNCIÓN DEL SIMULADOR (AQUÍ VA TODO TU CÓDIGO ORIGINAL) ---
# =============================================================================
def mostrar_simulador():
    """Simulador completo - TU CÓDIGO ORIGINAL VA AQUÍ"""
    
    # =========================================================================
    # --- TUS VARIABLES Y CONFIGURACIONES ORIGINALES ---
    # =========================================================================
    modo_auto = False
    p_activa = True
    p_magnitud = 0.045
    p_tiempo = 80
    modo_estres = False
    
    if 'ejecutando' not in st.session_state:
        st.session_state.ejecutando = False
    
    # =========================================================================
    # --- BARRA LATERAL (PARÁMETROS) - TU CÓDIGO ORIGINAL ---
    # =========================================================================
    st.sidebar.header("⚙️ Configuración del Sistema")
    
    with st.sidebar.container(border=True):
        op_tipo = st.sidebar.selectbox("Operación Principal", ["Llenado", "Vaciado"])
        geom_tanque = st.sidebar.selectbox("Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])
    
    with st.sidebar.expander("Especificaciones del Tanque", expanded=True):
        r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
        h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
        h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
        sp_nivel = st.slider("Consigna de Nivel (Setpoint) [m]", 0.1, float(h_total), float(h_total/2))
    
    with st.sidebar.expander("Dimensiones de Salida", expanded=True):
        d_pulgadas = st.number_input("Diámetro del Orificio (pulgadas)", value=1.0, min_value=0.1, step=0.1)
        d_metros = d_pulgadas * 0.0254
        area_orificio = np.pi * (d_metros / 2)**2
        st.caption(f"Área calculada: {area_orificio:.6f} m²")
    
    with st.sidebar.expander("🛡️ Escenario de Perturbación ($Q_p$)"):
        p_activa = st.toggle("Simular Falla/Fuga Externas", value=True)
        if p_activa:
            p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.045, format="%.4f")
            p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 80)
            modo_estres = st.toggle("🔥 Activar Modo Estrés")
        else:
            p_magnitud = 0.0
            p_tiempo = 0
            modo_estres = False
    
    with st.sidebar.expander("Parámetros del Controlador PID Robusto"):
        kp_sug, ki_sug, kd_sug = calcular_pid_adaptativo(geom_tanque, r_max, h_total)
        modo_auto = st.checkbox("🎯 Modo Robusto (Auto-sintonía optimizada)", value=True)
        
        st.markdown("---")
        st.subheader("🎛️ Configuración")
        
        if modo_auto:
            st.success("💡 Usando sintonización robusta anti-perturbaciones")
            kp_val = st.number_input("Kp (robusto)", value=kp_sug, key="kp_asist")
            ki_val = st.number_input("Ki (robusto)", value=ki_sug, format="%.3f", key="ki_asist")
            kd_val = st.number_input("Kd (robusto)", value=kd_sug, format="%.3f", key="kd_asist")
            st.caption("✅ Parámetros optimizados para rechazar perturbaciones")
        else:
            st.info("✍️ Modo Manual - Valores recomendados: Kp=7.5, Ki=1.2, Kd=0.4")
            kp_val = st.number_input("Kp", value=7.5, step=0.5, key="kp_man")
            ki_val = st.number_input("Ki", value=1.2, step=0.1, format="%.3f", key="ki_man")
            kd_val = st.number_input("Kd", value=0.4, step=0.1, format="%.3f", key="kd_man")
        
        tiempo_ensayo = st.slider("Tiempo de simulación [s]", 60, 600, 300)
    
    with st.sidebar.expander("📊 Cargar Datos Experimentales"):
        st.write("Ingresa los datos medidos en el laboratorio:")
        st.caption("⚠️ Nota: El nivel debe ingresarse en **centímetros (cm)**")
        
        df_exp_default = pd.DataFrame({
            "Tiempo (s)": [0, 60, 120, 180, 240, 300],
            "Nivel Medido (cm)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        datos_usr = st.data_editor(df_exp_default, num_rows="dynamic")
        mostrar_ref = st.checkbox("Mostrar referencia en gráfica", value=True)
    
    st.sidebar.markdown("---")
    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        iniciar_sim = st.button("▶️ Iniciar Simulación Robusta", use_container_width=True, type="primary")
    with col_btn2:
        btn_reset = st.button("🔄 Reset", use_container_width=True, type="secondary")
    
    if btn_reset:
        st.session_state.ejecutando = False
        st.rerun()
    
    # =========================================================================
    # --- INICIALIZACIÓN DE SIMULACIÓN ---
    # =========================================================================
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
                    cd_calc = calcular_cd_inteligente(df_calib[["Tiempo (s)", "Nivel Medido (m)"]], 
                                                       r_max, h_total, geom_tanque, area_orificio)
                    kp_a, ki_a, kd_a = sintonizar_controlador_robusto(
                        geom_tanque, r_max, h_total, cd_calc, area_orificio, op_tipo
                    )
                    st.session_state['kp_ejecucion'] = kp_a
                    st.session_state['ki_ejecucion'] = ki_a
                    st.session_state['kd_ejecucion'] = kd_a
                    st.session_state['cd_final'] = cd_calc
                    st.toast(f"🎯 Control Robusto Activado: Cd={cd_calc:.2f} | Kp={kp_a} | Ki={ki_a} | Kd={kd_a}")
                else:
                    st.session_state['kp_ejecucion'] = 15.0
                    st.session_state['ki_ejecucion'] = 3.5
                    st.session_state['kd_ejecucion'] = 1.5
                    st.session_state['cd_final'] = 0.61
                    st.info("💡 Usando configuración robusta por defecto (Kp=15.0, Ki=3.5, Kd=1.5)")
            else:
                st.session_state['kp_ejecucion'] = kp_val
                st.session_state['ki_ejecucion'] = ki_val
                st.session_state['kd_ejecucion'] = kd_val
                st.session_state['cd_final'] = 0.61
                st.info(f"✍️ Modo Manual: Kp={kp_val}, Ki={ki_val}, Kd={kd_val}")
        except Exception as e:
            st.session_state['kp_ejecucion'] = 15.0
            st.session_state['ki_ejecucion'] = 3.5
            st.session_state['kd_ejecucion'] = 1.5
            st.session_state['cd_final'] = 0.61
            st.warning(f"⚠️ Usando configuración robusta de emergencia (Kp=15, Ki=3.5, Kd=1.5)")
    
    # =========================================================================
    # --- SIMULACIÓN PRINCIPAL (TU CÓDIGO ORIGINAL COMPLETO) ---
    # =========================================================================
    if not st.session_state.ejecutando:
        st.info("💡 Ajuste los parámetros en la barra lateral y presione 'Iniciar Simulación Robusta' para comenzar.")
    else:
        col_graf, col_met = st.columns([2, 1])
        
        with col_graf:
            st.subheader("Monitor del Proceso - Control Robusto Anti-Perturbaciones")
            placeholder_tanque = st.empty()
            st.subheader("Tendencia Temporal")
            placeholder_grafico = st.empty()
            st.subheader("⚙️ Acción del Controlador")
            placeholder_u = st.empty()
            st.markdown("---")
            st.subheader("⚙️ Estado de Operación: Válvula de Control")
            placeholder_valvula = st.empty()
            st.markdown("---")
            st.subheader("📊 Comparativa: Modelo Teórico vs Datos Experimentales")
            placeholder_comparativa = st.empty()
        
        with col_met:
            st.subheader("Métricas de Control Robusto")
            kp_show = st.session_state.get('kp_ejecucion', 15.0)
            ki_show = st.session_state.get('ki_ejecucion', 3.5)
            cd_show = st.session_state.get('cd_final', 0.61)
            st.write(f"**Parámetros Activos (Robustos):**")
            st.caption(f"Kp: {kp_show} | Ki: {ki_show} | Kd: {st.session_state.get('kd_ejecucion', 1.5)}")
            st.caption(f"Cd: {cd_show:.3f} | Modo: {'Auto-Robusto' if modo_auto else 'Manual'}")
            st.markdown("---")
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
            st.caption("💡 El controlador robusto mantiene el nivel incluso con fugas")
        
        # Preparación de la simulación
        status_placeholder = st.empty()
        dt = 1.0
        vector_t = np.arange(0, tiempo_ensayo, dt)
        h_log, u_log, e_log = [], [], []
        
        if op_tipo == "Llenado":
            h_corrida = 0.001
        else:
            h_corrida = h_total * 0.95
        
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
        
        # Bucle de simulación
        for i, t_act in enumerate(vector_t):
            status_placeholder.markdown("💧 CONTROL ROBUSTO ACTIVADO - PROCESANDO...")
            
            if p_activa and t_act >= p_tiempo:
                if modo_estres:
                    factor = 1.5 if valor_presente < sp_nivel else 0.5
                    q_p_inst = p_magnitud * factor
                else:
                    q_p_inst = p_magnitud
            else:
                q_p_inst = 0.0
            
            k_p = st.session_state.get('kp_ejecucion', 15.0)
            k_i = st.session_state.get('ki_ejecucion', 3.5)
            k_d = st.session_state.get('kd_ejecucion', 1.5)
            
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
            
            m_h.metric("Nivel PV [m]", f"{valor_presente:.3f}")
            m_e.metric("Error [m]", f"{error_presente:.4f}")
            placeholder_iae.metric("IAE", f"{iae_acumulado:.2f}")
            placeholder_itae.metric("ITAE", f"{itae_acumulado:.2f}")
            
            # ========== DIBUJAR TANQUE ==========
            fig_t, ax_t = plt.subplots(figsize=(7, 5))
            ax_t.set_axis_off()
            ax_t.set_xlim(-r_max*3, r_max*3)
            ax_t.set_ylim(-0.8, h_total*1.3)
            
            color_agua = '#3498db'
            
            if geom_tanque == "Cilíndrico":
                c_in_x, c_in_y = -r_max, h_total*0.8
                c_out_x, c_out_y = r_max, 0.1
                ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=5, zorder=2)
                ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, valor_presente, color=color_agua, alpha=0.85, zorder=1, edgecolor='#2980b9', linewidth=1.5))
                if valor_presente > 0 and valor_presente < h_total:
                    ax_t.axhline(y=valor_presente, color='white', linestyle='-', linewidth=2, alpha=0.8, zorder=3)
                
            elif geom_tanque == "Cónico":
                c_in_x, c_in_y = -(r_max/h_total)*(h_total*0.8), h_total*0.8
                c_out_x, c_out_y = 0, 0
                ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=5, zorder=2)
                if valor_presente > 0:
                    radio_superficie = (r_max / h_total) * valor_presente
                    vertices = [[-radio_superficie, valor_presente], [radio_superficie, valor_presente], [0, 0]]
                    ax_t.add_patch(plt.Polygon(vertices, color=color_agua, alpha=0.85, zorder=1, edgecolor='#2980b9', linewidth=1.5))
                    ax_t.plot([-radio_superficie, radio_superficie], [valor_presente, valor_presente], color='white', linewidth=2, alpha=0.8, zorder=3)
                
            else:
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
            
            # Válvulas
            ax_t.add_patch(plt.Rectangle((c_in_x - 1.5, c_in_y - 0.1), 1.5, 0.2, color='silver', zorder=0))
            ax_t.add_patch(plt.Polygon([[c_in_x-1, c_in_y+0.2], [c_in_x-1, c_in_y-0.2], [c_in_x-0.6, c_in_y]], color='#2c3e50', zorder=2))
            ax_t.add_patch(plt.Polygon([[c_in_x-0.2, c_in_y+0.2], [c_in_x-0.2, c_in_y-0.2], [c_in_x-0.6, c_in_y]], color='#2c3e50', zorder=2))
            ax_t.text(c_in_x-0.6, c_in_y+0.4, "V-01", ha='center', fontsize=9, fontweight='bold')
            
            if geom_tanque == "Cilíndrico":
                ax_t.add_patch(plt.Rectangle((c_out_x, c_out_y - 0.1), 1.5, 0.2, color='silver', zorder=0))
                vs_x, vs_y = c_out_x + 0.8, c_out_y
            else:
                ax_t.add_patch(plt.Rectangle((c_out_x - 0.1, -0.6), 0.2, 0.6, color='silver', zorder=0))
                vs_x, vs_y = c_out_x, -0.4
            
            ax_t.add_patch(plt.Polygon([[vs_x-0.25, vs_y+0.2], [vs_x-0.25, vs_y-0.2], [vs_x, vs_y]], color='#2c3e50', zorder=2))
            ax_t.add_patch(plt.Polygon([[vs_x+0.25, vs_y+0.2], [vs_x+0.25, vs_y-0.2], [vs_x, vs_y]], color='#2c3e50', zorder=2))
            offset_t = 0.4 if geom_tanque == "Cilíndrico" else 0
            ax_t.text(vs_x + offset_t, vs_y - 0.5, "V-02 (CV)", ha='center', fontsize=9, fontweight='bold')
            
            ax_t.axhline(y=sp_nivel, color='red', ls='--', lw=2, zorder=3, alpha=0.8)
            ax_t.text(-r_max*2.8, sp_nivel + 0.05, f"SETPOINT: {sp_nivel:.2f}m", color='red', fontweight='bold', fontsize=9)
            ax_t.text(0, h_total * 1.2, f"PV: {valor_presente:.3f} m", ha='center', va='center', fontsize=11, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='#1a5276', boxstyle='round,pad=0.5', lw=2))
            
            placeholder_tanque.pyplot(fig_t)
            plt.close(fig_t)
            
           
