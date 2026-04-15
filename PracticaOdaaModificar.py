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
    initial_sidebar_state="collapsed"  # Sidebar colapsado inicialmente
)

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
                <p style="color: #d4e6f1 !important; margin: 0;">Escuela de Ingeniería Química | Facultad de Ingeniería - UCV</p>
            </div>
            <div style="width: 160px;">
                {f'<img src="data:image/png;base64,{logo_eiq_64}" width="150">' if logo_eiq_64 else "EIQ"}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# --- FUNCIONES DEL SIMULADOR (tu código original adaptado) ---
# =============================================================================

# [TODAS TUS FUNCIONES ORIGINALES VAN AQUÍ]
# get_area_transversal, calcular_pid_adaptativo, sintonizar_controlador_robusto,
# calcular_cd_inteligente, resolver_sistema_robusto

# Por razones de longitud, pondré las funciones esenciales:
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
# --- FUNCIÓN PRINCIPAL DEL SIMULADOR (tu código original) ---
# =============================================================================
def mostrar_simulador():
    """Función que contiene todo el simulador original"""
    
    # Inicializar estado
    if 'ejecutando' not in st.session_state:
        st.session_state.ejecutando = False
    
    # Barra lateral con parámetros
    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")
        
        with st.container(border=True):
            op_tipo = st.selectbox("Operación Principal", ["Llenado", "Vaciado"])
            geom_tanque = st.selectbox("Geometría del Equipo", ["Cilíndrico", "Cónico", "Esférico"])
        
        with st.expander("Especificaciones del Tanque", expanded=True):
            r_max = st.number_input("Radio de Diseño (R) [m]", value=1.0, min_value=0.1, step=0.1)
            h_sug = 3.0 if geom_tanque != "Esférico" else r_max * 2
            h_total = st.number_input("Altura de Diseño (H) [m]", value=float(h_sug), min_value=0.1, step=0.5)
            sp_nivel = st.slider("Consigna de Nivel (Setpoint) [m]", 0.1, float(h_total), float(h_total/2))
        
        with st.expander("Dimensiones de Salida", expanded=True):
            d_pulgadas = st.number_input("Diámetro del Orificio (pulgadas)", value=1.0, min_value=0.1, step=0.1)
            area_orificio = np.pi * ((d_pulgadas * 0.0254) / 2)**2
            st.caption(f"Área calculada: {area_orificio:.6f} m²")
        
        with st.expander("🛡️ Escenario de Perturbación"):
            p_activa = st.toggle("Simular Falla/Fuga Externas", value=True)
            if p_activa:
                p_magnitud = st.number_input("Magnitud Qp [m³/s]", value=0.045, format="%.4f")
                p_tiempo = st.slider("Inicio de perturbación [s]", 0, 500, 80)
                modo_estres = st.toggle("🔥 Activar Modo Estrés")
            else:
                p_magnitud, p_tiempo, modo_estres = 0.0, 0, False
        
        with st.expander("Parámetros del Controlador PID Robusto"):
            kp_sug, ki_sug, kd_sug = calcular_pid_adaptativo(geom_tanque, r_max, h_total)
            modo_auto = st.checkbox("🎯 Modo Robusto (Auto-sintonía optimizada)", value=True)
            
            if modo_auto:
                kp_val = st.number_input("Kp (robusto)", value=kp_sug, key="kp_asist")
                ki_val = st.number_input("Ki (robusto)", value=ki_sug, format="%.3f", key="ki_asist")
                kd_val = st.number_input("Kd (robusto)", value=kd_sug, format="%.3f", key="kd_asist")
            else:
                kp_val = st.number_input("Kp", value=15.0, step=1.0, key="kp_man")
                ki_val = st.number_input("Ki", value=3.0, step=0.5, format="%.3f", key="ki_man")
                kd_val = st.number_input("Kd", value=1.5, step=0.2, format="%.3f", key="kd_man")
            
            tiempo_ensayo = st.slider("Tiempo de simulación [s]", 60, 600, 300)
        
        with st.expander("📊 Cargar Datos Experimentales"):
            df_exp_default = pd.DataFrame({
                "Tiempo (s)": [0, 60, 120, 180, 240, 300],
                "Nivel Medido (cm)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            })
            datos_usr = st.data_editor(df_exp_default, num_rows="dynamic")
            mostrar_ref = st.checkbox("Mostrar referencia en gráfica", value=True)
        
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            iniciar_sim = st.button("▶️ Iniciar Simulación", use_container_width=True, type="primary")
        with col_btn2:
            btn_reset = st.button("🔄 Reset", use_container_width=True, type="secondary")
        
        if btn_reset:
            st.session_state.ejecutando = False
            st.rerun()
    
    # Inicialización de simulación
    if iniciar_sim:
        st.session_state.ejecutando = True
        st.session_state['kp_ejecucion'] = kp_val
        st.session_state['ki_ejecucion'] = ki_val
        st.session_state['kd_ejecucion'] = kd_val
        st.session_state['cd_final'] = 0.61
    
    # Mostrar simulador
    if not st.session_state.ejecutando:
        st.info("💡 Ajuste los parámetros en la barra lateral y presione 'Iniciar Simulación' para comenzar.")
    else:
        col_graf, col_met = st.columns([2, 1])
        
        with col_graf:
            st.subheader("📊 Monitor del Proceso")
            placeholder_tanque = st.empty()
            st.subheader("📈 Tendencia Temporal")
            placeholder_grafico = st.empty()
            st.subheader("⚙️ Acción del Controlador")
            placeholder_u = st.empty()
            st.subheader("📊 Comparativa: Modelo vs Datos Experimentales")
            placeholder_comparativa = st.empty()
        
        with col_met:
            st.subheader("📊 Métricas de Control")
            kp_show = st.session_state.get('kp_ejecucion', 15.0)
            ki_show = st.session_state.get('ki_ejecucion', 3.0)
            st.metric("Kp", f"{kp_show:.2f}")
            st.metric("Ki", f"{ki_show:.3f}")
            st.metric("Kd", f"{st.session_state.get('kd_ejecucion', 1.5):.2f}")
            st.markdown("---")
            placeholder_iae = st.empty()
            placeholder_itae = st.empty()
            placeholder_iae.metric("IAE", "0.00")
            placeholder_itae.metric("ITAE", "0.00")
            st.markdown("---")
            m_h = st.empty()
            m_e = st.empty()
            m_h.metric("Nivel [m]", "0.000")
            m_e.metric("Error [m]", "0.000")
        
        # Preparación de la simulación
        dt = 1.0
        vector_t = np.arange(0, tiempo_ensayo, dt)
        h_log, u_log, e_log = [], [], []
        
        if op_tipo == "Llenado":
            h_corrida = 0.001
        else:
            h_corrida = h_total * 0.95
        
        valor_presente = h_corrida
        err_int, err_pasado = 0.0, 0.0
        iae_acumulado, itae_acumulado = 0.0, 0.0
        
        # Procesar datos experimentales
        if "Nivel Medido (cm)" in datos_usr.columns and len(datos_usr) > 0:
            t_exp = datos_usr["Tiempo (s)"].values
            h_exp = [val / 100 for val in datos_usr["Nivel Medido (cm)"].values]
            tiene_datos_exp = True
        else:
            t_exp, h_exp, tiene_datos_exp = [], [], False
        
        barra_p = st.progress(0)
        cd_para_simular = st.session_state.get('cd_final', 0.61)
        
        for i, t_act in enumerate(vector_t):
            # Lógica de perturbación
            if p_activa and t_act >= p_tiempo:
                q_p_inst = p_magnitud
            else:
                q_p_inst = 0.0
            
            k_p = st.session_state.get('kp_ejecucion', 15.0)
            k_i = st.session_state.get('ki_ejecucion', 3.0)
            k_d = st.session_state.get('kd_ejecucion', 1.5)
            
            h_corrida, u_inst, e_inst, err_int, err_pasado = resolver_sistema_robusto(
                dt, h_corrida, sp_nivel, geom_tanque, r_max, h_total, q_p_inst,
                err_int, err_pasado, op_tipo, cd_para_simular,
                k_p, k_i, k_d, d_pulgadas
            )
            
            valor_presente = h_corrida
            iae_acumulado += abs(e_inst) * dt
            itae_acumulado += t_act * abs(e_inst) * dt
            
            h_log.append(h_corrida)
            u_log.append(u_inst)
            e_log.append(e_inst)
            
            # Actualizar métricas
            m_h.metric("Nivel [m]", f"{valor_presente:.3f}")
            m_e.metric("Error [m]", f"{e_inst:.4f}")
            placeholder_iae.metric("IAE", f"{iae_acumulado:.2f}")
            placeholder_itae.metric("ITAE", f"{itae_acumulado:.2f}")
            
            # Dibujar tanque
            fig_t, ax_t = plt.subplots(figsize=(6, 4))
            ax_t.set_axis_off()
            ax_t.set_xlim(-r_max*2.5, r_max*2.5)
            ax_t.set_ylim(-0.5, h_total*1.2)
            
            color_agua = '#3498db'
            
            if geom_tanque == "Cilíndrico":
                ax_t.plot([-r_max, -r_max, r_max, r_max], [h_total, 0, 0, h_total], color='#2c3e50', lw=4, zorder=2)
                ax_t.add_patch(plt.Rectangle((-r_max, 0), 2*r_max, valor_presente, color=color_agua, alpha=0.85, zorder=1))
                if valor_presente > 0 and valor_presente < h_total:
                    ax_t.axhline(y=valor_presente, color='white', linestyle='-', linewidth=2, alpha=0.8)
            elif geom_tanque == "Cónico":
                ax_t.plot([-r_max, 0, r_max], [h_total, 0, h_total], color='#2c3e50', lw=4, zorder=2)
                if valor_presente > 0:
                    radio_superficie = (r_max / h_total) * valor_presente
                    vertices = [[-radio_superficie, valor_presente], [radio_superficie, valor_presente], [0, 0]]
                    ax_t.add_patch(plt.Polygon(vertices, color=color_agua, alpha=0.85, zorder=1))
                    ax_t.plot([-radio_superficie, radio_superficie], [valor_presente, valor_presente], color='white', linewidth=2, alpha=0.8)
            else:  # Esférico
                import math
                agua_esf = plt.Circle((0, r_max), r_max, color=color_agua, alpha=0.85, zorder=1)
                ax_t.add_patch(agua_esf)
                recorte_nivel = plt.Rectangle((-r_max, 0), 2*r_max, valor_presente, transform=ax_t.transData)
                agua_esf.set_clip_path(recorte_nivel)
                ax_t.add_patch(plt.Circle((0, r_max), r_max, color='#2c3e50', fill=False, lw=4, zorder=2))
                if valor_presente > 0 and valor_presente < 2*r_max:
                    radio_nivel = math.sqrt(r_max**2 - (valor_presente - r_max)**2)
                    ax_t.plot([-radio_nivel, radio_nivel], [valor_presente, valor_presente], color='white', linewidth=2, alpha=0.8)
            
            ax_t.axhline(y=sp_nivel, color='red', ls='--', lw=2, alpha=0.8)
            ax_t.text(0, h_total * 1.1, f"PV: {valor_presente:.3f} m", ha='center', fontsize=10, fontweight='bold')
            
            placeholder_tanque.pyplot(fig_t)
            plt.close(fig_t)
            
            # Gráfico de tendencia
            fig_tr, ax_tr = plt.subplots(figsize=(8, 3))
            ax_tr.plot(vector_t[:i+1], h_log, color='#2980b9', lw=2)
            ax_tr.axhline(y=sp_nivel, color='red', ls='--', alpha=0.5)
            ax_tr.set_xlabel('Tiempo [s]')
            ax_tr.set_ylabel('Altura [m]')
            ax_tr.set_xlim(0, tiempo_ensayo)
            ax_tr.set_ylim(0, h_total * 1.1)
            ax_tr.grid(True, alpha=0.2)
            placeholder_grafico.pyplot(fig_tr)
            plt.close(fig_tr)
            
            # Gráfico de control
            fig_u, ax_u = plt.subplots(figsize=(8, 2.5))
            ax_u.plot(vector_t[:i+1], u_log, color='#e67e22', lw=2)
            ax_u.set_xlim(0, tiempo_ensayo)
            ax_u.set_ylim(0, max(max(u_log), 0.5) * 1.2 if u_log else 0.7)
            ax_u.grid(True, alpha=0.2)
            ax_u.set_xlabel('Tiempo [s]')
            ax_u.set_ylabel('Flujo [m³/s]')
            placeholder_u.pyplot(fig_u)
            plt.close(fig_u)
            
            # Gráfico comparativo
            fig_comp, ax_comp = plt.subplots(figsize=(8, 3))
            ax_comp.plot(vector_t[:i+1], h_log, color='#1f77b4', lw=2, label='Simulación')
            if mostrar_ref and tiene_datos_exp:
                ax_comp.scatter(t_exp, h_exp, color='red', marker='x', s=80, label='Experimental')
            ax_comp.set_xlabel("Tiempo [s]")
            ax_comp.set_ylabel("Nivel [m]")
            ax_comp.set_ylim(0, h_total * 1.1)
            ax_comp.grid(True, alpha=0.3)
            ax_comp.legend()
            placeholder_comparativa.pyplot(fig_comp)
            plt.close(fig_comp)
            
            time.sleep(0.01)
            barra_p.progress((i+1)/len(vector_t))
        
        st.success("✅ Simulación completada")
        st.balloons()


# =============================================================================
# --- PÁGINA DE INICIO Y BIBLIOTECA DE PRÁCTICAS ---
# =============================================================================
def pagina_inicio():
    """Página de inicio con las tarjetas de prácticas"""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #1a5276;">Bienvenido al Laboratorio Virtual de Operaciones Unitarias</h2>
        <p style="color: #5d6d7e;">Seleccione una práctica para comenzar</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tarjetas de prácticas para LOU I
    st.markdown("### 📚 Laboratorio de Operaciones Unitarias I (LOU I)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="practica-card" onclick="window.location.href='?practica=1'">
            <h3>🏭 Práctica N° 1</h3>
            <p>Balance de masa en estado no estacionario<br>Control PID de nivel en tanques</p>
            <span class="badge">🔬 Activa</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ir a Práctica 1", key="p1", use_container_width=True):
            st.session_state.pagina_actual = "simulador"
            st.session_state.practica_actual = "balance_masa"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="practica-card">
            <h3>⚡ Práctica N° 2</h3>
            <p>Bombas centrífugas en serie y paralelo<br>Curvas características</p>
            <span class="badge">🚧 Próximamente</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="practica-card">
            <h3>🌊 Práctica N° 3</h3>
            <p>Flujo en tuberías<br>Pérdidas de carga y accesorios</p>
            <span class="badge">🚧 Próximamente</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tarjetas de prácticas para LOU II
    st.markdown("### 📚 Laboratorio de Operaciones Unitarias II (LOU II)")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="practica-card">
            <h3>🔥 Práctica N° 1</h3>
            <p>Intercambiadores de calor<br>Coeficientes globales de transferencia</p>
            <span class="badge">🚧 Próximamente</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="practica-card">
            <h3>💨 Práctica N° 2</h3>
            <p>Destilación batch<br>Ecuaciones de Rayleigh</p>
            <span class="badge">🚧 Próximamente</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="practica-card">
            <h3>🧪 Práctica N° 3</h3>
            <p>Absorción de gases<br>Torres empacadas</p>
            <span class="badge">🚧 Próximamente</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# --- PÁGINA DE INFORMACIÓN DE LA PRÁCTICA ---
# =============================================================================
def pagina_info_practica():
    """Página con la guía teórica de la práctica"""
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #1a5276;">Práctica N° 1: Balance de Masa en Estado No Estacionario</h2>
        <p style="color: #5d6d7e;">Control PID Robusto para Rechazo de Perturbaciones</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_info, col_btn = st.columns([3, 1])
    
    with col_info:
        st.markdown(r"""
        ### 📖 Fundamento Teórico
        
        La dinámica del sistema se describe mediante el **Balance Global de Masa**:
        
        $$ A(h) \frac{dh}{dt} = Q_{in} - C_d \cdot a \cdot \sqrt{2gh} \pm Q_p $$
        
        ### 🎯 Objetivos de la Práctica
        
        1. Comprender el balance de masa en sistemas con nivel variable
        2. Implementar un controlador PID con anti-windup
        3. Evaluar el rechazo a perturbaciones en diferentes geometrías
        4. Analizar métricas de desempeño (IAE/ITAE)
        
        ### 📊 Métricas de Evaluación
        
        - **IAE (Integral del Error Absoluto):** $\int |e(t)| dt$
        - **ITAE (Integral del Tiempo por Error Absoluto):** $\int t \cdot |e(t)| dt$
        
        ### ⚙️ Parámetros Sintonizables
        
        | Parámetro | Descripción | Rango típico |
        |---|---|---|
        | Kp | Ganancia proporcional | 5 - 25 |
        | Ki | Ganancia integral | 1 - 8 |
        | Kd | Ganancia derivativa | 0.5 - 2.5 |
        | Cd | Coeficiente de descarga | 0.4 - 1.0 |
        """)
    
    with col_btn:
        if st.button("🚀 Iniciar Simulador", use_container_width=True, type="primary"):
            st.session_state.pagina_actual = "simulador"
            st.rerun()
    
    st.markdown("---")
    
    # Video o diagrama
    st.subheader("📹 Diagrama del Proceso")
    if os.path.exists("Captura de pantalla 2026-03-29 163125.png"):
        st.image("Captura de pantalla 2026-03-29 163125.png", use_container_width=True)
    else:
        st.info("📍 El diagrama del sistema se mostrará aquí.")


# =============================================================================
# --- FUNCIÓN PRINCIPAL CON PESTAÑAS LOU I Y LOU II ---
# =============================================================================
def main():
    """Función principal con la estructura de pestañas"""
    
    # Inicializar estado de página
    if 'pagina_actual' not in st.session_state:
        st.session_state.pagina_actual = "inicio"
    
    # Mostrar encabezado institucional
    mostrar_encabezado()
    
    # Si estamos en el simulador, mostrar solo el simulador
    if st.session_state.pagina_actual == "simulador":
        # Botón para volver al inicio
        if st.button("← Volver al inicio", use_container_width=False):
            st.session_state.pagina_actual = "inicio"
            st.session_state.ejecutando = False
            st.rerun()
        st.markdown("---")
        mostrar_simulador()
        return
    
    # Si estamos en la página de información de la práctica
    if st.session_state.pagina_actual == "info_practica":
        if st.button("← Volver al inicio", use_container_width=False):
            st.session_state.pagina_actual = "inicio"
            st.rerun()
        st.markdown("---")
        pagina_info_practica()
        return
    
    # Página de inicio con pestañas LOU I y LOU II
    tab1, tab2 = st.tabs(["🏭 LABORATORIO DE OPERACIONES UNITARIAS I (LOU I)", 
                          "⚗️ LABORATORIO DE OPERACIONES UNITARIAS II (LOU II)"])
    
    with tab1:
        st.markdown("### 📚 Prácticas Disponibles - LOU I")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="practica-card">
                <h3>🏭 Práctica N° 1</h3>
                <p><strong>Balance de masa en estado no estacionario</strong></p>
                <p>Control PID de nivel en tanques con diferentes geometrías. Análisis de rechazo a perturbaciones.</p>
                <span class="badge">✅ Disponible</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📖 Ver teoría", key="teoria1", use_container_width=True):
                st.session_state.pagina_actual = "info_practica"
                st.rerun()
            if st.button("🚀 Iniciar simulación", key="sim1", use_container_width=True):
                st.session_state.pagina_actual = "simulador"
                st.rerun()
        
        with col2:
            st.markdown("""
            <div class="practica-card">
                <h3>⚡ Práctica N° 2</h3>
                <p><strong>Bombas centrífugas en serie y paralelo</strong></p>
                <p>Curvas características, punto de operación y eficiencia.</p>
                <span class="badge">🚧 En desarrollo</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="practica-card">
                <h3>🌊 Práctica N° 3</h3>
                <p><strong>Flujo en tuberías</strong></p>
                <p>Pérdidas de carga primarias y secundarias, número de Reynolds.</p>
                <span class="badge">🚧 En desarrollo</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("💡 **Práctica 1**: Simulación de control PID robusto para tanques con geometrías cilíndrica, cónica y esférica. Incluye análisis de perturbaciones y métricas IAE/ITAE.")
    
    with tab2:
        st.markdown("### 📚 Prácticas Disponibles - LOU II")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="practica-card">
                <h3>🔥 Práctica N° 1</h3>
                <p><strong>Intercambiadores de calor</strong></p>
                <p>Coeficientes globales de transferencia, MLDT, eficiencia.</p>
                <span class="badge">🚧 Próximamente</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="practica-card">
                <h3>💨 Práctica N° 2</h3>
                <p><strong>Destilación batch</strong></p>
                <p>Ecuaciones de Rayleigh, composición de destilado y residuo.</p>
                <span class="badge">🚧 Próximamente</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="practica-card">
                <h3>🧪 Práctica N° 3</h3>
                <p><strong>Absorción de gases</strong></p>
                <p>Torres empacadas, altura de la unidad de transferencia.</p>
                <span class="badge">🚧 Próximamente</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Universidad Central de Venezuela - Escuela de Ingeniería Química</p>
        <p>Laboratorio de Operaciones Unitarias I y II | Simulador Interactivo | © 2025</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# --- EJECUCIÓN PRINCIPAL ---
# =============================================================================
if __name__ == "__main__":
    main()
