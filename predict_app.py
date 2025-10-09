import streamlit as st
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# --- Konfiguration MUSS ZUERST gehen ---
st.set_page_config(
    page_title="Análisis de Riesgo Crediticio",
    page_icon="🇩🇪",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Globale Variablen ---
# Columnas EXACTAS que espera el modelo. ¡Sin desviaciones!
TRAINING_COLUMNS = [
    'status', 'duration', 'credit_history', 'purpose', 'amount', 
    'savings', 'employment_duration', 'installment_rate', 
    'personal_status_sex', 'other_debtors', 'present_residence', 
    'property', 'age', 'other_installment_plans', 'housing', 
    'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker'
]

# --- Función de Carga del Modelo ---
@st.cache_resource
def load_model():
    """Carga el modelo con la máxima eficiencia y ORDEN."""
    model_path = 'models/xgboost_model.pkl'
    
    if not os.path.exists(model_path):
        return None, "¡Achtung! Modelo no encontrado."
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return model, f"Cargado ({size_mb:.2f} MB). ¡Eficiente!"
    
    except Exception as e:
        return None, f"Error grave: {str(e)}"


# --- Función de Preprocesamiento ---
def preprocess_input(data: dict):
    """Preprocesa los datos de entrada. ¡Todo debe ser correcto!"""
    try:
        df = pd.DataFrame([data])
        df_encoded = pd.get_dummies(df, drop_first=True)
        final_df = pd.DataFrame(0.0, index=[0], columns=TRAINING_COLUMNS, dtype=float)
        
        for col in df_encoded.columns:
            if col in TRAINING_COLUMNS:
                final_df.loc[0, col] = float(df_encoded[col].values[0])
        
        return final_df
    
    except Exception as e:
        st.error(f"Error en el preprocesamiento: {e}. ¡Esto no está en ORDEN!")
        return None


# --- Predicción Simulada ---
def simulate_prediction(data):
    """Reglas de scoring simples. ¡Pura lógica!"""
    score = 0
    if data['status'] >= 3: score += 2
    if data['duration'] <= 24: score += 1
    if data['credit_history'] <= 2: score += 2
    if data['amount'] <= 5000: score += 1
    if data['age'] >= 25: score += 1
    return (1 if score >= 4 else 0), score


# ==================== DIE ANWENDUNG (La Aplicación) ====================

st.title("🇩🇪 ¡Achtung! Análisis de Riesgo Crediticio")
st.markdown("Wilkommen! Gracias por elejir nuestro banco. Nos complace evaluar su solicitud de crédito con la máxima eficiencia.")
st.markdown("---")

# --- Sidebar (Panel de Kontrol) ---
with st.sidebar:
    st.header("📊 Panel de Kontrol")
    
    model, msg = load_model()
    
    if model:
        st.success("🟢 Modelo Operativo")
        st.caption("¡Precisión garantizada!")
    else:
        st.warning("🟡 Modo Simulación")
        st.caption("No es lo óptimo, pero funcional.")
    
    st.caption(msg)
    st.markdown("---")
    
    # Info de versiones
    try:
        import xgboost as xgb
        st.caption(f"XGBoost: {xgb.__version__}")
    except:
        st.caption("XGBoost: N/A")
    
    st.caption(f"Pandas: {pd.__version__}")
    st.caption(f"Streamlit: {st.__version__}")

# --- Formulario ---
st.info("👤 ¡Rellene el formulario con ORDEN para evaluar el crédito! Schnell!")

with st.form("kredit_form"):
    
    st.subheader("📋 Información del Solicitante")
    
    status_opts = {
        1: '< 0 DM (sobregiro)',
        2: '0 - 200 DM',
        3: '≥ 200 DM',
        4: 'Sin cuenta (Kein Konto)'
    }
    
    history_opts = {
        0: 'Sin créditos previos',
        1: 'Todo pagado puntualmente',
        2: 'Créditos actuales al día',
        3: 'Retrasos en el pasado',
        4: 'Cuenta crítica (¡Sehr schlecht!)'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        status = st.selectbox(
            "🏦 Estado de la Cuenta Corriente",
            options=list(status_opts.keys()),
            format_func=lambda x: status_opts[x]
        )
        duration = st.slider("📅 Duración del Crédito (meses)", 4, 72, 24)
        amount = st.number_input(
            "💰 Monto (DM)", min_value=250, max_value=20000, value=2500, step=250
        )
    
    with col2:
        credit_history = st.selectbox(
            "📊 Historial Crediticio",
            options=list(history_opts.keys()),
            format_func=lambda x: history_opts[x]
        )
        age = st.slider("👤 Edad del Solicitante", 18, 75, 35)
    
    submitted = st.form_submit_button("🔍 ¡Evaluar con Precisión Alemana!", use_container_width=True, type="primary")


# --- Procesamiento (Die Verarbeitung) ---
if submitted:
    
    input_data = {
        'status': float(status), 'duration': float(duration), 'credit_history': float(credit_history),
        'amount': float(amount), 'age': float(age), 'installment_rate': 4.0, 'present_residence': 4.0,
        'number_credits': 1.0, 'people_liable': 1.0, 'purpose': 3.0, 'savings': 1.0,
        'employment_duration': 3.0, 'other_debtors': 1.0, 'property': 3.0,
        'other_installment_plans': 3.0, 'housing': 2.0, 'job': 3.0, 'telephone': 1.0,
        'foreign_worker': 1.0,
    }
    
    st.markdown("---")
    
    with st.expander("¿Así que estos son los datos, Ja?"):
        st.write(f"**Estado Cuenta:** {status_opts[status]}")
        st.write(f"**Duración:** {duration} meses")
        st.write(f"**Historial:** {history_opts[credit_history]}")
        st.write(f"**Monto:** {amount:,} DM")
        st.write(f"**Edad:** {age} años")
    
    # Predicción
    prediction = None
    use_sim = False
    
    with st.spinner("🔄 Procesando con rigor alemán..."):
        if model:
            try:
                processed = preprocess_input(input_data)
                if processed is not None:
                    prediction = int(model.predict(processed)[0])
                    st.success("✅ ¡Análisis completado con éxito!")
                else:
                    use_sim = True
            except Exception as e:
                st.warning(f"⚠️ ¡Achtung! Error en el modelo: {e}")
                use_sim = True
        else:
            use_sim = True
        
        if use_sim:
            st.info("ℹ️ Usando simulación. No es lo óptimo, pero funcional.")
            prediction, score = simulate_prediction(input_data)
    
    # Resultado (Das Ergebnis)
    st.markdown("---")
    st.subheader("🎯 El Veredicto Final")
    
    if prediction == 1:
        st.success("### ✅ ¡APROBADO! Perfil en ORDEN.")
        st.markdown("""
        **Recomendación:** Proceder.
        
        Este solicitante cumple con los parámetros de eficiencia y bajo riesgo.
        """)
        st.balloons()
    else:
        st.error("### 🛑 ¡RECHAZADO! Alto Riesgo.")
        st.markdown("""
        **Recomendación:** Denegar o exigir garantías adicionales.
        
        Este solicitante presenta un riesgo inaceptable. ¡No cumple la norma!
        """)
    
    if use_sim:
        st.info(f"📊 Score Lógico: {score}/7")

st.markdown("---")
st.caption("Un Sistema con Precisión Alemana | German Credit Dataset")
