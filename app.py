import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from PIL import Image
import os
import base64
from fpdf import FPDF
from datetime import datetime
import tempfile

# Configuración de la página
st.set_page_config(page_title="Clasificador de Camisetas", page_icon="👕", layout="wide")

# Inicialización de session_state
if 'prediccion' not in st.session_state:
    st.session_state.prediccion = None
if 'imagen' not in st.session_state:
    st.session_state.imagen = None
if 'archivo_subido' not in st.session_state:
    st.session_state.archivo_subido = None

# ======================
# CONFIGURACIÓN INICIAL
# ======================
MODEL_PATH = 'model/alexnet_final.keras'  # Cambiar la ruta al modelo AlexNet
ATTRIBUTES = ['gender', 'usage']
IMG_SIZE = (227, 227)  # Cambiar de (224, 224) a (227, 227) para AlexNet
# ======================
# FUNCIONES AUXILIARES
# ======================
@st.cache_resource
def cargar_modelo():
    try:
        modelo = tf.keras.models.load_model(MODEL_PATH)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

@st.cache_resource
def obtener_codificadores_etiquetas():
    return {
        'gender': LabelEncoder().fit(['Men', 'Women']),
        'usage': LabelEncoder().fit(['Casual', 'Sports'])
    }

def predecir_atributos_camiseta(archivo_subido, modelo, codificadores):
    try:
        img = Image.open(archivo_subido)
        img = img.resize(IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predicciones = modelo.predict(img_array, verbose=0)

        resultados = {}
        for i, attr in enumerate(ATTRIBUTES):
            clase_predicha = np.argmax(predicciones[i])
            etiqueta_predicha = codificadores[attr].inverse_transform([clase_predicha])[0]
            confianza = np.max(predicciones[i])
            resultados[attr] = {
                'label': etiqueta_predicha, 
                'confidence': float(confianza),
                'probabilities': {cls: float(prob) for cls, prob in 
                                zip(codificadores[attr].classes_, predicciones[i][0])}
            }
        
        return resultados, img

    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {str(e)}")
        return None, None

def generar_enlace_descarga_pdf(ruta_archivo, texto_boton):
    with open(ruta_archivo, "rb") as f:
        datos = f.read()
    bin_str = base64.b64encode(datos).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="reporte_camiseta.pdf">{texto_boton}</a>'
    return href

def generar_reporte_prediccion(prediccion, img, nombre_modelo="ResNet-50"):
    """Genera un PDF con el reporte de predicción y evaluación de modelos"""
    # Crear PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Encabezado
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Reporte de Análisis de Camiseta", 0, 1, 'C')
    pdf.ln(5)
    
    # Información general
    pdf.set_font("Helvetica", '', 12)
    pdf.cell(0, 10, f"Fecha del análisis: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 0, 1)
    pdf.cell(0, 10, f"Modelo utilizado: {nombre_modelo}", 0, 1)
    pdf.ln(10)
    
    # Guardar imagen temporalmente
    temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(temp_img.name, format='JPEG', quality=90)
    
    # Agregar imagen al PDF
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Imagen analizada:", 0, 1)
    pdf.image(temp_img.name, x=10, w=180)
    pdf.ln(15)
    
    # Traducciones
    traduccion_atributos = {'gender': 'Género', 'usage': 'Uso'}
    traduccion_valores = {'Men': 'Hombre', 'Women': 'Mujer', 'Casual': 'Casual', 'Sports': 'Deportivo'}
    
    # Resultados principales
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Resultados principales:", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(70, 10, "Atributo", 1)
    pdf.cell(70, 10, "Predicción", 1)
    pdf.cell(50, 10, "Confianza", 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 12)
    for attr, data in prediccion.items():
        atributo = traduccion_atributos.get(attr, attr)
        valor = traduccion_valores.get(data['label'], data['label'])
        
        pdf.cell(70, 10, atributo, 1)
        pdf.cell(70, 10, valor, 1)
        pdf.cell(50, 10, f"{data['confidence']:.1%}", 1)
        pdf.ln()
    
    # Detalles por atributo
    pdf.ln(15)
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "Detalles por atributo:", 0, 1)
    pdf.ln(5)
    
    for attr, data in prediccion.items():
        atributo = traduccion_atributos.get(attr, attr)
        
        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, f"Atributo: {atributo}", 0, 1)
        pdf.ln(3)
        
        pdf.set_font("Helvetica", '', 12)
        pdf.cell(0, 10, f"Predicción: {traduccion_valores.get(data['label'], data['label'])}", 0, 1)
        pdf.cell(0, 10, f"Confianza: {data['confidence']:.1%}", 0, 1)
        pdf.ln(3)
        
        # Tabla de probabilidades
        pdf.set_font("Helvetica", 'B', 11)
        pdf.cell(90, 8, "Categoría", 1)
        pdf.cell(90, 8, "Probabilidad", 1)
        pdf.ln()
        
        pdf.set_font("Helvetica", '', 11)
        for cls, prob in data['probabilities'].items():
            clase_traducida = traduccion_valores.get(cls, cls)
            pdf.cell(90, 8, clase_traducida, 1)
            pdf.cell(90, 8, f"{prob:.1%}", 1)
            pdf.ln()
        
        pdf.ln(10)
    
    # ===================================
    # SECCIÓN DE EVALUACIÓN DE MODELOS
    # ===================================
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "Evaluación de Modelos", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", '', 12)
    pdf.multi_cell(0, 10, "Esta sección muestra las matrices de confusión y métricas de los diferentes modelos evaluados para la clasificación de camisetas.")
    pdf.ln(15)

    # --------------------------
    # MODELO ALEXNET
    # --------------------------
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "AlexNet", 0, 1)
    pdf.ln(5)
    
    # Matriz de confusión
    pdf.image("matriz/alexnet.png", x=10, w=180)
    pdf.ln(10)
    
    # Métricas para Género
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Métricas para Género:", 0, 1)
    pdf.ln(3)
    
    # Tabla de métricas de género
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(45, 8, "Categoría", 1)
    pdf.cell(35, 8, "Precisión", 1)
    pdf.cell(45, 8, "Sensibilidad (Recall)", 1)
    pdf.cell(35, 8, "F1-Score", 1)
    pdf.cell(30, 8, "Exactitud", 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 11)
    # Hombres
    pdf.cell(45, 8, "Hombres", 1)
    pdf.cell(35, 8, "0.9898", 1)
    pdf.cell(45, 8, "0.9966", 1)
    pdf.cell(35, 8, "0.9932", 1)
    pdf.cell(30, 8, "0.9887", 1)
    pdf.ln()
    # Mujeres
    pdf.cell(45, 8, "Mujeres", 1)
    pdf.cell(35, 8, "0.9833", 1)
    pdf.cell(45, 8, "0.9516", 1)
    pdf.cell(35, 8, "0.9672", 1)
    pdf.cell(30, 8, "0.9887", 1)
    pdf.ln()
    # Promedio
    pdf.cell(45, 8, "Promedio", 1)
    pdf.cell(35, 8, "0.9865", 1)
    pdf.cell(45, 8, "0.9741", 1)
    pdf.cell(35, 8, "0.9802", 1)
    pdf.cell(30, 8, "0.9887", 1)
    pdf.ln()
    
    pdf.ln(10)
    
    # Métricas para Uso
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Métricas para Uso:", 0, 1)
    pdf.ln(3)
    
    # Tabla de métricas de uso
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(45, 8, "Categoría", 1)
    pdf.cell(35, 8, "Precisión", 1)
    pdf.cell(45, 8, "Sensibilidad (Recall)", 1)
    pdf.cell(35, 8, "F1-Score", 1)
    pdf.cell(30, 8, "Exactitud", 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 11)
    # Casual
    pdf.cell(45, 8, "Casual", 1)
    pdf.cell(35, 8, "0.9394", 1)
    pdf.cell(45, 8, "0.9714", 1)
    pdf.cell(35, 8, "0.9551", 1)
    pdf.cell(30, 8, "0.9221", 1)
    pdf.ln()
    # Deportivo
    pdf.cell(45, 8, "Deportivo", 1)
    pdf.cell(35, 8, "0.7922", 1)
    pdf.cell(45, 8, "0.6348", 1)
    pdf.cell(35, 8, "0.7048", 1)
    pdf.cell(30, 8, "0.9221", 1)
    pdf.ln()
    # Promedio
    pdf.cell(45, 8, "Promedio", 1)
    pdf.cell(35, 8, "0.8658", 1)
    pdf.cell(45, 8, "0.8031", 1)
    pdf.cell(35, 8, "0.8300", 1)
    pdf.cell(30, 8, "0.9221", 1)
    pdf.ln()
    
    pdf.ln(15)

    # --------------------------
    # MODELO RESNET50
    # --------------------------
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "ResNet-50", 0, 1)
    pdf.ln(5)
    
    # Matriz de confusión
    pdf.image("matriz/resnet50.png", x=10, w=180)
    pdf.ln(10)
    
    # Métricas para Género
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Métricas para Género:", 0, 1)
    pdf.ln(3)
    
    # Tabla de métricas de género
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(45, 8, "Categoría", 1)
    pdf.cell(35, 8, "Precisión", 1)
    pdf.cell(45, 8, "Sensibilidad (Recall)", 1)
    pdf.cell(35, 8, "F1-Score", 1)
    pdf.cell(30, 8, "Exactitud", 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 11)
    # Hombres
    pdf.cell(45, 8, "Hombre", 1)
    pdf.cell(35, 8, "0.95", 1)
    pdf.cell(45, 8, "0.99", 1)
    pdf.cell(35, 8, "0.97", 1)
    pdf.cell(30, 8, "0.95", 1)
    pdf.ln()
    # Mujeres
    pdf.cell(45, 8, "Mujer", 1)
    pdf.cell(35, 8, "0.94", 1)
    pdf.cell(45, 8, "0.78", 1)
    pdf.cell(35, 8, "0.85", 1)
    pdf.cell(30, 8, "0.95", 1)
    pdf.ln()
    
    pdf.ln(10)
    
    # Métricas para Uso
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Métricas para Uso:", 0, 1)
    pdf.ln(3)
    
    # Tabla de métricas de uso
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(45, 8, "Categoría", 1)
    pdf.cell(35, 8, "Precisión", 1)
    pdf.cell(45, 8, "Sensibilidad (Recall)", 1)
    pdf.cell(35, 8, "F1-Score", 1)
    pdf.cell(30, 8, "Exactitud", 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 11)
    # Casual
    pdf.cell(45, 8, "Casual", 1)
    pdf.cell(35, 8, "0.91", 1)
    pdf.cell(45, 8, "0.96", 1)
    pdf.cell(35, 8, "0.94", 1)
    pdf.cell(30, 8, "0.88", 1)
    pdf.ln()
    # Deportivo
    pdf.cell(45, 8, "Deportivo", 1)
    pdf.cell(35, 8, "0.67", 1)
    pdf.cell(45, 8, "0.44", 1)
    pdf.cell(35, 8, "0.53", 1)
    pdf.cell(30, 8, "0.88", 1)
    pdf.ln()

    
    pdf.ln(15)

    # --------------------------
    # MODELO EFFICIENTNET
    # --------------------------
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, "EfficientNet", 0, 1)
    pdf.ln(5)
    
    # Matriz de confusión
    pdf.image("matriz/efficientnet.png", x=10, w=180)
    pdf.ln(10)
    
    # Métricas para Género
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Métricas para Género:", 0, 1)
    pdf.ln(3)
    
    # Tabla de métricas de género
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(45, 8, "Categoría", 1)
    pdf.cell(35, 8, "Precisión", 1)
    pdf.cell(45, 8, "Sensibilidad (Recall)", 1)
    pdf.cell(35, 8, "F1-Score", 1)
    pdf.cell(30, 8, "Exactitud", 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 11)
    # Hombres
    pdf.cell(45, 8, "Hombre", 1)
    pdf.cell(35, 8, "0.8317", 1)
    pdf.cell(45, 8, "0.9389", 1)
    pdf.cell(35, 8, "0.8821", 1)
    pdf.cell(30, 8, "0.7930", 1)
    pdf.ln()
    # Mujeres
    pdf.cell(45, 8, "Mujer", 1)
    pdf.cell(35, 8, "0.2711", 1)
    pdf.cell(45, 8, "0.1067", 1)
    pdf.cell(35, 8, "0.1532", 1)
    pdf.cell(30, 8, "0.7930", 1)
    pdf.ln()

    
    pdf.ln(10)
    
    # Métricas para Uso
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 10, "Métricas para Uso:", 0, 1)
    pdf.ln(3)
    
    # Tabla de métricas de uso
    pdf.set_font("Helvetica", 'B', 11)
    pdf.cell(45, 8, "Categoría", 1)
    pdf.cell(35, 8, "Precisión", 1)
    pdf.cell(45, 8, "Sensibilidad (Recall)", 1)
    pdf.cell(35, 8, "F1-Score", 1)
    pdf.cell(30, 8, "Exactitud", 1)
    pdf.ln()
    
    pdf.set_font("Helvetica", '', 11)
    # Casual
    pdf.cell(45, 8, "Casual", 1)
    pdf.cell(35, 8, "0.4733", 1)
    pdf.cell(45, 8, "0.0343", 1)
    pdf.cell(35, 8, "0.0639", 1)
    pdf.cell(30, 8, "0.1432", 1)
    pdf.ln()
    # Deportivo
    pdf.cell(45, 8, "Deportivo", 1)
    pdf.cell(35, 8, "0.1214", 1)
    pdf.cell(45, 8, "0.7777", 1)
    pdf.cell(35, 8, "0.2100", 1)
    pdf.cell(30, 8, "0.1432", 1)
    pdf.ln()
   
    
    # Pie de página
    pdf.ln(10)
    pdf.set_font("Helvetica", 'I', 10)
    pdf.cell(0, 10, "Reporte generado automáticamente por el Clasificador de Camisetas", 0, 0, 'C')
    
    # Guardar PDF temporal
    temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    pdf.output(temp_pdf.name)
    
    # Limpiar archivos temporales
    try:
        os.unlink(temp_img.name)
    except:
        pass
    
    return temp_pdf.name
# ======================
# INTERFAZ PRINCIPAL
# ======================
# Cargar recursos
modelo = cargar_modelo()
codificadores = obtener_codificadores_etiquetas()

# Título de la aplicación
st.title("👕 Clasificador de Atributos de Camisetas")
st.markdown("""
Suba una imagen de una camiseta para analizar sus atributos:
- **Género**: Hombre / Mujer
- **Uso**: Casual / Deportivo
""")

# Widget para subir archivo
archivo_subido = st.file_uploader("Seleccione una imagen de camiseta", 
                                type=['jpg', 'jpeg', 'png'],
                                key="subidor_archivos")

# Actualizar session_state
if archivo_subido is not None:
    st.session_state.archivo_subido = archivo_subido

# Mostrar imagen cargada si existe
if st.session_state.archivo_subido is not None:
    st.image(st.session_state.archivo_subido, caption="Imagen cargada", width=300)
    
    # Botón para realizar predicción
    if st.button("Analizar imagen"):
        with st.spinner("Analizando imagen..."):
            st.session_state.prediccion, st.session_state.imagen = predecir_atributos_camiseta(
                st.session_state.archivo_subido, modelo, codificadores)
            
        if st.session_state.prediccion:
            st.success("¡Análisis completado con éxito!")

# Mostrar resultados si existen
if st.session_state.prediccion and st.session_state.imagen:
    # Mostrar resultados en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Analizada")
        st.image(st.session_state.imagen, use_container_width=True)
    
    with col2:
        st.subheader("Resultados del Análisis")
        
        # Traducciones para la interfaz
        nombres_atributos = {'gender': 'Género', 'usage': 'Uso'}
        traduccion_valores = {'Men': 'Hombre', 'Women': 'Mujer', 'Casual': 'Casual', 'Sports': 'Deportivo'}
        
        for attr, data in st.session_state.prediccion.items():
            nombre_atributo = nombres_atributos[attr]
            valor_traducido = traduccion_valores[data['label']]
            
            # Barra de progreso para la confianza
            st.progress(data['confidence'], text=f"**{nombre_atributo}**: {valor_traducido} ({data['confidence']:.1%})")
            
            # Mostrar probabilidades en un expander
            with st.expander(f"Detalles de {nombre_atributo.lower()}"):
                for cls, prob in data['probabilities'].items():
                    clase_traducida = traduccion_valores[cls]
                    st.metric(label=clase_traducida, value=f"{prob:.1%}")
    
    # Sección para generar reporte PDF
    st.markdown("---")
    st.subheader("Generar Reporte")
    
    if st.button("Generar Reporte en PDF"):
        with st.spinner("Generando documento PDF..."):
            try:
                pdf_path = generar_reporte_prediccion(
                    st.session_state.prediccion, 
                    st.session_state.imagen,
                    nombre_modelo="AlexNet"
                )
                
                st.success("✅ Reporte generado con éxito!")
                st.markdown(generar_enlace_descarga_pdf(pdf_path, "⬇️ Descargar Reporte Completo"), 
                          unsafe_allow_html=True)
                
                # Limpiar archivo temporal
                try:
                    os.unlink(pdf_path)
                except:
                    pass
            except Exception as e:
                st.error(f"❌ Error al generar el reporte: {str(e)}")

# Información adicional
st.sidebar.markdown("## Acerca de esta aplicación")
st.sidebar.info("""
Esta herramienta utiliza inteligencia artificial para analizar atributos de camisetas.

**Atributos que puede identificar:**
- **Género**: Hombre / Mujer
- **Uso**: Casual / Deportivo

Suba una imagen clara de una camiseta para obtener el análisis.
""")