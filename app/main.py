"""
Aplicación principal de Streamlit para el chat RAG con manuales biomédicos.
"""
import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import tempfile
from app.config import load_config
from app.services.rag_pipeline import RAGPipeline
from app.services.azure_speech_client import AzureSpeechClient

# Configurar página
st.set_page_config(
    page_title="Chat con Manuales Biomédicos",
    page_icon="🏥",
    layout="wide"
)

# Inicializar configuración
try:
    config = load_config()
except ValueError as e:
    st.error(f"Error de configuración: {str(e)}")
    st.stop()

# Inicializar pipeline RAG (una sola vez, usando cache)
# Nota: Si cambias el código de RAGPipeline, presiona Ctrl+C y reinicia Streamlit
# o usa el botón "Clear cache" en el menú de Streamlit
@st.cache_resource
def get_rag_pipeline():
    """Inicializa y cachea el pipeline RAG."""
    return RAGPipeline(
        search_config=config.azure_search,
        openai_config=config.azure_openai
    )

rag_pipeline = get_rag_pipeline()

# Verificar que el método de streaming existe
if not hasattr(rag_pipeline, 'rag_answer_stream'):
    st.error("⚠️ El método rag_answer_stream no está disponible. Por favor, reinicia Streamlit (Ctrl+C y vuelve a ejecutar).")
    st.stop()

# Inicializar cliente de Azure Speech si está configurado
speech_client = None
if config.azure_speech:
    try:
        speech_client = AzureSpeechClient(config.azure_speech)
    except Exception as e:
        st.warning(f"⚠️ No se pudo inicializar Azure Speech Services: {str(e)}. La funcionalidad de voz estará deshabilitada.")

# Inicializar historial de chat en session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Título y descripción
st.title("🏥 Chat con Manuales Biomédicos")
st.markdown("""
**Aplicación RAG (Retrieval Augmented Generation)** para consultar manuales técnicos y de usuario 
de dispositivos biomédicos usando Azure AI Search y Azure OpenAI.

Esta herramienta está diseñada para ayudar a **field engineers** a encontrar información técnica 
durante el mantenimiento de equipos.

**🎤 Funcionalidad de voz disponible**: Puedes hacer preguntas con voz y recibir respuestas habladas.
""")

# Sidebar con parámetros y configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Parámetros ajustables
    top_k = st.slider(
        "Número de documentos a recuperar (top_k)",
        min_value=1,
        max_value=15,
        value=5,
        help="Cantidad de fragmentos de manuales que se usarán como contexto. Valores más altos proporcionan más contexto pero usan más tokens."
    )
    
    temperature = st.slider(
        "Temperatura del modelo",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Valores más bajos dan respuestas más deterministas. Nota: Algunos modelos solo soportan el valor por defecto (1.0)"
    )
    
    st.divider()
    
    st.info("💡 **Optimización de tokens**: El sistema limita automáticamente el tamaño del contexto para evitar límites de tasa. Los chunks muy largos se truncarán si es necesario.")
    
    st.divider()
    
    st.subheader("📖 Instrucciones de uso")
    st.markdown("""
    **Ejemplos de preguntas:**
    - "¿Cómo calibro el sensor de oxígeno del modelo X?"
    - "¿Cuál es el procedimiento de mantenimiento preventivo?"
    - "¿Qué código de error significa E-123?"
    - "¿Cómo cambio el filtro del dispositivo Y?"
    
    **Consejos:**
    - Sé específico con modelos y números de parte
    - Usa términos técnicos cuando los conozcas
    - Si no encuentras respuesta, reformula la pregunta
    """)
    
    # Botón para limpiar conversación
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Cuerpo principal: historial de chat
st.subheader("💬 Conversación")

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Mostrar fuentes si existen (solo para mensajes del asistente)
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources:
                st.markdown("---")
                st.markdown("**📚 Fuentes utilizadas:**")
                for i, source in enumerate(sources, 1):
                    source_name = source.get("source", "Unknown")
                    score = source.get("score", 0.0)
                    
                    source_text = f"{i}. {source_name}"
                    if score > 0:
                        source_text += f" - Relevancia: {score:.2f}"
                    
                    st.caption(source_text)

# Función para procesar preguntas (común para texto y voz)
def process_question(prompt: str, is_from_voice: bool = False):
    """Procesa una pregunta y genera la respuesta usando RAG."""
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        if is_from_voice:
            st.markdown(f"🎤 {prompt}")
        else:
            st.markdown(prompt)
    
    # Generar respuesta usando RAG con streaming
    with st.chat_message("assistant"):
        try:
            # Mostrar spinner mientras busca
            with st.spinner("Buscando en los manuales..."):
                # Crear generador para streaming
                stream_generator = rag_pipeline.rag_answer_stream(
                    user_question=prompt,
                    top_k=top_k,
                    temperature=temperature
                )
            
            # Mostrar respuesta en tiempo real usando streaming
            answer_placeholder = st.empty()
            full_answer = ""
            
            for chunk in stream_generator:
                full_answer += chunk
                answer_placeholder.markdown(full_answer)
            
            # Obtener las fuentes después del streaming
            try:
                search_results = rag_pipeline.search_client.search_documents_text_only(
                    query=prompt,
                    top_k=top_k
                )
                sources = []
                for doc in search_results:
                    source_info = {
                        "source": doc.get("source", "Unknown"),
                        "score": doc.get("score", 0.0)
                    }
                    if "path" in doc:
                        source_info["path"] = doc["path"]
                    sources.append(source_info)
            except:
                sources = []
            
            # Mostrar fuentes
            if sources:
                st.markdown("---")
                st.markdown("**📚 Fuentes utilizadas:**")
                for i, source in enumerate(sources, 1):
                    source_name = source.get("source", "Unknown")
                    score = source.get("score", 0.0)
                    
                    source_text = f"{i}. {source_name}"
                    if score > 0:
                        source_text += f" - Relevancia: {score:.2f}"
                    
                    st.caption(source_text)
            
            # Convertir respuesta a voz si está disponible
            if speech_client and full_answer:
                with st.spinner("🔊 Generando audio de la respuesta..."):
                    try:
                        audio_data = speech_client.text_to_speech(full_answer)
                        st.audio(audio_data, format="audio/wav", autoplay=True)
                        st.success("✅ Audio generado. Reproduciendo respuesta...")
                    except Exception as tts_error:
                        st.warning(f"⚠️ No se pudo generar el audio: {str(tts_error)}")
            
            # Guardar respuesta en el historial
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            })
                
        except Exception as e:
            error_msg = f"❌ Error al procesar la pregunta: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

# Sección de entrada: texto y voz integrados en un solo widget
# Usar chat_input con accept_audio si speech_client está disponible
if speech_client:
    # Chat input con soporte de audio integrado
    user_input = st.chat_input(
        "Escribe tu pregunta o graba un audio...",
        accept_audio=True,
        audio_sample_rate=16000
    )
    
    if user_input:
        # user_input puede ser un string o un objeto dict-like con text y audio
        if isinstance(user_input, str):
            # Solo texto
            process_question(user_input, is_from_voice=False)
        else:
            # Objeto dict-like (puede tener text, audio, o ambos)
            text_prompt = user_input.text if hasattr(user_input, 'text') else user_input.get('text', '')
            audio_file = user_input.audio if hasattr(user_input, 'audio') else user_input.get('audio', None)
            
            # Si hay audio, procesarlo primero
            if audio_file:
                try:
                    with st.spinner("🎤 Procesando audio..."):
                        # Leer los bytes del archivo de audio
                        audio_bytes = audio_file.read()
                        
                        # Guardar audio temporalmente
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_audio_path = tmp_file.name
                        
                        try:
                            # Convertir voz a texto
                            transcribed_text = speech_client.speech_to_text_from_file(tmp_audio_path)
                            
                            if transcribed_text and transcribed_text.strip():
                                # Usar el texto transcrito (o combinarlo con texto escrito si existe)
                                final_prompt = transcribed_text.strip()
                                if text_prompt:
                                    final_prompt = f"{text_prompt} {final_prompt}".strip()
                                
                                # Procesar la pregunta usando la función común
                                process_question(final_prompt, is_from_voice=True)
                            else:
                                st.warning("⚠️ No se pudo transcribir el audio. Por favor, intenta de nuevo hablando más claro.")
                            
                            # Limpiar archivo temporal
                            if os.path.exists(tmp_audio_path):
                                os.unlink(tmp_audio_path)
                                
                        except Exception as stt_error:
                            st.error(f"❌ Error al procesar el audio: {str(stt_error)}")
                            if os.path.exists(tmp_audio_path):
                                os.unlink(tmp_audio_path)
                            
                except Exception as e:
                    st.error(f"❌ Error al procesar la grabación: {str(e)}")
            elif text_prompt:
                # Solo texto (sin audio)
                process_question(text_prompt, is_from_voice=False)
else:
    # Chat input sin soporte de audio
    prompt = st.chat_input("Escribe tu pregunta sobre los manuales biomédicos...")
    if prompt:
        process_question(prompt, is_from_voice=False)

# Footer
st.divider()
footer_text = "💡 Esta aplicación usa Azure AI Search para búsqueda semántica y Azure OpenAI para generación de respuestas."
if speech_client:
    footer_text += " También incluye Azure Speech Services para entrada y salida de voz."
st.caption(footer_text)

