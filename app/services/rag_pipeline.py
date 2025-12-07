"""
Pipeline RAG (Retrieval Augmented Generation) que orquesta la búsqueda
y generación de respuestas.
"""
from typing import Dict, List, Optional, Generator
from app.services.azure_search_client import AzureSearchClient
from app.services.azure_openai_client import AzureOpenAIClient
from app.config import AzureSearchConfig, AzureOpenAIConfig


class RAGPipeline:
    """Pipeline que implementa el patrón RAG completo."""
    
    def __init__(
        self,
        search_config: AzureSearchConfig,
        openai_config: AzureOpenAIConfig
    ):
        """
        Inicializa el pipeline RAG con los clientes necesarios.
        
        Args:
            search_config: Configuración de Azure AI Search.
            openai_config: Configuración de Azure OpenAI.
        """
        self.search_client = AzureSearchClient(search_config)
        self.openai_client = AzureOpenAIClient(openai_config)
        
        # Prompt del sistema para el modelo
        self.system_prompt = """Eres un asistente especializado para field engineers de dispositivos biomédicos. 
Tu función es ayudar a los técnicos a encontrar información en los manuales técnicos y de usuario.

INSTRUCCIONES:
- Usa la información proporcionada en el contexto de los manuales para responder la pregunta.
- Si encuentras información relevante en el contexto, aunque sea parcial, proporciona una respuesta basada en esa información.
- Si el contexto menciona algo relacionado con la pregunta, aunque no sea una respuesta completa, comparte esa información.
- Proporciona respuestas claras, concisas y técnicas.
- Si mencionas procedimientos, sé específico sobre los pasos.
- Si hay información sobre modelos o números de parte, inclúyela en tu respuesta.
- Solo di "No encontré información suficiente" si realmente no hay NADA relacionado con la pregunta en el contexto."""
    
    def rag_answer(
        self,
        user_question: str,
        top_k: int = 5,
        temperature: float = 1.0
    ) -> Dict:
        """
        Ejecuta el pipeline RAG completo: búsqueda + generación de respuesta.
        
        Args:
            user_question: Pregunta del usuario.
            top_k: Número de documentos a recuperar de Azure Search.
            temperature: Temperatura para la generación del modelo.
        
        Returns:
            Diccionario con:
                - "answer": texto de la respuesta generada.
                - "sources": lista de fuentes usadas (cada fuente es un dict con
                            "source", "pageNumber", "score", etc.).
        """
        try:
            # Paso 1: Buscar documentos relevantes en Azure AI Search
            search_results = self.search_client.search_documents_text_only(
                query=user_question,
                top_k=top_k
            )
            
            # Paso 2: Validar que se encontraron resultados
            if not search_results or len(search_results) == 0:
                return {
                    "answer": "No se encontró información relevante en los manuales para responder tu pregunta. Por favor, intenta reformularla o usar términos más específicos.",
                    "sources": []
                }
            
            # Paso 3: Extraer y limitar fragmentos de texto (campo "content")
            # Aplicar límites para evitar exceder el límite de tokens
            MAX_CHARS_PER_CHUNK = 4000  # Máximo de caracteres por chunk (aumentado)
            MAX_TOTAL_CONTEXT = 12000   # Máximo total de caracteres en el contexto (aumentado significativamente)
            
            context_chunks = []
            total_chars = 0
            
            for doc in search_results:
                content = doc.get("content", "")
                if not content:
                    continue
                
                # Truncar chunks muy largos de forma inteligente
                # Mantener el inicio y el final del chunk si es posible
                if len(content) > MAX_CHARS_PER_CHUNK:
                    # Truncar desde el medio, manteniendo inicio y final
                    half = MAX_CHARS_PER_CHUNK // 2
                    content = content[:half] + "... [texto truncado] ..." + content[-half:]
                
                # Verificar si añadir este chunk excedería el límite total
                chunk_size = len(content)
                if total_chars + chunk_size > MAX_TOTAL_CONTEXT:
                    # Si ya tenemos al menos un chunk, parar aquí
                    if context_chunks:
                        break
                    # Si es el primer chunk y es muy grande, truncarlo más
                    content = content[:MAX_TOTAL_CONTEXT] + "... [texto truncado]"
                    context_chunks.append(content)
                    break
                
                context_chunks.append(content)
                total_chars += chunk_size
            
            if not context_chunks:
                # Debug: mostrar qué se encontró pero no se pudo procesar
                debug_info = f"Se encontraron {len(search_results)} documentos pero no contenían texto útil."
                if search_results:
                    debug_info += f" Scores: {[doc.get('score', 0) for doc in search_results[:3]]}"
                return {
                    "answer": f"{debug_info} Por favor, intenta otra pregunta o reformula con términos más específicos.",
                    "sources": []
                }
            
            # Paso 4: Generar respuesta usando Azure OpenAI con el contexto
            answer = self.openai_client.generate_response(
                system_prompt=self.system_prompt,
                user_message=user_question,
                context_chunks=context_chunks,
                temperature=temperature
            )
            
            # Paso 5: Preparar información de fuentes
            # Los resultados ya vienen con el campo "source" mapeado desde metadata_storage_name
            sources = []
            for doc in search_results:
                source_info = {
                    "source": doc.get("source", "Unknown"),  # Nombre del PDF
                    "score": doc.get("score", 0.0)  # Score de relevancia
                }
                # Opcional: incluir path para depuración si es necesario
                if "path" in doc:
                    source_info["path"] = doc["path"]
                
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            # Manejo de errores con mensajes más específicos
            error_message = str(e)
            
            # Si el error ya tiene un mensaje claro (como rate limit), usarlo directamente
            if "Límite de tasa alcanzado" in error_message or "rate limit" in error_message.lower():
                return {
                    "answer": f"⚠️ **Límite de tasa alcanzado**\n\n{error_message}\n\nPor favor, espera un momento antes de hacer otra pregunta.",
                    "sources": []
                }
            
            # Otros errores
            return {
                "answer": f"❌ **Error al procesar tu pregunta**\n\n{error_message}\n\nPor favor, intenta de nuevo o verifica tu configuración de Azure.",
                "sources": []
            }
    
    def rag_answer_stream(
        self,
        user_question: str,
        top_k: int = 5,
        temperature: float = 1.0
    ) -> Generator[str, None, Dict]:
        """
        Ejecuta el pipeline RAG con streaming: búsqueda + generación de respuesta en tiempo real.
        
        Args:
            user_question: Pregunta del usuario.
            top_k: Número de documentos a recuperar de Azure Search.
            temperature: Temperatura para la generación del modelo.
        
        Yields:
            Fragmentos de texto de la respuesta conforme se generan.
        
        Returns:
            Diccionario con "sources" al finalizar.
        """
        try:
            # Paso 1: Buscar documentos relevantes en Azure AI Search
            search_results = self.search_client.search_documents_text_only(
                query=user_question,
                top_k=top_k
            )
            
            # Paso 2: Validar que se encontraron resultados
            if not search_results or len(search_results) == 0:
                yield "No se encontró información relevante en los manuales para responder tu pregunta. Por favor, intenta reformularla o usar términos más específicos."
                return {"sources": []}
            
            # Paso 3: Extraer y limitar fragmentos de texto
            MAX_CHARS_PER_CHUNK = 4000
            MAX_TOTAL_CONTEXT = 12000
            
            context_chunks = []
            total_chars = 0
            
            for doc in search_results:
                content = doc.get("content", "")
                if not content:
                    continue
                
                if len(content) > MAX_CHARS_PER_CHUNK:
                    half = MAX_CHARS_PER_CHUNK // 2
                    content = content[:half] + "... [texto truncado] ..." + content[-half:]
                
                chunk_size = len(content)
                if total_chars + chunk_size > MAX_TOTAL_CONTEXT:
                    if context_chunks:
                        break
                    content = content[:MAX_TOTAL_CONTEXT] + "... [texto truncado]"
                    context_chunks.append(content)
                    break
                
                context_chunks.append(content)
                total_chars += chunk_size
            
            if not context_chunks:
                debug_info = f"Se encontraron {len(search_results)} documentos pero no contenían texto útil."
                if search_results:
                    debug_info += f" Scores: {[doc.get('score', 0) for doc in search_results[:3]]}"
                yield f"{debug_info} Por favor, intenta otra pregunta o reformula con términos más específicos."
                return {"sources": []}
            
            # Paso 4: Generar respuesta con streaming
            full_answer = ""
            for chunk in self.openai_client.generate_response_stream(
                system_prompt=self.system_prompt,
                user_message=user_question,
                context_chunks=context_chunks,
                temperature=temperature
            ):
                full_answer += chunk
                yield chunk
            
            # Paso 5: Preparar información de fuentes
            sources = []
            for doc in search_results:
                source_info = {
                    "source": doc.get("source", "Unknown"),
                    "score": doc.get("score", 0.0)
                }
                if "path" in doc:
                    source_info["path"] = doc["path"]
                sources.append(source_info)
            
            return {"sources": sources}
            
        except Exception as e:
            error_message = str(e)
            if "Límite de tasa alcanzado" in error_message or "rate limit" in error_message.lower():
                yield f"⚠️ **Límite de tasa alcanzado**\n\n{error_message}\n\nPor favor, espera un momento antes de hacer otra pregunta."
            else:
                yield f"❌ **Error al procesar tu pregunta**\n\n{error_message}\n\nPor favor, intenta de nuevo o verifica tu configuración de Azure."
            return {"sources": []}

