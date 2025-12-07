"""
Cliente para interactuar con Azure OpenAI.
"""
from typing import List, Generator
from openai import AzureOpenAI

from app.config import AzureOpenAIConfig


class AzureOpenAIClient:
    """Cliente para generar respuestas usando Azure OpenAI."""
    
    def __init__(self, config: AzureOpenAIConfig):
        """
        Inicializa el cliente de Azure OpenAI.
        
        Args:
            config: Configuración de Azure OpenAI.
        """
        self.config = config
        self.client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version="2024-02-15-preview"  # Versión que soporta chat completions
        )
    
    def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: List[str],
        temperature: float = 0.2
    ) -> str:
        """
        Genera una respuesta usando Azure OpenAI con el contexto proporcionado.
        
        Args:
            system_prompt: Instrucciones del sistema para el modelo.
            user_message: Pregunta o mensaje del usuario.
            context_chunks: Lista de fragmentos de texto del contexto recuperado.
            temperature: Temperatura para la generación (0.0-1.0). Valores más bajos
                        dan respuestas más deterministas.
        
        Returns:
            Texto de la respuesta generada por el modelo.
        """
        try:
            # Construir el contexto concatenando los chunks
            context_text = "\n\n".join([
                f"[Fragmento {i+1}]\n{chunk}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Construir el prompt del usuario con el contexto
            user_prompt_with_context = f"""Contexto de los manuales técnicos:

{context_text}

---

Pregunta del usuario: {user_message}

Basándote en el contexto proporcionado, responde la pregunta del usuario. Si encuentras información relevante, aunque sea parcial, compártela. Si el contexto menciona algo relacionado con la pregunta, inclúyelo en tu respuesta."""
            
            # Preparar mensajes en formato chat
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_with_context}
            ]
            
            # Preparar parámetros de la llamada
            # Algunos modelos solo soportan temperature=1 (valor por defecto)
            call_params = {
                "model": self.config.deployment_name,
                "messages": messages,
                "max_completion_tokens": 800  # Aumentado para respuestas más completas
            }
            
            # Solo añadir temperature si es diferente de 1.0
            # Si el modelo no lo soporta, se intentará sin este parámetro
            if temperature != 1.0:
                call_params["temperature"] = temperature
            
            # Llamar al modelo
            try:
                response = self.client.chat.completions.create(**call_params)
            except Exception as temp_error:
                # Si falla por temperature no soportado, reintentar sin ese parámetro
                error_str = str(temp_error)
                if "temperature" in error_str.lower() and "unsupported" in error_str.lower():
                    # Reintentar sin temperature (usará el valor por defecto del modelo)
                    call_params.pop("temperature", None)
                    response = self.client.chat.completions.create(**call_params)
                else:
                    # Si es otro error, relanzarlo
                    raise
            
            # Extraer y retornar el texto de la respuesta
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "No se pudo generar una respuesta."
                
        except Exception as e:
            # Detectar errores específicos de Azure OpenAI
            error_str = str(e)
            
            # Error 429: Rate Limit (límite de tasa alcanzado)
            if "429" in error_str or "RateLimitReached" in error_str or "rate limit" in error_str.lower():
                raise Exception(
                    "Límite de tasa alcanzado: Has excedido el límite de tokens por minuto de tu plan de Azure OpenAI. "
                    "Por favor, espera 60 segundos antes de intentar de nuevo. "
                    "Para aumentar el límite, visita: https://aka.ms/oai/quotaincrease"
                )
            
            # Error 400: Bad Request
            elif "400" in error_str:
                raise Exception(f"Error de solicitud: {error_str}")
            
            # Otros errores
            else:
                raise Exception(f"Error al generar respuesta con Azure OpenAI: {error_str}")
    
    def generate_response_stream(
        self,
        system_prompt: str,
        user_message: str,
        context_chunks: List[str],
        temperature: float = 1.0
    ) -> Generator[str, None, None]:
        """
        Genera una respuesta usando Azure OpenAI con streaming (carácter por carácter).
        
        Args:
            system_prompt: Instrucciones del sistema para el modelo.
            user_message: Pregunta o mensaje del usuario.
            context_chunks: Lista de fragmentos de texto del contexto recuperado.
            temperature: Temperatura para la generación (0.0-1.0).
        
        Yields:
            Fragmentos de texto de la respuesta conforme se generan.
        """
        try:
            # Construir el contexto concatenando los chunks
            context_text = "\n\n".join([
                f"[Fragmento {i+1}]\n{chunk}"
                for i, chunk in enumerate(context_chunks)
            ])
            
            # Construir el prompt del usuario con el contexto
            user_prompt_with_context = f"""Contexto de los manuales técnicos:

{context_text}

---

Pregunta del usuario: {user_message}

Basándote en el contexto proporcionado, responde la pregunta del usuario. Si encuentras información relevante, aunque sea parcial, compártela. Si el contexto menciona algo relacionado con la pregunta, inclúyelo en tu respuesta."""
            
            # Preparar mensajes en formato chat
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_with_context}
            ]
            
            # Preparar parámetros de la llamada con streaming
            call_params = {
                "model": self.config.deployment_name,
                "messages": messages,
                "max_completion_tokens": 800,
                "stream": True  # Activar streaming
            }
            
            # Solo añadir temperature si es diferente de 1.0
            if temperature != 1.0:
                call_params["temperature"] = temperature
            
            # Llamar al modelo con streaming
            try:
                stream = self.client.chat.completions.create(**call_params)
            except Exception as temp_error:
                # Si falla por temperature no soportado, reintentar sin ese parámetro
                error_str = str(temp_error)
                if "temperature" in error_str.lower() and "unsupported" in error_str.lower():
                    call_params.pop("temperature", None)
                    stream = self.client.chat.completions.create(**call_params)
                else:
                    raise
            
            # Yielding fragmentos conforme se reciben
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                        
        except Exception as e:
            # Detectar errores específicos de Azure OpenAI
            error_str = str(e)
            
            # Error 429: Rate Limit
            if "429" in error_str or "RateLimitReached" in error_str or "rate limit" in error_str.lower():
                yield f"\n\n⚠️ **Límite de tasa alcanzado**: Has excedido el límite de tokens por minuto. Espera 60 segundos antes de intentar de nuevo."
            
            # Error 400: Bad Request
            elif "400" in error_str:
                yield f"\n\n❌ **Error de solicitud**: {error_str}"
            
            # Otros errores
            else:
                yield f"\n\n❌ **Error**: {error_str}"

