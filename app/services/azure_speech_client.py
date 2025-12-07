"""
Cliente para interactuar con Azure Speech Services (STT y TTS).
"""
import io
import azure.cognitiveservices.speech as speechsdk
from typing import Optional
from app.config import AzureSpeechConfig


class AzureSpeechClient:
    """Cliente para convertir voz a texto (STT) y texto a voz (TTS) usando Azure Speech Services."""
    
    def __init__(self, config: AzureSpeechConfig):
        """
        Inicializa el cliente de Azure Speech Services.
        
        Args:
            config: Configuración de Azure Speech Services.
        """
        self.config = config
        
        # Configurar credenciales de Azure Speech
        self.speech_config = speechsdk.SpeechConfig(
            subscription=config.api_key,
            region=config.region
        )
        
        # Configurar idioma (español por defecto)
        self.speech_config.speech_recognition_language = config.language
        self.speech_config.speech_synthesis_language = config.language
        
        # Configurar voz para TTS (español)
        if config.language.startswith("es"):
            self.speech_config.speech_synthesis_voice_name = config.voice_name or "es-ES-ElviraNeural"
        else:
            self.speech_config.speech_synthesis_voice_name = config.voice_name or "en-US-JennyNeural"
    
    def speech_to_text(self, audio_data: bytes, audio_format: str = "wav") -> Optional[str]:
        """
        Convierte audio a texto usando Azure Speech-to-Text.
        
        Args:
            audio_data: Datos de audio en bytes.
            audio_format: Formato del audio ("wav", "mp3", etc.). Por defecto "wav".
        
        Returns:
            Texto transcrito o None si hay error.
        """
        try:
            # Configurar formato de audio
            if audio_format.lower() == "wav":
                audio_stream = speechsdk.audio.PushAudioInputStream()
                audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            else:
                # Para otros formatos, usar el stream directamente
                audio_stream = speechsdk.audio.PushAudioInputStream()
                audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
            
            # Crear reconocedor de voz
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Escribir datos de audio al stream
            audio_stream.write(audio_data)
            audio_stream.close()
            
            # Realizar reconocimiento
            result = speech_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return None
            else:
                error_details = result.cancellation_details
                if error_details:
                    raise Exception(f"Error en reconocimiento de voz: {error_details.reason} - {error_details.error_details}")
                return None
                
        except Exception as e:
            raise Exception(f"Error al convertir voz a texto: {str(e)}")
    
    def speech_to_text_from_file(self, audio_file_path: str) -> Optional[str]:
        """
        Convierte un archivo de audio a texto.
        
        Args:
            audio_file_path: Ruta al archivo de audio.
        
        Returns:
            Texto transcrito o None si hay error.
        """
        try:
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            result = speech_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                return None
            else:
                error_details = result.cancellation_details
                if error_details:
                    raise Exception(f"Error en reconocimiento de voz: {error_details.reason} - {error_details.error_details}")
                return None
                
        except Exception as e:
            raise Exception(f"Error al convertir voz a texto desde archivo: {str(e)}")
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convierte texto a audio usando Azure Text-to-Speech.
        
        Args:
            text: Texto a convertir a voz.
        
        Returns:
            Datos de audio en bytes (formato WAV).
        """
        try:
            # Configurar formato de audio (WAV 16kHz mono)
            self.speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio16Khz128KBitRateMonoMp3
            )
            
            # Crear sintetizador de voz sin configuración de salida (genera en memoria)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            # Sintetizar voz
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.CancellationDetails(result)
                raise Exception(
                    f"Error en síntesis de voz: {cancellation_details.reason} - "
                    f"{cancellation_details.error_details}"
                )
            else:
                raise Exception(f"Error desconocido en síntesis de voz: {result.reason}")
                
        except Exception as e:
            raise Exception(f"Error al convertir texto a voz: {str(e)}")
    
    def text_to_speech_to_file(self, text: str, output_file_path: str) -> bool:
        """
        Convierte texto a audio y lo guarda en un archivo.
        
        Args:
            text: Texto a convertir a voz.
            output_file_path: Ruta donde guardar el archivo de audio.
        
        Returns:
            True si se guardó correctamente, False en caso contrario.
        """
        try:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return True
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.CancellationDetails(result)
                raise Exception(
                    f"Error en síntesis de voz: {cancellation_details.reason} - "
                    f"{cancellation_details.error_details}"
                )
            return False
                
        except Exception as e:
            raise Exception(f"Error al convertir texto a voz y guardar archivo: {str(e)}")

