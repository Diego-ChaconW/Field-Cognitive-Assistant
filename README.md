# 🏥 Azure RAG Chat - Chat con Manuales Biomédicos

Aplicación de chat basada en el patrón **RAG (Retrieval Augmented Generation)** que permite a field engineers consultar información técnica de manuales de dispositivos biomédicos usando Azure AI Search y Azure OpenAI.

## 📋 Descripción del Proyecto

Esta aplicación está diseñada para que los **field engineers** puedan hacer preguntas sobre manuales técnicos y de usuario de dispositivos biomédicos durante el mantenimiento de equipos. La aplicación:

1. **Recibe preguntas** del usuario a través de una interfaz de chat en Streamlit (texto o voz).
2. **Busca información relevante** en un índice de Azure AI Search que contiene chunks de manuales biomédicos.
3. **Genera respuestas contextualizadas** usando Azure OpenAI con el contexto recuperado.
4. **Reproduce respuestas en voz** (opcional) usando Azure Speech Services para una experiencia hands-free.

## 🏗️ Arquitectura

La aplicación utiliza una arquitectura RAG con los siguientes componentes:

- **Frontend**: Streamlit (interfaz de chat interactiva con soporte de voz)
- **Motor de búsqueda**: Azure AI Search (índice con chunks de manuales biomédicos)
- **Modelo de lenguaje**: Azure OpenAI (generación de respuestas contextualizadas)
- **Servicios de voz**: Azure Speech Services (opcional, para entrada y salida de voz)
- **Patrón**: RAG (Retrieval Augmented Generation)

### Flujo de datos:

```
Usuario → Streamlit UI (texto o voz) → Azure Speech STT (si es voz)
                                              ↓
                                    Texto de la pregunta
                                              ↓
                                    RAG Pipeline → Azure AI Search (búsqueda)
                                              ↓
                                    Contexto recuperado
                                              ↓
                                    Azure OpenAI (generación)
                                              ↓
                                    Respuesta + Fuentes
                                              ↓
                                    Azure Speech TTS (opcional) → Audio
                                              ↓
                                    Usuario (texto + audio)
```

## 🔧 Requisitos Previos

Antes de ejecutar la aplicación, necesitas:

1. **Cuenta de Azure** con acceso a:
   - Azure AI Search (servicio creado)
   - Azure OpenAI (recurso con deployment de modelo de chat, por ejemplo GPT-4 o GPT-3.5-turbo)
   - Azure Speech Services (opcional, solo si quieres usar funcionalidad de voz)

2. **Índice de Azure AI Search**:
   - Nombre del índice: **biomed-manuals-demo-index**
   - El índice debe estar creado y poblado con chunks de manuales biomédicos (PDFs procesados)
   - Los manuales deben estar subidos a Azure Blob Storage y procesados mediante un indexer o el wizard de "Import Data" en Azure Portal
   - Campos del índice que usa la aplicación:
     - `content` (String, searchable): Texto de los manuales
     - `metadata_storage_name` (String, filterable, sortable, facetable): Nombre del archivo PDF (mostrado como "source" en la UI)
     - `metadata_storage_path` (String, key): Clave interna del documento

3. **Python 3.x** instalado (recomendado 3.8+)

4. **Variables de entorno** configuradas (ver sección de configuración)

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd azure-rag-chat
```

### 2. Crear y activar entorno virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto basándote en `.env.example`:

```bash
cp .env.example .env
```

Edita el archivo `.env` y completa con tus credenciales de Azure:

```env
# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT="https://<tu-servicio-search>.search.windows.net"
AZURE_SEARCH_INDEX="biomed-manuals-demo-index"
AZURE_SEARCH_API_KEY="<tu-api-key-search>"

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT="https://<tu-recurso-openai>.openai.azure.com"
AZURE_OPENAI_API_KEY="<tu-api-key-openai>"
AZURE_OPENAI_DEPLOYMENT="<nombre-del-deployment-del-modelo>"

# Azure Speech Services Configuration (Opcional - para funcionalidad de voz)
# Si no configuras estas variables, la aplicación funcionará solo con texto
AZURE_SPEECH_API_KEY="<tu-api-key-speech>"
AZURE_SPEECH_REGION="<tu-region-speech>"
AZURE_SPEECH_LANGUAGE="es-ES"
AZURE_SPEECH_VOICE="es-ES-ElviraNeural"

# Streamlit Configuration (opcional)
STREAMLIT_SERVER_PORT="8501"
```

### 📍 Cómo obtener las credenciales de Azure Speech Services

Si quieres habilitar la funcionalidad de voz, necesitas crear un recurso de **Speech Services** en Azure:

#### 1. Crear el recurso de Speech Services

1. Ve al [Portal de Azure](https://portal.azure.com)
2. Haz clic en **"Crear un recurso"** o **"+ Crear"**
3. Busca **"Speech"** o **"Speech Services"**
4. Selecciona **"Speech"** de Microsoft
5. Haz clic en **"Crear"**
6. Completa el formulario:
   - **Suscripción**: Selecciona tu suscripción
   - **Grupo de recursos**: Crea uno nuevo o usa uno existente
   - **Región**: Selecciona una región cercana (ej: `eastus`, `westeurope`, `southcentralus`)
   - **Nombre**: Elige un nombre único para tu recurso (ej: `mi-speech-service`)
   - **Plan de tarifa**: Selecciona `Free F0` (para pruebas) o `Standard S0` (para producción)
7. Haz clic en **"Revisar y crear"** y luego en **"Crear"**

#### 2. Obtener la API Key y la Región

Una vez creado el recurso:

1. Ve a tu recurso de Speech Services en el Portal de Azure
2. En el menú lateral, busca la sección **"Claves y punto de conexión"** (Keys and Endpoint)
3. Ahí encontrarás:
   - **KEY 1** o **KEY 2**: Esta es tu `AZURE_SPEECH_API_KEY`
   - **Ubicación/Región**: Esta es tu `AZURE_SPEECH_REGION` (ej: `eastus`, `westeurope`)

#### 3. Configurar idioma y voz

- **AZURE_SPEECH_LANGUAGE**: Código de idioma (ej: `es-ES` para español de España, `es-MX` para español de México)
- **AZURE_SPEECH_VOICE**: Nombre de la voz neural. Algunas opciones en español:
  - `es-ES-ElviraNeural` (femenina, España)
  - `es-ES-AlvaroNeural` (masculina, España)
  - `es-MX-DaliaNeural` (femenina, México)
  - `es-MX-JorgeNeural` (masculina, México)

Puedes ver todas las voces disponibles en: [Documentación de voces de Azure](https://learn.microsoft.com/azure/ai-services/speech-service/language-support?tabs=tts)

#### 4. Ejemplo de configuración completa

```env
AZURE_SPEECH_API_KEY="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
AZURE_SPEECH_REGION="eastus"
AZURE_SPEECH_LANGUAGE="es-ES"
AZURE_SPEECH_VOICE="es-ES-ElviraNeural"
```

**Nota**: Si no configuras estas variables, la aplicación funcionará normalmente pero solo con entrada de texto (sin funcionalidad de voz).

## 🚀 Ejecutar la Aplicación

Una vez configurado todo, ejecuta:

**Windows (PowerShell):**
```powershell
# Activar el entorno virtual
.\venv2\Scripts\Activate.ps1

# Si tienes problemas con la política de ejecución, ejecuta primero:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Ejecutar la aplicación
streamlit run app/main.py
```

**Windows (CMD):**
```cmd
venv2\Scripts\activate
streamlit run app/main.py
```

**Linux/Mac:**
```bash
source venv2/bin/activate
streamlit run app/main.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

## 📁 Estructura del Proyecto

```
azure-rag-chat/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Aplicación principal de Streamlit
│   ├── config.py                  # Gestión de configuración y variables de entorno
│   └── services/
│       ├── __init__.py
│       ├── azure_search_client.py # Cliente para Azure AI Search
│       ├── azure_openai_client.py # Cliente para Azure OpenAI
│       ├── azure_speech_client.py  # Cliente para Azure Speech Services (STT/TTS)
│       └── rag_pipeline.py        # Pipeline RAG que orquesta todo
├── docs/
│   ├── search-index-demo.json          # Esquema simplificado de índice (demo)
│   └── search-index-prod-example.json  # Esquema completo para producción
├── .env.example                  # Plantilla de variables de entorno
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Este archivo
```

## 📊 Esquema del Índice

### Índice Real en Azure (`biomed-manuals-demo-index`)

La aplicación está configurada para trabajar con el índice **biomed-manuals-demo-index** que debe estar creado en Azure AI Search. Este índice utiliza el siguiente esquema:

**Campos principales que usa la aplicación:**
- `content` (String, searchable, retrievable): Texto extraído de los chunks de los manuales biomédicos. Este es el campo principal sobre el que se realiza la búsqueda textual.
- `metadata_storage_name` (String, filterable, sortable, facetable, retrievable): Nombre del archivo PDF de origen. La aplicación lo mapea internamente como "source" para mostrarlo en la interfaz.
- `metadata_storage_path` (String, key, retrievable): Ruta de almacenamiento del documento. Este campo es la clave (key) del índice.

**Notas:**
- El índice no incluye campos como `id`, `source` directo, `pageNumber`, `contentVector` ni configuración de búsqueda vectorial.
- La aplicación realiza búsqueda textual estándar sobre el campo `content`.
- Los archivos JSON en `docs/` (`search-index-demo.json` y `search-index-prod-example.json`) fueron diseños iniciales de ejemplo, pero la implementación actual está adaptada al esquema real del índice creado en Azure Portal.

## 🎯 Uso de la Aplicación

1. **Abre la aplicación** en tu navegador (se abre automáticamente al ejecutar Streamlit).

2. **Ajusta parámetros** en la barra lateral (opcional):
   - `top_k`: Número de documentos a recuperar (1-10)
   - `temperature`: Temperatura del modelo (0.0-1.0)

3. **Haz tu pregunta** de dos formas:
   - **Texto**: Escribe tu pregunta en el campo de chat y presiona Enter o haz clic en enviar.
   - **Voz** (si está configurado): Haz clic en el botón de micrófono 🎙️ integrado en el campo de chat, graba tu pregunta y envía.
   
   Ejemplos de preguntas:
   - "¿Cómo calibro el sensor de oxígeno del modelo X?"
   - "¿Cuál es el procedimiento de mantenimiento preventivo?"
   - "¿Qué código de error significa E-123?"
   - "¿Cómo cambio el filtro del dispositivo Y?"

4. **Revisa la respuesta** que aparece en tiempo real (streaming) y las fuentes utilizadas (nombre del PDF).

5. **Escucha la respuesta** (si está configurado Azure Speech): La respuesta se reproduce automáticamente en audio después de generarse.

6. **Continúa la conversación** haciendo más preguntas (puedes alternar entre texto y voz).

7. **Limpia la conversación** usando el botón en la barra lateral cuando quieras empezar de nuevo.

## 🔍 Características

- ✅ Interfaz de chat intuitiva con Streamlit
- ✅ **Entrada de voz integrada**: Graba preguntas directamente desde el campo de chat
- ✅ **Salida de voz**: Respuestas habladas automáticamente (opcional)
- ✅ Búsqueda semántica en manuales biomédicos
- ✅ Generación de respuestas contextualizadas con streaming en tiempo real
- ✅ Visualización de fuentes (PDF y relevancia)
- ✅ Parámetros ajustables (top_k, temperature)
- ✅ Manejo de errores robusto
- ✅ Historial de conversación completo
- ✅ Flujo unificado: todas las preguntas (texto o voz) aparecen en el mismo chat

## 📝 Notas Técnicas

- La aplicación usa **búsqueda por texto** por defecto. El código está preparado para usar búsqueda vectorial si proporcionas embeddings.
- El modelo de Azure OpenAI debe ser un modelo de **chat** (por ejemplo, GPT-4, GPT-3.5-turbo).
- La versión de la API de Azure OpenAI usada es `2024-02-15-preview` (ajustable en `azure_openai_client.py`).
- Los chunks de los manuales deben estar previamente indexados en Azure AI Search.
- **Azure Speech Services** es opcional. Si no está configurado, la aplicación funciona solo con texto.
- El widget de chat de Streamlit integra el botón de micrófono cuando `accept_audio=True`, permitiendo grabar directamente desde el campo de entrada.
- Las respuestas de voz se generan automáticamente después de cada respuesta del asistente (si Azure Speech está configurado).

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request si tienes sugerencias o mejoras.

## 📄 Licencia

Este proyecto es una demo educativa.

---

**Desarrollado con ❤️ para field engineers de dispositivos biomédicos**

