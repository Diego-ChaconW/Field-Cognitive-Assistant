"""
Cliente para interactuar con Azure AI Search.
"""
from typing import List, Dict
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from app.config import AzureSearchConfig


class AzureSearchClient:
    """Cliente para realizar búsquedas en Azure AI Search."""
    
    def __init__(self, config: AzureSearchConfig):
        """
        Inicializa el cliente de Azure AI Search.
        
        Args:
            config: Configuración de Azure AI Search.
        """
        self.config = config
        self.client = SearchClient(
            endpoint=config.endpoint,
            index_name=config.index_name,
            credential=AzureKeyCredential(config.api_key)
        )
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Busca documentos relevantes en el índice usando búsqueda textual sobre el campo 'content'.
        
        El índice real usa:
        - content: texto de los manuales (searchable)
        - metadata_storage_name: nombre del archivo PDF (se mapea a "source")
        - metadata_storage_path: clave del documento (key field)
        
        Args:
            query: Texto de búsqueda del usuario.
            top_k: Número máximo de documentos a retornar.
        
        Returns:
            Lista de diccionarios con documentos encontrados. Cada documento contiene:
            - "content": texto del campo content
            - "source": nombre del archivo PDF (desde metadata_storage_name)
            - "path": ruta del documento (desde metadata_storage_path, para depuración)
            - "score": score de relevancia de la búsqueda
        """
        try:
            # Búsqueda textual estándar sobre el campo 'content'
            search_options = {
                "search_text": query,
                "top": top_k,
                "include_total_count": True
            }
            
            # Ejecutar búsqueda
            results = self.client.search(**search_options)
            
            # Procesar resultados y mapear campos del índice real
            documents = []
            for result in results:
                doc = {
                    "content": result.get("content", ""),
                    # Mapear metadata_storage_name -> source (nombre del PDF)
                    "source": result.get("metadata_storage_name", "Unknown"),
                    # Mapear metadata_storage_path -> path (clave del documento)
                    "path": result.get("metadata_storage_path", ""),
                    # Score de relevancia de Azure AI Search
                    "score": result.get("@search.score", 0.0)
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            # Manejo básico de errores
            raise Exception(f"Error al buscar en Azure AI Search: {str(e)}")
    
    def search_documents_text_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Busca documentos usando solo búsqueda por texto (sin vectores).
        Método de conveniencia que llama a search_documents.
        
        Args:
            query: Texto de búsqueda.
            top_k: Número máximo de documentos a retornar.
        
        Returns:
            Lista de documentos encontrados.
        """
        return self.search_documents(query=query, top_k=top_k)

