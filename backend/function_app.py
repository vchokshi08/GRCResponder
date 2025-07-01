import azure.functions as func
import logging
import json
import os
from datetime import datetime, timedelta
from urllib.request import urlopen, Request
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError, URLError
import ssl
import re
import time
from typing import List, Dict, Any, Optional
from io import BytesIO

# Azure SDK imports (confirmed working)
from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import SearchIndex, SearchField, SearchFieldDataType, SimpleField, SearchableField, VectorSearch, VectorSearchProfile, VectorSearchAlgorithmConfiguration, HnswAlgorithmConfiguration

# Other working imports
from bs4 import BeautifulSoup
import PyPDF2
import google.generativeai as genai
from dateutil.parser import parse as date_parse
import numpy as np

# Create function app
app = func.FunctionApp()

# Global configuration
STORAGE_ACCOUNT_NAME = "grcresponderdevstorage"
KEY_VAULT_NAME = "grcresponder-dev-kv"
SEARCH_SERVICE_NAME = "grcresponder-dev-search"
SEARCH_INDEX_NAME = "cpuc-documents"

# Global clients - initialized on first use
_azure_clients = None
_gemini_model = None

def get_azure_clients():
    """Get Azure clients with caching"""
    global _azure_clients
    if _azure_clients is None:
        try:
            credential = DefaultAzureCredential()
            
            kv_client = SecretClient(
                vault_url=f"https://{KEY_VAULT_NAME}.vault.azure.net/",
                credential=credential
            )
            
            storage_client = BlobServiceClient(
                account_url=f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
                credential=credential
            )
            
            search_endpoint = f"https://{SEARCH_SERVICE_NAME}.search.windows.net"
            search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=SEARCH_INDEX_NAME,
                credential=credential
            )
            
            search_index_client = SearchIndexClient(
                endpoint=search_endpoint,
                credential=credential
            )
            
            _azure_clients = (kv_client, storage_client, search_client, search_index_client)
            logging.info("Azure clients initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Azure clients: {str(e)}")
            _azure_clients = (None, None, None, None)
    
    return _azure_clients

def get_gemini_model():
    """Get Gemini model with caching"""
    global _gemini_model
    if _gemini_model is None:
        try:
            kv_client, _, _, _ = get_azure_clients()
            if kv_client:
                gemini_key = kv_client.get_secret("gemini-api-key").value
                genai.configure(api_key=gemini_key)
                _gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logging.info("Gemini model initialized successfully")
            else:
                logging.error("Cannot initialize Gemini - Key Vault client not available")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {str(e)}")
            _gemini_model = None
    
    return _gemini_model

def safe_urlopen(url: str, headers: Dict[str, str] = None, timeout: int = 30) -> str:
    """Safely open URL using urllib"""
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        if headers:
            default_headers.update(headers)
        
        req = Request(url)
        for key, value in default_headers.items():
            req.add_header(key, value)
        
        with urlopen(req, timeout=timeout, context=ssl_context) as response:
            return response.read().decode('utf-8')
            
    except Exception as e:
        logging.error(f"Error opening URL {url}: {str(e)}")
        raise

def download_pdf(url: str) -> bytes:
    """Download PDF using urllib"""
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        req = Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urlopen(req, timeout=60, context=ssl_context) as response:
            return response.read()
    except Exception as e:
        logging.error(f"Failed to download PDF from {url}: {str(e)}")
        raise

class CPUCProceedingScraper:
    """CPUC Proceeding Scraper"""
    
    def __init__(self):
        self.base_url = "https://apps.cpuc.ca.gov"
        self.search_url = f"{self.base_url}/apex/f?p=401:56"
        
    def scrape_recent_proceedings(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Scrape recent CPUC proceedings"""
        try:
            logging.info(f"Scraping CPUC proceedings from last {days_back} days")
            
            html_content = safe_urlopen(self.search_url)
            soup = BeautifulSoup(html_content, 'html.parser')
            
            proceedings = []
            
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and ('proceeding' in href.lower() or 'p=401:' in href):
                    proceeding_url = urljoin(self.base_url, href)
                    proceeding_data = self._extract_proceeding_info(link, proceeding_url)
                    if proceeding_data:
                        proceedings.append(proceeding_data)
            
            logging.info(f"Found {len(proceedings)} proceedings")
            return proceedings[:20]
            
        except Exception as e:
            logging.error(f"Failed to scrape CPUC proceedings: {str(e)}")
            return []
    
    def _extract_proceeding_info(self, link_element, url: str) -> Optional[Dict[str, Any]]:
        """Extract proceeding information"""
        try:
            text = link_element.get_text(strip=True)
            if not text or len(text) < 5:
                return None
                
            proceeding_match = re.search(r'[A-Z]\.\d{2}-\d{2}-\d{3}', text)
            proceeding_number = proceeding_match.group(0) if proceeding_match else None
            
            return {
                'proceeding_number': proceeding_number,
                'title': text,
                'url': url,
                'scraped_date': datetime.utcnow().isoformat(),
                'type': 'proceeding'
            }
        except Exception:
            return None

class DocumentDownloader:
    """Document Downloader"""
    
    def __init__(self):
        _, self.storage_client, _, _ = get_azure_clients()
        self.container_name = "documents"
        
    def download_proceeding_documents(self, proceeding: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Download documents for a proceeding"""
        try:
            if not self.storage_client:
                logging.error("Storage client not available")
                return []
                
            logging.info(f"Downloading documents for: {proceeding.get('proceeding_number')}")
            
            proceeding_url = proceeding['url']
            html_content = safe_urlopen(proceeding_url)
            soup = BeautifulSoup(html_content, 'html.parser')
            
            documents = []
            
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and href.lower().endswith('.pdf'):
                    doc_url = urljoin(proceeding_url, href)
                    doc_info = self._download_single_document(doc_url, proceeding, link.get_text(strip=True))
                    if doc_info:
                        documents.append(doc_info)
                        if len(documents) >= 5:
                            break
            
            logging.info(f"Downloaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logging.error(f"Failed to download documents: {str(e)}")
            return []
    
    def _download_single_document(self, url: str, proceeding: Dict, title: str) -> Optional[Dict[str, Any]]:
        """Download and process single PDF"""
        try:
            proceeding_num = proceeding.get('proceeding_number', 'unknown')
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            blob_name = f"{proceeding_num}/{timestamp}_{title[:30]}.pdf"
            blob_name = re.sub(r'[^\w\-_./]', '_', blob_name)
            
            pdf_content = download_pdf(url)
            text_content = self._extract_pdf_text(pdf_content)
            
            blob_client = self.storage_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            blob_client.upload_blob(pdf_content, overwrite=True)
            
            return {
                'title': title,
                'url': url,
                'blob_name': blob_name,
                'text_content': text_content,
                'proceeding_number': proceeding_num,
                'file_size': len(pdf_content),
                'download_date': datetime.utcnow().isoformat(),
                'type': 'document'
            }
            
        except Exception as e:
            logging.error(f"Failed to download document from {url}: {str(e)}")
            return None
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            return text_content.strip()
            
        except Exception as e:
            logging.error(f"Failed to extract PDF text: {str(e)}")
            return ""

class VectorDatabaseBuilder:
    """Vector Database Builder"""
    
    def __init__(self):
        _, _, self.search_client, self.search_index_client = get_azure_clients()
        
    def create_search_index(self):
        """Create search index"""
        try:
            if not self.search_index_client:
                logging.error("Search index client not available")
                return
                
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="proceeding_number", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="url", type=SearchFieldDataType.String),
                SimpleField(name="created_date", type=SearchFieldDataType.DateTimeOffset, filterable=True),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    vector_search_dimensions=768,
                    vector_search_profile_name="myHnswProfile"
                )
            ]
            
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(name="myHnsw")
                ]
            )
            
            index = SearchIndex(
                name=SEARCH_INDEX_NAME,
                fields=fields,
                vector_search=vector_search
            )
            
            self.search_index_client.create_or_update_index(index)
            logging.info(f"Created/updated search index: {SEARCH_INDEX_NAME}")
            
        except Exception as e:
            logging.error(f"Failed to create search index: {str(e)}")
    
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Gemini"""
        try:
            model = get_gemini_model()
            if not model:
                return []
                
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            return result['embedding']
        except Exception as e:
            logging.error(f"Failed to generate embeddings: {str(e)}")
            return []
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents"""
        try:
            if not self.search_client or not documents:
                return
                
            logging.info(f"Indexing {len(documents)} documents")
            
            search_documents = []
            for i, doc in enumerate(documents):
                content = doc.get('text_content', '')[:8000]
                embeddings = self.generate_embeddings(content)
                
                if embeddings:
                    search_doc = {
                        'id': f"{doc.get('proceeding_number', 'unknown')}_{i}",
                        'title': doc.get('title', ''),
                        'content': content,
                        'proceeding_number': doc.get('proceeding_number', ''),
                        'document_type': doc.get('type', 'document'),
                        'url': doc.get('url', ''),
                        'created_date': doc.get('download_date', datetime.utcnow().isoformat()),
                        'content_vector': embeddings
                    }
                    search_documents.append(search_doc)
            
            if search_documents:
                self.search_client.upload_documents(documents=search_documents)
                logging.info(f"Successfully indexed {len(search_documents)} documents")
            
        except Exception as e:
            logging.error(f"Failed to index documents: {str(e)}")

class RAGSearchAPI:
    """RAG Search API"""
    
    def __init__(self):
        _, _, self.search_client, _ = get_azure_clients()
        
    def search_documents(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search documents"""
        try:
            if not self.search_client:
                return {'query': query, 'results': [], 'error': 'Search client not available'}
            
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="RETRIEVAL_QUERY"
            )['embedding']
            
            vector_query = VectorizedQuery(
                vector=query_embedding,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["title", "content", "proceeding_number", "url", "created_date"]
            )
            
            documents = []
            for result in results:
                documents.append({
                    'title': result.get('title', ''),
                    'content': result.get('content', '')[:800],
                    'proceeding_number': result.get('proceeding_number', ''),
                    'url': result.get('url', ''),
                    'score': result.get('@search.score', 0),
                    'created_date': result.get('created_date', '')
                })
            
            return {
                'query': query,
                'results': documents,
                'total_results': len(documents)
            }
            
        except Exception as e:
            logging.error(f"Search failed: {str(e)}")
            return {'query': query, 'results': [], 'error': str(e)}
    
    def generate_rag_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate RAG response"""
        try:
            model = get_gemini_model()
            if not model:
                return "RAG response generation not available - Gemini model not initialized"
            
            context = "\n\n".join([
                f"Document: {doc['title']}\nContent: {doc['content'][:400]}..."
                for doc in context_docs[:3]
            ])
            
            prompt = f"""Based on these CPUC regulatory documents, answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the regulatory documents. If insufficient information, please indicate so.

Answer:"""
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logging.error(f"RAG response generation failed: {str(e)}")
            return f"Error generating response: {str(e)}"

# Initialize global instances
scraper = CPUCProceedingScraper()
downloader = DocumentDownloader()
vector_builder = VectorDatabaseBuilder()
rag_api = RAGSearchAPI()

# HTTP ENDPOINTS - Using the working naming pattern

@app.function_name(name="HttpTriggerHealth")
@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    logging.info("Health check called")
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "GRCResponder Backend - Complete Version",
            "version": "1.0.0"
        }),
        mimetype="application/json"
    )

@app.function_name(name="HttpTriggerSearch")
@app.route(route="search", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET", "POST"])
def search_documents_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """Search documents endpoint"""
    try:
        query = req.params.get('q') or req.params.get('query')
        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Query parameter 'q' is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        top_k = int(req.params.get('top_k', 5))
        search_results = rag_api.search_documents(query, top_k)
        
        return func.HttpResponse(
            json.dumps(search_results),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Search endpoint error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="HttpTriggerAsk")
@app.route(route="ask", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET", "POST"])
def ask_question_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """RAG question answering endpoint"""
    try:
        query = req.params.get('q') or req.params.get('query')
        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Query parameter 'q' is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        search_results = rag_api.search_documents(query, top_k=5)
        rag_response = rag_api.generate_rag_response(query, search_results['results'])
        
        return func.HttpResponse(
            json.dumps({
                "query": query,
                "answer": rag_response,
                "sources": search_results['results']
            }),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Ask endpoint error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="HttpTriggerScrape")
@app.route(route="scrape", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET", "POST"])
def scrape_proceedings_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """Manual scraping trigger"""
    try:
        days_back = int(req.params.get('days', 30))
        proceedings = scraper.scrape_recent_proceedings(days_back)
        
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "proceedings_found": len(proceedings),
                "proceedings": proceedings[:5]
            }),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Scrape endpoint error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="HttpTriggerStatus")
@app.route(route="status", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def system_status(req: func.HttpRequest) -> func.HttpResponse:
    """System status check"""
    try:
        kv_client, storage_client, search_client, search_index_client = get_azure_clients()
        gemini_model = get_gemini_model()
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "key_vault": "available" if kv_client else "unavailable",
                "storage": "available" if storage_client else "unavailable", 
                "search": "available" if search_client else "unavailable",
                "gemini": "available" if gemini_model else "unavailable"
            }
        }
        
        return func.HttpResponse(
            json.dumps(status),
            mimetype="application/json"
        )
        
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="HttpTriggerPipeline")
@app.route(route="pipeline", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def manual_pipeline_trigger(req: func.HttpRequest) -> func.HttpResponse:
    """Manual pipeline execution"""
    try:
        logging.info("Manual pipeline execution started")
        
        proceedings = scraper.scrape_recent_proceedings(days_back=7)
        logging.info(f"Found {len(proceedings)} proceedings")
        
        if not proceedings:
            return func.HttpResponse(
                json.dumps({
                    "status": "completed",
                    "message": "No proceedings found",
                    "proceedings_processed": 0,
                    "documents_downloaded": 0
                }),
                mimetype="application/json"
            )
        
        all_documents = []
        for proceeding in proceedings[:3]:
            documents = downloader.download_proceeding_documents(proceeding)
            all_documents.extend(documents)
        
        logging.info(f"Downloaded {len(all_documents)} documents")
        
        if all_documents:
            vector_builder.create_search_index()
            vector_builder.index_documents(all_documents)
        
        return func.HttpResponse(
            json.dumps({
                "status": "completed",
                "proceedings_processed": len(proceedings[:3]),
                "documents_downloaded": len(all_documents),
                "timestamp": datetime.utcnow().isoformat()
            }),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Manual pipeline error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

# TIMER FUNCTIONS

@app.function_name(name="TimerTriggerDaily")
@app.timer_trigger(schedule="0 0 2 * * *", arg_name="timer", run_on_startup=False)
def daily_pipeline(timer: func.TimerRequest) -> None:
    """Daily pipeline - 2 AM UTC"""
    try:
        logging.info("Starting daily pipeline")
        
        proceedings = scraper.scrape_recent_proceedings(days_back=7)
        logging.info(f"Found {len(proceedings)} proceedings")
        
        if not proceedings:
            return
        
        all_documents = []
        for proceeding in proceedings[:3]:
            documents = downloader.download_proceeding_documents(proceeding)
            all_documents.extend(documents)
        
        logging.info(f"Downloaded {len(all_documents)} documents")
        
        if all_documents:
            vector_builder.create_search_index()
            vector_builder.index_documents(all_documents)
        
        logging.info("Daily pipeline completed")
        
    except Exception as e:
        logging.error(f"Daily pipeline failed: {str(e)}")

@app.function_name(name="TimerTriggerInit")
@app.timer_trigger(schedule="0 0 3 * * SUN", arg_name="timer", run_on_startup=True)
def initialize_system(timer: func.TimerRequest) -> None:
    """System initialization"""
    try:
        logging.info("Initializing system")
        vector_builder.create_search_index()
        
        _, storage_client, _, _ = get_azure_clients()
        if storage_client:
            try:
                storage_client.create_container("documents")
            except Exception:
                pass
        
        logging.info("System initialized")
        
    except Exception as e:
        logging.error(f"System initialization failed: {str(e)}")