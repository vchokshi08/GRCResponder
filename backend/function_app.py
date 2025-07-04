import azure.functions as func
import logging
import json
import os
import google.generativeai as genai
from datetime import datetime, timedelta  # ADD timedelta
from typing import Dict, Any, List, Optional  # ADD List, Optional
import re  # ADD re
import hashlib  # ADD hashlib
import uuid  # ADD uuid
import PyPDF2  # ADD PyPDF2
import io  # ADD io
from urllib.parse import urljoin  # ADD urljoin

# Azure imports - now including Key Vault
from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

# Add these imports at the top
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from bs4 import BeautifulSoup
import aiohttp
import asyncio

# Initialize Function App
app = func.FunctionApp()

# Configuration
STORAGE_CONNECTION_STRING = os.environ.get('AzureWebJobsStorage')
KEY_VAULT_URL = os.environ.get('KEY_VAULT_URL', 'https://grcresponder-dev-kv.vault.azure.net/')
SEARCH_SERVICE_ENDPOINT = 'https://grcresponder-dev-search.search.windows.net'

# ADD MISSING CONSTANTS
REQUEST_DELAY = 2.0  # seconds between requests to avoid CPUC blocking
MAX_CONCURRENT_DOWNLOADS = 5
CHUNK_SIZE = 1000  # characters for text chunking
CHUNK_OVERLAP = 200

# Initialize Blob Service Client
blob_service_client = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])
container_client = blob_service_client.get_container_client("documents")


# ADD MISSING SERVICE MANAGER CLASS
class ServiceManager:
    """Simple service manager for Azure services"""
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.blob_client = None
        self.search_client = None
        self.gemini_client = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize all Azure services and Gemini client"""
        if self._initialized:
            return
            
        try:
            # Initialize Key Vault and get Gemini API key
            kv_client = SecretClient(vault_url=KEY_VAULT_URL, credential=self.credential)
            gemini_api_key = kv_client.get_secret("gemini-api-key").value
            
            # Configure Gemini Flash 2.0
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai.GenerativeModel(
                model_name='gemini-2.0-flash-exp',
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                }
            )
            
            # Initialize Azure Blob Storage
            self.blob_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
            
            # Initialize Azure AI Search
            search_key = kv_client.get_secret("search-admin-key").value
            self.search_client = SearchClient(
                endpoint=SEARCH_SERVICE_ENDPOINT,
                index_name="grc-documents",
                credential=AzureKeyCredential(search_key)
            )
            
            self._initialized = True
            logging.info("Azure services initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Azure services: {str(e)}")
            raise

# CREATE GLOBAL SERVICE MANAGER INSTANCE
service_manager = ServiceManager()

@app.route(route="upload_document", auth_level=func.AuthLevel.ANONYMOUS)
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Get the uploaded file
        file = req.files.get('file')
        if not file:
            return func.HttpResponse("No file uploaded", status_code=400)

        # Validate file is a PDF
        if not file.filename.endswith('.pdf'):
            return func.HttpResponse("Only PDF files are supported", status_code=400)

        # Read PDF content
        pdf_file = io.BytesIO(file.read())
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Upload PDF to Azure Blob Storage
        blob_client = container_client.get_blob_client(file.filename)
        blob_client.upload_blob(pdf_file.getvalue(), overwrite=True)

        # Return extracted text
        return func.HttpResponse(
            json.dumps({"text": text}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error in upload_document: {str(e)}")
        return func.HttpResponse(f"Error processing file: {str(e)}", status_code=500)

@app.function_name("HealthCheck")
@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check with Azure Storage and Key Vault test"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "message": "GRCResponder with Storage + Key Vault!",
            "version": "2.1.0"
        }
        
        # Test Azure Storage
        if STORAGE_CONNECTION_STRING:
            try:
                blob_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
                containers = list(blob_client.list_containers())
                health_status["storage_status"] = "connected"
                health_status["storage_containers"] = [c.name for c in containers[:5]]
            except Exception as e:
                health_status["storage_status"] = f"error: {str(e)[:100]}"
        
        # Test Azure Search (add this after Key Vault test)
        try:
            # For now, we'll skip Search test since we don't have the API key
            # We'll get it from Key Vault in the next step
            health_status["search_status"] = "endpoint_configured"
            health_status["search_endpoint"] = SEARCH_SERVICE_ENDPOINT
        except Exception as e:
            health_status["search_status"] = f"error: {str(e)[:100]}"

        # Initialize Key Vault client outside try blocks
        credential = DefaultAzureCredential()
        kv_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)

        # Test Key Vault
        try:
            secret = kv_client.get_secret("gemini-api-key")
            health_status["keyvault_status"] = "connected"
            health_status["gemini_key_available"] = len(secret.value) > 0 if secret.value else False
        except Exception as e:
            health_status["keyvault_status"] = f"error: {str(e)[:100]}"

        # Test Google Gemini AI (now kv_client is available)
        try:
            gemini_key = kv_client.get_secret("gemini-api-key").value
            if gemini_key:
                genai.configure(api_key=gemini_key)
                health_status["gemini_status"] = "configured"
                health_status["gemini_model"] = "gemini-2.0-flash-exp"
            else:
                health_status["gemini_status"] = "no_api_key"
        except Exception as e:
            health_status["gemini_status"] = f"error: {str(e)[:100]}"

        return func.HttpResponse(
            json.dumps(health_status, indent=2),
            status_code=200,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.function_name("SearchAPI")
@app.route(route="search", methods=["GET", "POST"], auth_level=func.AuthLevel.ANONYMOUS)
def search_api(req: func.HttpRequest) -> func.HttpResponse:
    """Search API with actual Gemini AI integration"""
    try:
        if req.method == "GET":
            query = req.params.get('q', '').strip()
        else:
            req_body = req.get_json()
            query = req_body.get('query', '').strip() if req_body else ''
        
        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Query parameter 'q' is required"}),
                status_code=400,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        # Get Gemini API key and make actual AI call
        try:
            credential = DefaultAzureCredential()
            kv_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)
            gemini_key = kv_client.get_secret("gemini-api-key").value
            
            # Configure and call Gemini
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Make actual AI call
            response = model.generate_content(f"Please provide a helpful response to this query: {query}")
            ai_response = response.text
            
        except Exception as e:
            ai_response = f"AI service temporarily unavailable. Error: {str(e)[:100]}"
        
        response_data = {
            "query": query,
            "ai_response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "status": "gemini_ai_active",
            "model": "gemini-2.0-flash-exp"
        }
        
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        logging.error(f"Search API failed: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )

@app.function_name("ChatAPI")
@app.route(route="chat", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def chat_api(req: func.HttpRequest) -> func.HttpResponse:
    """Enhanced Chat API with conversation history and RAG document retrieval"""
    try:
        req_body = req.get_json()
        if not req_body:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        message = req_body.get("message", "").strip()
        session_id = req_body.get("session_id", "default_session")
        
        if not message:
            return func.HttpResponse(
                json.dumps({"error": "Message is required"}),
                status_code=400,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        # Get conversation history
        conversation_history = get_conversation_history(session_id)
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-6:]])
            context = f"Previous conversation:\n{context}\n\nCurrent message: {message}"
        else:
            context = f"Current message: {message}"
        
        # Search for relevant documents
        search_results = search_documents(message)
        
        # Get Gemini API key and make actual AI call
        try:
            credential = DefaultAzureCredential()
            kv_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)
            gemini_key = kv_client.get_secret("gemini-api-key").value
            
            # Configure and call Gemini
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Build enhanced prompt with document context
            if search_results:
                doc_context = "\n\n".join([f"Document: {doc['filename']}\nContent: {doc['content'][:500]}..." for doc in search_results[:3]])
                enhanced_prompt = f"""
You are a helpful assistant with access to relevant documents. 

Relevant Documents:
{doc_context}

Conversation Context:
{context}

Please respond to the current message, using the relevant documents when helpful. If the documents don't contain relevant information, respond based on your general knowledge.
"""
            else:
                enhanced_prompt = f"You are a helpful assistant. Here's the conversation context:\n{context}\n\nPlease respond naturally to the current message, considering the conversation history."
            
            response = model.generate_content(enhanced_prompt)
            ai_response = response.text
            
        except Exception as e:
            ai_response = f"AI service temporarily unavailable. Error: {str(e)[:100]}"
        
        # Save conversation
        save_conversation_message(session_id, message, ai_response)
        
        response_data = {
            "response": ai_response,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "gemini_ai_active",
            "model": "gemini-2.0-flash-exp",
            "documents_found": len(search_results),
            "sources": [{"filename": doc.get("filename", "Unknown"), "relevance": doc.get("score", 0)} for doc in search_results[:3]]
        }
        
        return func.HttpResponse(
            json.dumps(response_data),
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        logging.error(f"Chat API failed: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )

def search_documents(query: str, top_k: int = 5) -> list:
    """Search for relevant documents using Azure Search"""
    try:
        # Get search API key from Key Vault
        credential = DefaultAzureCredential()
        kv_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)
        search_key = kv_client.get_secret("search-admin-key").value
        
        # Initialize search client
        search_client = SearchClient(
            endpoint=SEARCH_SERVICE_ENDPOINT,
            index_name="grc-documents",
            credential=AzureKeyCredential(search_key)
        )
        
        # Perform search
        results = search_client.search(
            search_text=query,
            top=top_k,
            select=["filename", "content", "proceeding_id"]
        )
        
        # Format results
        documents = []
        for result in results:
            documents.append({
                "filename": result.get("filename", "Unknown"),
                "content": result.get("content", ""),
                "proceeding_id": result.get("proceeding_id", ""),
                "score": result.get("@search.score", 0)
            })
        
        return documents
        
    except Exception as e:
        logging.error(f"Document search failed: {str(e)}")
        return []  # Return empty list if search fails

def get_conversation_history(session_id: str) -> list:
    """Get conversation history from Azure Blob Storage"""
    try:
        blob_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        container_name = "chat-sessions"
        
        # Create container if it doesn't exist
        try:
            container_client = blob_client.get_container_client(container_name)
            container_client.create_container()
        except Exception:
            pass  # Container already exists
        
        blob_name = f"{session_id}.json"
        blob_client_instance = blob_client.get_blob_client(container=container_name, blob=blob_name)
        
        if blob_client_instance.exists():
            blob_data = blob_client_instance.download_blob().readall()
            session_data = json.loads(blob_data)
            return session_data.get('messages', [])
        
        return []
        
    except Exception as e:
        logging.error(f"Failed to get conversation history: {str(e)}")
        return []

def save_conversation_message(session_id: str, user_message: str, ai_response: str):
    """Save conversation message to Azure Blob Storage"""
    try:
        blob_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        container_name = "chat-sessions"
        
        # Create container if it doesn't exist
        try:
            container_client = blob_client.get_container_client(container_name)
            container_client.create_container()
        except Exception:
            pass  # Container already exists
        
        # Get existing history
        history = get_conversation_history(session_id)
        
        # Add new messages
        history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        history.append({
            "role": "assistant", 
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 messages
        if len(history) > 20:
            history = history[-20:]
        
        # Save to blob
        blob_name = f"{session_id}.json"
        blob_client_instance = blob_client.get_blob_client(container=container_name, blob=blob_name)
        
        session_data = {
            "session_id": session_id,
            "messages": history,
            "updated_at": datetime.now().isoformat()
        }
        
        blob_client_instance.upload_blob(
            json.dumps(session_data, indent=2),
            overwrite=True
        )
        
        logging.info(f"Saved conversation for session {session_id}")
        
    except Exception as e:
        logging.error(f"Failed to save conversation: {str(e)}")

@app.function_name("TestTrigger")
@app.route(route="test", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def test_trigger(req: func.HttpRequest) -> func.HttpResponse:
    """Test endpoint with Key Vault integration"""
    return func.HttpResponse(
        json.dumps({
            "message": "🔐 Azure Key Vault Integration Added!",
            "step": "2.1 - Storage + Key Vault",
            "achievements": [
                "✅ Azure Functions working",
                "✅ Azure Storage connected", 
                "✅ Azure Key Vault integrated",
                "✅ Secure API key management"
            ],
            "next_steps": [
                "Add Azure Search",
                "Add Google Gemini AI", 
                "Add original GRC retrieval logic"
            ],
            "timestamp": datetime.now().isoformat()
        }, indent=2),
        mimetype="application/json",
        headers={"Access-Control-Allow-Origin": "*"}
    )

# CORS Handler
@app.function_name("CorsHandler")
@app.route(route="{*path}", methods=["OPTIONS"], auth_level=func.AuthLevel.ANONYMOUS)
def cors_handler(req: func.HttpRequest) -> func.HttpResponse:
    """Handle CORS preflight requests"""
    return func.HttpResponse(
        "",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "86400"
        }
    )

# Add these 3 functions to your function_app.py
# These create the complete automated pipeline like the original

# Helper Functions for Incremental Updates (add these first)
async def get_existing_proceedings() -> Dict[str, Dict]:
    """Get previously processed proceedings for incremental updates"""
    try:
        container_name = "proceedings-tracking"
        blob_name = "processed_proceedings.json"
        
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        try:
            download_stream = await blob_client.download_blob()
            existing_data = json.loads(await download_stream.readall())
            return existing_data.get("proceedings", {})
        except Exception:
            logging.info("No existing proceedings tracking file found - this appears to be the first run")
            return {}
            
    except Exception as e:
        logging.error(f"Failed to get existing proceedings: {str(e)}")
        return {}

def is_after_jan_2020(date_str: str) -> bool:
    """Check if a date is after January 1, 2020 - battle-tested from original code"""
    try:
        if not date_str:
            return False
            
        # Extract year using regex (original pattern)
        year_match = re.search(r'20\d{2}', date_str)
        if year_match:
            year = int(year_match.group(0))
            if year < 2020:
                return False
            elif year > 2020:
                return True
            # If it's exactly 2020, need to check month and day
            
        # Month name mapping (original pattern)
        month_names = {
            "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, 
            "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, 
            "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10, 
            "november": 11, "nov": 11, "december": 12, "dec": 12
        }
        
        # Try different date formats (original pattern)
        formats = [
            "%B %d, %Y",  # January 15, 2020
            "%b %d, %Y",  # Jan 15, 2020
            "%m/%d/%Y",   # 01/15/2020
            "%Y-%m-%d",   # 2020-01-15
            "%d-%m-%Y",   # 15-01-2020
            "%d %B %Y",   # 15 January 2020
            "%d %b %Y",   # 15 Jan 2020
            "%B %Y",      # January 2020
            "%b %Y",      # Jan 2020
            "%m-%Y",      # 01-2020
            "%Y/%m/%d",   # 2020/01/15
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj >= datetime(2020, 1, 1)
            except ValueError:
                continue
                
        # Default to False if we can't parse the date (conservative approach)
        logging.warning(f"Could not parse date: '{date_str}' - excluding from processing")
        return False
        
    except Exception as e:
        logging.warning(f"Date parsing error for '{date_str}': {str(e)}")
        return False

def create_proceeding_hash(proceeding: Dict) -> str:
    """Create a hash of proceeding content to detect changes"""
    content = f"{proceeding.get('title', '')}|{proceeding.get('status', '')}|{proceeding.get('last_updated', '')}|{proceeding.get('summary', '')}"
    return hashlib.md5(content.encode()).hexdigest()

async def filter_and_deduplicate_proceedings(
    all_proceedings: List[Dict], 
    existing_proceedings: Dict[str, Dict]
) -> List[Dict]:
    """Filter proceedings for Jan 1, 2020+ and implement incremental deduplication"""
    filtered_proceedings = []
    
    for proceeding in all_proceedings:
        # Step 1: Apply date filter (Jan 1, 2020+)
        if not is_after_jan_2020(proceeding.get('date_filed', '')):
            logging.debug(f"Skipping {proceeding.get('proceeding_id')} - filed before Jan 1, 2020: {proceeding.get('date_filed')}")
            continue
        
        # Step 2: Check if proceeding is new or changed (incremental logic)
        proceeding_id = proceeding.get('proceeding_id')
        current_hash = create_proceeding_hash(proceeding)
        
        if proceeding_id in existing_proceedings:
            existing_hash = existing_proceedings[proceeding_id].get("content_hash", "")
            last_processed = existing_proceedings[proceeding_id].get("last_processed", "")
            
            if current_hash == existing_hash:
                # Proceeding hasn't changed - skip unless it's been more than 30 days
                try:
                    last_proc_date = datetime.fromisoformat(last_processed.replace('Z', '+00:00'))
                    if datetime.now().replace(tzinfo=last_proc_date.tzinfo) - last_proc_date < timedelta(days=30):
                        logging.debug(f"Skipping {proceeding_id} - no changes detected and processed recently")
                        continue
                except Exception:
                    pass  # If date parsing fails, include the proceeding
        
        # Step 3: This is a new or changed proceeding - include it
        logging.info(f"Including {proceeding_id} for processing (new or changed)")
        filtered_proceedings.append(proceeding)
    
    return filtered_proceedings

async def update_proceedings_tracking(proceedings: List[Dict]) -> None:
    """Update the tracking file with newly processed proceedings"""
    try:
        # Get existing tracking data
        existing_proceedings = await get_existing_proceedings()
        
        # Update with new proceedings
        for proceeding in proceedings:
            proceeding_id = proceeding.get('proceeding_id')
            content_hash = create_proceeding_hash(proceeding)
            
            existing_proceedings[proceeding_id] = {
                "content_hash": content_hash,
                "last_processed": datetime.now().isoformat(),
                "title": proceeding.get('title', ''),
                "date_filed": proceeding.get('date_filed', ''),
                "status": proceeding.get('status', ''),
                "category": proceeding.get('category', '')
            }
        
        # Save updated tracking data
        container_name = "proceedings-tracking"
        blob_name = "processed_proceedings.json"
        
        tracking_data = {
            "last_updated": datetime.now().isoformat(),
            "total_proceedings": len(existing_proceedings),
            "proceedings": existing_proceedings
        }
        
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(
            json.dumps(tracking_data, indent=2, default=str),
            overwrite=True
        )
        
        logging.info(f"Updated proceedings tracking with {len(proceedings)} new/changed proceedings")
        
    except Exception as e:
        logging.error(f"Failed to update proceedings tracking: {str(e)}")

# Add these existing functions if you don't have them already
async def get_download_queue() -> List[Dict]:
    """Get items from download queue"""
    try:
        container_name = "processing-queue"
        container_client = service_manager.blob_client.get_container_client(container_name)
        queue_items = []
        
        async for blob in container_client.list_blobs():
            if blob.name.startswith("download_queue_"):
                blob_client = service_manager.blob_client.get_blob_client(
                    container=container_name,
                    blob=blob.name
                )
                
                download_stream = await blob_client.download_blob()
                queue_data = json.loads(await download_stream.readall())
                queue_items.extend(queue_data)
                
                await blob_client.delete_blob()
        
        return queue_items
    except Exception as e:
        logging.error(f"Failed to get download queue: {str(e)}")
        return []

async def store_proceedings_metadata(proceedings: List[Dict]) -> None:
    """Store proceedings metadata in Azure Blob Storage"""
    try:
        container_name = "proceedings-metadata"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f"proceedings_{timestamp}.json"
        
        json_data = json.dumps(proceedings, indent=2, default=str)
        
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(json_data, overwrite=True)
        logging.info(f"Stored proceedings metadata: {blob_name}")
        
    except Exception as e:
        logging.error(f"Failed to store proceedings metadata: {str(e)}")
        raise

async def create_download_queue(proceedings: List[Dict]) -> None:
    """Create download queue for Document Downloader"""
    try:
        container_name = "processing-queue"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f"download_queue_{timestamp}.json"
        
        queue_data = []
        for proceeding in proceedings:
            queue_item = {
                "proceeding_id": proceeding.get("proceeding_id"),
                "proceeding_url": proceeding.get("url"),
                "title": proceeding.get("title"),
                "category": proceeding.get("category"),
                "status": proceeding.get("status"),
                "queued_at": datetime.now().isoformat(),
                "retry_count": 0
            }
            queue_data.append(queue_item)
        
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(json.dumps(queue_data, indent=2), overwrite=True)
        logging.info(f"Created download queue with {len(queue_data)} items: {blob_name}")
        
    except Exception as e:
        logging.error(f"Failed to create download queue: {str(e)}")
        raise

async def create_vector_queue(documents: List[Dict]) -> None:
    """Create queue for vector processing"""
    try:
        container_name = "processing-queue"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f"vector_queue_{timestamp}.json"
        
        queue_data = []
        for doc in documents:
            queue_item = {
                "document_id": doc["document_id"],
                "proceeding_id": doc["proceeding_id"],
                "filename": doc["filename"],
                "blob_url": doc["blob_url"],
                "text_content": doc["text_content"],
                "file_size": doc["file_size"],
                "metadata": doc["metadata"],
                "queued_at": datetime.now().isoformat()
            }
            queue_data.append(queue_item)
        
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(json.dumps(queue_data, indent=2), overwrite=True)
        logging.info(f"Created vector queue with {len(documents)} items: {blob_name}")
        
    except Exception as e:
        logging.error(f"Failed to create vector queue: {str(e)}")
        raise

async def get_vector_queue() -> List[Dict]:
    """Get items from vector processing queue"""
    try:
        container_name = "processing-queue"
        container_client = service_manager.blob_client.get_container_client(container_name)
        queue_items = []
        
        async for blob in container_client.list_blobs():
            if blob.name.startswith("vector_queue_"):
                blob_client = service_manager.blob_client.get_blob_client(
                    container=container_name,
                    blob=blob.name 
                )
                
                download_stream = await blob_client.download_blob()
                queue_data = json.loads(await download_stream.readall())
                queue_items.extend(queue_data)
                
                await blob_client.delete_blob()
        
        return queue_items
    except Exception as e:
        logging.error(f"Failed to get vector queue: {str(e)}")
        return []

# Part 1: Scheduled Proceeding Scraper (runs 3x per week)
@app.function_name("ScheduledProceedingScraper")
@app.schedule(schedule="0 0 6 * * 2,4,6", arg_name="timer", run_on_startup=False)  # Tues, Thurs, Sat at 6 AM
async def scheduled_proceeding_scraper(timer: func.TimerRequest) -> None:
    """
    Scheduled CPUC Proceeding Scraper - Part 1 of 3-step pipeline
    Runs 3x per week to discover new and updated proceedings with incremental updates
    """
    logging.info("Starting Scheduled CPUC Proceeding Scraper with Incremental Updates - Step 1/3")
    
    try:
        await service_manager.initialize()
        
        # Step 1: Get previously processed proceedings
        existing_proceedings = await get_existing_proceedings()
        logging.info(f"Found {len(existing_proceedings)} previously processed proceedings")
        
        # Step 2: Scrape proceedings using comprehensive approach
        all_proceedings = await scrape_all_cpuc_proceedings()
        
        if not all_proceedings:
            logging.warning("No proceedings found during scraping")
            return
        
        # Step 3: Apply date filtering and incremental logic
        filtered_proceedings = await filter_and_deduplicate_proceedings(
            all_proceedings, 
            existing_proceedings
        )
        
        if not filtered_proceedings:
            logging.info("No new proceedings to process after filtering and deduplication")
            return
        
        logging.info(f"Processing {len(filtered_proceedings)} new/updated proceedings (filtered from {len(all_proceedings)} total)")
        
        # Step 4: Store proceedings metadata
        await store_proceedings_metadata(filtered_proceedings)
        
        # Step 5: Create download queue for Document Downloader
        await create_download_queue(filtered_proceedings)
        
        # Step 6: Update tracking for incremental processing
        await update_proceedings_tracking(filtered_proceedings)
        
        logging.info(f"Scheduled proceeding scraper completed successfully. Queued {len(filtered_proceedings)} new proceedings")
        
    except Exception as e:
        logging.error(f"Scheduled proceeding scraper failed: {str(e)}")
        raise

async def scrape_all_cpuc_proceedings() -> List[Dict]:
    """Comprehensive CPUC proceeding scraper - based on original threaded_webscraper.py"""
    
    # Start with known recent proceeding IDs and expand
    base_proceeding_ids = [
        "A2502016", "A2502011", "A2502006", "A2502003", "A2502019",
        "A2408011", "A2408015", "A2408023", "A2408007", "A2408014",
        "A2405012", "A2405015", "A2405008", "A2405003", "A2405019",
        "A2403005", "A2403012", "A2403008", "A2403015", "A2403019",
        "A2401005", "A2401012", "A2401008", "A2401015", "A2401019",
        "A2312008", "A2312015", "A2312019", "A2312003", "A2312012",
        "A2311003", "A2311008", "A2311012", "A2311015", "A2311019",
        "A2308012", "A2308015", "A2308019", "A2308003", "A2308008",
        "A2307005", "A2307012", "A2307015", "A2307019", "A2307003",
        "A2306002", "A2306008", "A2306012", "A2306015", "A2306019",
        "A2305008", "A2305012", "A2305015", "A2305019", "A2305003",
        "A2304010", "A2304015", "A2304019", "A2304003", "A2304008",
        "A2303005", "A2303012", "A2303015", "A2303019", "A2303003",
        "A2302012", "A2302015", "A2302019", "A2302003", "A2302008",
        "A2301005", "A2301012", "A2301015", "A2301019", "A2301003"
    ]
    
    # Add Rulemaking proceedings
    for year in [25, 24, 23, 22, 21, 20]:
        for month in range(1, 13):
            for num in range(1, 20):
                base_proceeding_ids.append(f"R.{year:02d}-{month:02d}-{num:03d}")
    
    session_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    valid_proceedings = []
    
    async with aiohttp.ClientSession(headers=session_headers) as session:
        # Process proceedings in batches to avoid overwhelming CPUC
        batch_size = 5
        
        for i in range(0, len(base_proceeding_ids), batch_size):
            batch = base_proceeding_ids[i:i+batch_size]
            
            # Process batch concurrently
            tasks = [parse_single_proceeding(session, proc_id) for proc_id in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logging.warning(f"Batch processing error: {str(result)}")
                    continue
                if result:
                    valid_proceedings.append(result)
            
            # Rate limiting between batches
            await asyncio.sleep(REQUEST_DELAY)
    
    return valid_proceedings

async def parse_single_proceeding(session: aiohttp.ClientSession, proceeding_id: str) -> Optional[Dict]:
    """Parse single proceeding - based on original patterns"""
    detail_url = "https://apps.cpuc.ca.gov/apex/f"
    params = {
        "p": f"401:56::::RP,57,RIR:P5_PROCEEDING_SELECT:{proceeding_id}"
    }
    
    try:
        async with session.get(detail_url, params=params, timeout=30) as response:
            if response.status != 200:
                return None
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract proceeding metadata using original regex patterns
            page_text = soup.get_text()
            
            # Check if proceeding exists (has real content)
            if "No data found" in page_text or len(page_text.strip()) < 100:
                return None
            
            proceeding_data = {
                'proceeding_id': proceeding_id,
                'title': '',
                'url': f"https://apps.cpuc.ca.gov/apex/f?p=401:56::::RP,57,RIR:P5_PROCEEDING_SELECT:{proceeding_id}",
                'status': 'Active',
                'date_filed': '',
                'last_updated': datetime.now().isoformat(),
                'category': 'Energy',
                'summary': '',
                'filed_by': '',
                'current_status': '',
                'description': ''
            }
            
            # Extract metadata using original regex patterns
            filed_by_match = re.search(r'Filed By:\s*([^\n]+)', page_text)
            if filed_by_match:
                proceeding_data['filed_by'] = filed_by_match.group(1).strip()
            
            filing_date_match = re.search(r'Filing Date:\s*([^\n]+?)(?=Category:|$)', page_text)
            if filing_date_match:
                proceeding_data['date_filed'] = filing_date_match.group(1).strip()
            
            category_match = re.search(r'Category:\s*([^\n]+?)(?=Current Status:|$)', page_text)
            if category_match:
                proceeding_data['category'] = category_match.group(1).strip()
            
            status_match = re.search(r'Current Status:\s*([^\n]+?)(?=Description:|$)', page_text)
            if status_match:
                proceeding_data['current_status'] = status_match.group(1).strip()
                proceeding_data['status'] = status_match.group(1).strip()
            
            desc_match = re.search(r'Description:\s*([^\n]+)', page_text)
            if desc_match:
                proceeding_data['description'] = desc_match.group(1).strip()
                proceeding_data['title'] = desc_match.group(1).strip()
                proceeding_data['summary'] = desc_match.group(1).strip()[:200]
            
            # Only return if we have essential data
            if proceeding_data['date_filed'] or proceeding_data['description']:
                return proceeding_data
            
            return None
            
    except Exception as e:
        logging.debug(f"Error parsing proceeding {proceeding_id}: {str(e)}")
        return None

# Part 2: Document Downloader (runs every 6 hours)
@app.function_name("ScheduledDocumentDownloader")
@app.schedule(schedule="0 30 */6 * * *", arg_name="timer", run_on_startup=False)  # Every 6 hours
async def scheduled_document_downloader(timer: func.TimerRequest) -> None:
    """
    Scheduled Document Downloader - Part 2 of 3-step pipeline
    Downloads and processes PDF documents from queued proceedings
    """
    logging.info("Starting Scheduled Document Downloader with Deduplication - Step 2/3")
    
    try:
        await service_manager.initialize()
        
        # Get queued proceedings for download
        queue_items = await get_download_queue()
        
        if not queue_items:
            logging.info("No items in download queue")
            return
        
        # Get existing document hashes for deduplication
        existing_documents = await get_existing_document_hashes()
        
        # Process downloads with enhanced deduplication
        processed_documents = await process_download_queue_with_deduplication(queue_items, existing_documents)
        
        if processed_documents:
            # Create vector processing queue
            await create_vector_queue(processed_documents)
            
            # Update document tracking
            await update_document_tracking(processed_documents)
        
        logging.info(f"Scheduled document downloader completed. Processed {len(processed_documents)} new documents")
        
    except Exception as e:
        logging.error(f"Scheduled document downloader failed: {str(e)}")
        raise

async def get_existing_document_hashes() -> Dict[str, str]:
    """Get hashes of already processed documents for deduplication"""
    try:
        container_name = "document-tracking"
        blob_name = "processed_documents.json"
        
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        try:
            download_stream = await blob_client.download_blob()
            existing_data = json.loads(await download_stream.readall())
            return existing_data.get("document_hashes", {})
        except Exception:
            return {}
            
    except Exception as e:
        logging.error(f"Failed to get existing document hashes: {str(e)}")
        return {}

async def process_download_queue_with_deduplication(queue_items: List[Dict], existing_documents: Dict[str, str]) -> List[Dict]:
    """Process download queue with deduplication logic"""
    processed_documents = []
    
    # Process in batches to avoid overwhelming CPUC servers
    batch_size = MAX_CONCURRENT_DOWNLOADS
    
    for i in range(0, len(queue_items), batch_size):
        batch = queue_items[i:i+batch_size]
        
        # Process batch concurrently
        tasks = [process_single_proceeding_with_dedup(item, existing_documents) for item in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                logging.error(f"Batch processing error: {str(result)}")
                continue
            if result:
                processed_documents.extend(result)
        
        # Rate limiting between batches
        if i + batch_size < len(queue_items):
            await asyncio.sleep(REQUEST_DELAY * batch_size)
    
    return processed_documents

async def process_single_proceeding_with_dedup(queue_item: Dict, existing_documents: Dict[str, str]) -> List[Dict]:
    """Process documents with deduplication logic"""
    documents = []
    
    try:
        proceeding_url = queue_item.get("proceeding_url", "")
        proceeding_id = queue_item.get("proceeding_id", "")
        
        # Discover document links from proceeding page
        document_urls = await discover_proceeding_documents_enhanced(proceeding_url)
        
        async with aiohttp.ClientSession() as session:
            for doc_url, doc_filename in document_urls:
                try:
                    # Create document hash for deduplication
                    doc_hash = hashlib.md5(f"{proceeding_id}|{doc_url}".encode()).hexdigest()
                    
                    # Check if document already processed
                    if doc_hash in existing_documents:
                        logging.debug(f"Skipping {doc_filename} - already processed")
                        continue
                    
                    # Rate limiting between document downloads
                    await asyncio.sleep(REQUEST_DELAY)
                    
                    # Download PDF content
                    pdf_content = await download_pdf_document_enhanced(session, doc_url)
                    if not pdf_content:
                        continue
                    
                    # Extract text content
                    text_content = extract_text_from_pdf_enhanced(pdf_content)
                    if not text_content or len(text_content.strip()) < 100:
                        continue
                    
                    # Store PDF in blob storage
                    blob_url = await store_document_blob_enhanced(pdf_content, proceeding_id, doc_filename)
                    
                    # Create processed document
                    document = {
                        "document_id": str(uuid.uuid4()),
                        "proceeding_id": proceeding_id,
                        "filename": doc_filename,
                        "blob_url": blob_url,
                        "text_content": text_content,
                        "file_size": len(pdf_content),
                        "processed_timestamp": datetime.now().isoformat(),
                        "metadata": {
                            "source_url": doc_url,
                            "proceeding_title": queue_item.get("title", ""),
                            "category": queue_item.get("category", ""),
                            "text_length": len(text_content),
                            "status": queue_item.get("status", ""),
                            "document_hash": doc_hash
                        }
                    }
                    
                    documents.append(document)
                    
                    # Update existing documents tracking
                    existing_documents[doc_hash] = datetime.now().isoformat()
                    
                except Exception as e:
                    logging.warning(f"Failed to process document {doc_url}: {str(e)}")
                    continue
        
    except Exception as e:
        logging.error(f"Failed to process proceeding {queue_item.get('proceeding_id')}: {str(e)}")
    
    return documents

async def discover_proceeding_documents_enhanced(proceeding_url: str) -> List[tuple]:
    """Enhanced document discovery based on original patterns"""
    document_urls = []
    
    if not proceeding_url:
        return document_urls
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(proceeding_url) as response:
                if response.status != 200:
                    return document_urls
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find tab links first (original pattern)
                tabs_ul = soup.select_one("div.sHorizontalTabsInner ul")
                if tabs_ul:
                    for li in tabs_ul.find_all("li"):
                        a_tag = li.find("a")
                        if a_tag:
                            tab_title = a_tag.get_text(strip=True)
                            if tab_title in ["Documents", "Rulings", "Decisions"]:
                                href = a_tag.get("href")
                                tab_url = urljoin("https://apps.cpuc.ca.gov/apex/", href)
                                
                                # Get documents from this tab
                                tab_documents = await get_documents_from_tab(tab_url, tab_title)
                                document_urls.extend(tab_documents)
                
                # Also look for direct PDF links
                pdf_patterns = [
                    'a[href$=".pdf"]',
                    'a[href*=".pdf"]',
                    'a[href*="PublishedDocs"]'
                ]
                
                for pattern in pdf_patterns:
                    links = soup.select(pattern)
                    for link in links:
                        href = link.get('href')
                        if href and '.pdf' in href.lower():
                            full_url = urljoin(proceeding_url, href)
                            filename = link.get_text().strip() or href.split('/')[-1]
                            document_urls.append((full_url, filename))
        
    except Exception as e:
        logging.warning(f"Failed to discover documents from {proceeding_url}: {str(e)}")
    
    return document_urls

async def get_documents_from_tab(tab_url: str, tab_name: str) -> List[tuple]:
    """Get documents from specific tab - based on original tab parsing"""
    documents = []
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(tab_url) as response:
                if response.status != 200:
                    return documents
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Look for interactive report table (original pattern)
                table = soup.select_one("table.a-IRR-table")
                if table:
                    rows = table.select("tr")[1:]  # Skip header
                    
                    for row in rows:
                        cells = row.select("td.u-tL")
                        if len(cells) >= 2:
                            # Look for PDF links in cells
                            for cell in cells:
                                link_tag = cell.find("a")
                                if link_tag and link_tag.get("href"):
                                    href = link_tag["href"]
                                    if '.pdf' in href.lower() or 'PublishedDocs' in href:
                                        if href.startswith("http://"):
                                            href = href.replace("http://", "https://")
                                        full_url = urljoin(tab_url, href)
                                        filename = link_tag.get_text().strip() or href.split('/')[-1]
                                        documents.append((full_url, f"{tab_name}_{filename}"))
        
    except Exception as e:
        logging.warning(f"Failed to get documents from tab {tab_url}: {str(e)}")
    
    return documents

async def download_pdf_document_enhanced(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Enhanced PDF download with better error handling"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200:
                content = await response.read()
                
                # Check if it's actually a PDF
                if content.startswith(b'%PDF') and len(content) > 1000:
                    return content
                    
    except Exception as e:
        logging.warning(f"Failed to download PDF from {url}: {str(e)}")
    
    return None

def extract_text_from_pdf_enhanced(pdf_content: bytes) -> str:
    """Enhanced PDF text extraction based on original logic"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text + "\n"
            except Exception as e:
                logging.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                continue
        
        # Clean and normalize text (original pattern)
        text_content = re.sub(r'\s+', ' ', text_content)
        text_content = text_content.strip()
        
        return text_content
        
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {str(e)}")
        return ""

async def store_document_blob_enhanced(pdf_content: bytes, proceeding_id: str, filename: str) -> str:
    """Enhanced document storage with better filename handling"""
    try:
        container_name = "documents"
        # Clean filename for blob storage
        clean_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        blob_name = f"{proceeding_id}/{clean_filename}"
        
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(pdf_content, overwrite=True, content_type='application/pdf')
        return blob_client.url
        
    except Exception as e:
        logging.error(f"Failed to store document blob: {str(e)}")
        raise

async def update_document_tracking(documents: List[Dict]) -> None:
    """Update document tracking for deduplication"""
    try:
        existing_hashes = await get_existing_document_hashes()
        
        # Add new document hashes
        for doc in documents:
            doc_hash = doc["metadata"]["document_hash"]
            existing_hashes[doc_hash] = doc["processed_timestamp"]
        
        # Save updated tracking data
        container_name = "document-tracking"
        blob_name = "processed_documents.json"
        
        tracking_data = {
            "last_updated": datetime.now().isoformat(),
            "total_documents": len(existing_hashes),
            "document_hashes": existing_hashes
        }
        
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(
            json.dumps(tracking_data, indent=2, default=str),
            overwrite=True
        )
        
        logging.info(f"Updated document tracking with {len(documents)} new documents")
        
    except Exception as e:
        logging.error(f"Failed to update document tracking: {str(e)}")

# Part 3: Vector Database Builder (runs every 4 hours)  
@app.function_name("ScheduledVectorBuilder")
@app.schedule(schedule="0 0 */4 * * *", arg_name="timer", run_on_startup=False)  # Every 4 hours
async def scheduled_vector_builder(timer: func.TimerRequest) -> None:
    """
    Scheduled Vector Database Builder - Part 3 of 3-step pipeline
    Builds vector embeddings using Gemini Flash 2.0 and indexes in Azure AI Search
    """
    logging.info("Starting Scheduled Vector Database Builder - Step 3/3")
    
    try:
        await service_manager.initialize()
        
        # Get queued documents for vector processing
        queue_items = await get_vector_queue()
        
        if not queue_items:
            logging.info("No items in vector processing queue")
            return
        
        # Process documents with multithreading (original pattern)
        await process_vector_queue_enhanced(queue_items)
        
        logging.info(f"Scheduled vector builder completed. Processed {len(queue_items)} documents")
        
    except Exception as e:
        logging.error(f"Scheduled vector builder failed: {str(e)}")
        raise

async def process_vector_queue_enhanced(queue_items: List[Dict]) -> None:
    """Enhanced vector processing based on original multithreaded_insert.py patterns"""
    
    # Process in batches for better memory management
    batch_size = 10
    
    for i in range(0, len(queue_items), batch_size):
        batch = queue_items[i:i+batch_size]
        
        # Process each document in the batch
        search_documents = []
        for doc_data in batch:
            try:
                # Create text chunks - based on original chunking strategy
                chunks = create_text_chunks_enhanced(doc_data["text_content"])
                
                # Generate embeddings for each chunk using Gemini Flash 2.0
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_embeddings = await generate_embeddings_with_retry_enhanced(chunk)
                    
                    if chunk_embeddings:
                        search_doc = {
                            "id": f"{doc_data['document_id']}_chunk_{chunk_idx}",
                            "document_id": doc_data["document_id"],
                            "proceeding_id": doc_data["proceeding_id"],
                            "filename": doc_data["filename"],
                            "blob_url": doc_data["blob_url"],
                            "content": chunk,
                            "content_vector": chunk_embeddings,
                            "metadata": json.dumps(doc_data["metadata"]),
                            "file_size": doc_data["file_size"],
                            "chunk_index": chunk_idx,
                            "total_chunks": len(chunks),
                            "processed_at": datetime.now().isoformat()
                        }
                        search_documents.append(search_doc)
                
            except Exception as e:
                logging.error(f"Failed to process document {doc_data.get('document_id')}: {str(e)}")
                continue
        
        # Index batch of documents in Azure AI Search
        if search_documents:
            await index_documents_batch_enhanced(search_documents)

def create_text_chunks_enhanced(text: str) -> List[str]:
    """Enhanced text chunking based on original logic"""
    if not text or len(text.strip()) < 100:
        return []
    
    chunks = []
    words = text.split()
    
    # Calculate chunk size in words (original pattern)
    chunk_size_words = CHUNK_SIZE // 5  # Approximate words per chunk
    overlap_words = CHUNK_OVERLAP // 5
    
    for i in range(0, len(words), chunk_size_words - overlap_words):
        chunk_words = words[i:i + chunk_size_words]
        chunk_text = " ".join(chunk_words)
        
        if len(chunk_text.strip()) > 50:  # Minimum chunk size
            chunks.append(chunk_text.strip())
        
        if i + chunk_size_words >= len(words):
            break
    
    return chunks

async def generate_embeddings_with_retry_enhanced(text: str, max_retries: int = 3) -> List[float]:
    """Enhanced embedding generation with retry logic"""
    for attempt in range(max_retries):
        try:
            # Use Gemini Flash 2.0 for embeddings
            result = await service_manager.gemini_client.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
                title="CPUC Regulatory Document"
            )
            
            return result['embedding']
            
        except Exception as e:
            logging.warning(f"Embedding generation attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                logging.error(f"Failed to generate embeddings after {max_retries} attempts")
                return []

async def index_documents_batch_enhanced(documents: List[Dict]) -> None:
    """Enhanced document indexing in Azure AI Search"""
    try:
        # Upload documents to search index
        result = await service_manager.search_client.upload_documents(documents)
        
        successful_uploads = sum(1 for r in result if r.succeeded)
        failed_uploads = len(result) - successful_uploads
        
        if failed_uploads > 0:
            logging.warning(f"Vector indexing: {successful_uploads} succeeded, {failed_uploads} failed")
        else:
            logging.info(f"Successfully indexed {successful_uploads} document chunks")
        
    except Exception as e:
        logging.error(f"Failed to index document batch: {str(e)}")
        raise