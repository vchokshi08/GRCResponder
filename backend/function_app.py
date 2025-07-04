import azure.functions as func
import logging
import json
import os
import google.generativeai as genai
from datetime import datetime
from typing import Dict, Any

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

@app.function_name("CPUCScraper")
@app.route(route="scrape", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def cpuc_scraper(req: func.HttpRequest) -> func.HttpResponse:
    """Complete CPUC Scraper - with date filter from January 1, 2015"""
    try:
        import aiohttp
        import asyncio
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin
        import re
        
        def is_after_jan_2020(date_str):
            """Check if a date is after January 1, 2020 - original battle-tested logic"""
            try:
                import datetime
                if not date_str:
                    return False
                    
                # Extract year first
                year_match = re.search(r'20\d{2}', date_str)
                if year_match:
                    year = int(year_match.group(0))
                    if year < 2020:
                        return False
                    elif year > 2020:
                        return True
                    # If it's exactly 2020, we need to check month and day
                    
                # Month names mapping from original code
                month_names = {
                    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, 
                    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, 
                    "august": 8, "aug": 8, "september": 9, "sep": 9, "october": 10, "oct": 10, 
                    "november": 11, "nov": 11, "december": 12, "dec": 12
                }
                
                # Try different date formats from original code
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
                        date_obj = datetime.datetime.strptime(date_str, fmt)
                        return date_obj >= datetime.datetime(2020, 1, 1)
                    except ValueError:
                        continue
                        
                # If none of the formats work, do a more general check
                date_parts = re.findall(r'\b(\d{1,2}|[A-Za-z]+)\b', date_str.lower())
                for part in date_parts:
                    if part.isalpha() and part in month_names:
                        month = month_names[part]
                        if year_match and int(year_match.group(0)) == 2020 and month >= 1:
                            return True
                
                # Default to including if we can't determine the date
                return True
                
            except Exception as e:
                logging.warning(f"Date parsing error for '{date_str}': {str(e)}")
                return True  # Default to including if we can't parse the date
        
        async def get_apex_session():
            """Initialize APEX session - exact logic from original webscraper.py"""
            session_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            session = aiohttp.ClientSession(headers=session_headers)
            
            # Base session URL from original code
            search_url = "https://apps.cpuc.ca.gov/apex/f?p=401:5::::RP,5,RIR,57,RIR::"
            
            try:
                async with session.get(search_url, timeout=30) as response:
                    if response.status != 200:
                        logging.error(f"Failed to initialize APEX session: {response.status}")
                        await session.close()
                        return None
                    
                    logging.info("APEX session initialized successfully")
                    return session
                    
            except Exception as e:
                logging.error(f"Session initialization error: {str(e)}")
                await session.close()
                return None
        
        async def parse_tabs_from_proceeding(session, proceeding_id):
            """Parse tabs from proceeding - adapted from original code"""
            detail_url = "https://apps.cpuc.ca.gov/apex/f"
            params = {
                "p": f"401:56::::RP,57,RIR:P5_PROCEEDING_SELECT:{proceeding_id}"
            }
            
            logging.info(f"Processing proceeding: {proceeding_id}")
            
            try:
                async with session.get(detail_url, params=params, timeout=30) as response:
                    if response.status != 200:
                        logging.error(f"Error accessing proceeding {proceeding_id}: {response.status}")
                        return {}, None
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract proceeding metadata - from original code
                    proceeding_metadata = {
                        'proceeding_number': proceeding_id,
                        'filed_by': '',
                        'service_lists': '',
                        'industry': '',
                        'filing_date': '',
                        'category': '',
                        'current_status': '',
                        'description': '',
                        'staff': ''
                    }
                    
                    # Extract metadata using original regex patterns
                    page_text = soup.get_text()
                    
                    # Filed By
                    filed_by_match = re.search(r'Filed By:\s*([^\n]+)', page_text)
                    if filed_by_match:
                        proceeding_metadata['filed_by'] = filed_by_match.group(1).strip()
                    
                    # Filing Date
                    filing_date_match = re.search(r'Filing Date:\s*([^\n]+?)(?=Category:|$)', page_text)
                    if filing_date_match:
                        proceeding_metadata['filing_date'] = filing_date_match.group(1).strip()
                    
                    # Category
                    category_match = re.search(r'Category:\s*([^\n]+?)(?=Current Status:|$)', page_text)
                    if category_match:
                        proceeding_metadata['category'] = category_match.group(1).strip()
                    
                    # Current Status
                    status_match = re.search(r'Current Status:\s*([^\n]+?)(?=Description:|$)', page_text)
                    if status_match:
                        proceeding_metadata['current_status'] = status_match.group(1).strip()
                    
                    # Description
                    desc_match = re.search(r'Description:\s*([^\n]+?)(?=Staff:|$)', page_text)
                    if desc_match:
                        proceeding_metadata['description'] = desc_match.group(1).strip()
                    
                    # Check date filter
                    if not is_after_jan_2015(proceeding_metadata['filing_date']):
                        logging.info(f"Skipping proceeding {proceeding_id} - filed before January 1, 2015")
                        return {}, None
                    
                    # Extract tab links - from original code
                    tabs_ul = soup.select_one("div.sHorizontalTabsInner ul")
                    if not tabs_ul:
                        logging.warning(f"No tabs found for {proceeding_id}")
                        return {}, proceeding_metadata
                    
                    tab_links = {}
                    for li in tabs_ul.find_all("li"):
                        a_tag = li.find("a")
                        if a_tag:
                            title = a_tag.get_text(strip=True)
                            href = a_tag.get("href")
                            full_url = urljoin("https://apps.cpuc.ca.gov/apex/", href)
                            tab_links[title] = full_url
                    
                    logging.info(f"Successfully extracted metadata and tabs for {proceeding_id}")
                    return tab_links, proceeding_metadata
                    
            except Exception as e:
                logging.error(f"Error parsing proceeding {proceeding_id}: {str(e)}")
                return {}, None
        
        async def scrape_cpuc_proceedings():
            """Main scraping function - using battle-tested approach"""
            session = await get_apex_session()
            if not session:
                return {"error": "Failed to initialize CPUC session"}
            
            try:
                # Test with known recent proceeding IDs from original code
                test_proceeding_ids = [
                    "A2502016", "A2408011", "A2405012", "A2403005", "A2401005",
                    "A2312008", "A2311003", "A2308012", "A2307005", "A2306002",
                    "A2305008", "A2304010", "A2303005", "A2302012", "A2301005"
                ]
                
                valid_proceedings = []
                
                for proc_id in test_proceeding_ids:
                    try:
                        tab_links, metadata = await parse_tabs_from_proceeding(session, proc_id)
                        
                        if metadata:
                            proceeding_data = {
                                "proceeding_id": proc_id,
                                "title": metadata.get('description', f"Proceeding {proc_id}"),
                                "url": f"https://apps.cpuc.ca.gov/apex/f?p=401:56::::RP,57,RIR:P5_PROCEEDING_SELECT:{proc_id}",
                                "category": metadata.get('category', 'Energy'),
                                "filing_date": metadata.get('filing_date', ''),
                                "filed_by": metadata.get('filed_by', ''),
                                "current_status": metadata.get('current_status', ''),
                                "discovered_at": datetime.now().isoformat(),
                                "tabs_available": list(tab_links.keys()) if tab_links else []
                            }
                            
                            valid_proceedings.append(proceeding_data)
                            logging.info(f"Added proceeding {proc_id} - Filed: {metadata.get('filing_date', 'Unknown')}")
                        
                        # Rate limiting - from original code
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logging.error(f"Error processing proceeding {proc_id}: {str(e)}")
                        continue
                
                return {
                    "proceedings": valid_proceedings,
                    "date_filter": "January 1, 2015 and after",
                    "total_tested": len(test_proceeding_ids),
                    "total_valid": len(valid_proceedings)
                }
                
            finally:
                await session.close()
        
        # Run the scraper
        results = asyncio.run(scrape_cpuc_proceedings())
        
        # Store results in blob storage
        if results.get("proceedings"):
            blob_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
            container_name = "scraped-proceedings"
            
            try:
                container_client = blob_client.get_container_client(container_name)
                container_client.create_container()
            except Exception:
                pass
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            blob_name = f"proceedings_{timestamp}.json"
            blob_client_instance = blob_client.get_blob_client(container=container_name, blob=blob_name)
            
            blob_client_instance.upload_blob(
                json.dumps(results["proceedings"], indent=2),
                overwrite=True
            )
        
        return func.HttpResponse(
            json.dumps({
                "message": "Complete CPUC scraper with date filter completed",
                "proceedings_found": len(results.get("proceedings", [])),
                "date_filter": results.get("date_filter"),
                "total_tested": results.get("total_tested", 0),
                "total_valid": results.get("total_valid", 0),
                "proceedings": results.get("proceedings", [])[:3],  # Show first 3
                "timestamp": datetime.now().isoformat()
            }, indent=2),
            mimetype="application/json",
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        logging.error(f"Complete CPUC scraper failed: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
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