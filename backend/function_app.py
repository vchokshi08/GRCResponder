import azure.functions as func
import azure.durable_functions as df
import logging
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import hashlib
import re
from urllib.parse import urljoin, urlparse, quote
import PyPDF2
import io
import time
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import google.generativeai as genai
import numpy as np
from dataclasses import dataclass, asdict
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Initialize Function App
app = func.FunctionApp()

# Configuration
STORAGE_CONNECTION_STRING = os.environ.get('AzureWebJobsStorage')
KEY_VAULT_URL = os.environ.get('KEY_VAULT_URL', 'https://grcresponder-dev-kv.vault.azure.net/')
SEARCH_SERVICE_ENDPOINT = os.environ.get('SEARCH_SERVICE_ENDPOINT', 'https://grcresponder-dev-search.search.windows.net')
SEARCH_API_KEY = os.environ.get('SEARCH_API_KEY')
GEMINI_MODEL_NAME = 'gemini-2.0-flash-exp'

# Global Constants - Based on Original GRCResponder Patterns
CPUC_BASE_URL = "https://apps.cpuc.ca.gov"
PROCEEDINGS_ENDPOINTS = [
    "/apex/f?p=401:1:0",    # General proceedings
    "/apex/f?p=401:56:0",   # Energy proceedings  
    "/apex/f?p=401:57:0"    # Rulemaking proceedings
]

# Rate limiting configuration (from original code patterns)
REQUEST_DELAY = 2.0  # seconds between requests to avoid CPUC blocking
MAX_CONCURRENT_DOWNLOADS = 5
CHUNK_SIZE = 1000  # characters for text chunking
CHUNK_OVERLAP = 200

# Data Models - Mirror original structure
@dataclass
class CPUCProceeding:
    proceeding_id: str
    title: str
    url: str
    status: str
    date_filed: str
    last_updated: str
    category: str
    summary: str = ""
    documents: List[Dict] = None
    
    def __post_init__(self):
        if self.documents is None:
            self.documents = []

@dataclass
class ProcessedDocument:
    document_id: str
    proceeding_id: str
    filename: str
    blob_url: str
    text_content: str
    file_size: int
    processed_timestamp: str
    metadata: Dict
    chunks: List[str] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []

# Utility Classes
class AzureServiceManager:
    """Azure services initialization and management"""
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
            
            # Configure Gemini Flash 2.0 - following original patterns
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = genai.GenerativeModel(
                model_name=GEMINI_MODEL_NAME,
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
            self.search_client = SearchClient(
                endpoint=SEARCH_SERVICE_ENDPOINT,
                index_name="grc-documents",
                credential=AzureKeyCredential(SEARCH_API_KEY)
            )
            
            self._initialized = True
            logging.info("Azure services initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Azure services: {str(e)}")
            raise

class CPUCScraperService:
    """CPUC proceeding scraper - based on original grc_tools logic"""
    
    def __init__(self, service_manager: AzureServiceManager):
        self.service_manager = service_manager
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_all_proceedings(self) -> List[CPUCProceeding]:
        """Main scraper method - replicates original proceeding discovery logic"""
        all_proceedings = []
        
        for endpoint in PROCEEDINGS_ENDPOINTS:
            try:
                category = self._extract_category_from_endpoint(endpoint)
                logging.info(f"Scraping {category} proceedings from {endpoint}")
                
                proceedings = await self._scrape_proceedings_page(endpoint, category)
                all_proceedings.extend(proceedings)
                
                # Rate limiting between endpoint requests
                await asyncio.sleep(REQUEST_DELAY)
                
            except Exception as e:
                logging.error(f"Failed to scrape endpoint {endpoint}: {str(e)}")
                continue
        
        logging.info(f"Total proceedings discovered: {len(all_proceedings)}")
        return all_proceedings
    
    def _extract_category_from_endpoint(self, endpoint: str) -> str:
        """Extract proceeding category from CPUC endpoint"""
        if "f?p=401:56" in endpoint:
            return "Energy"
        elif "f?p=401:57" in endpoint:
            return "Rulemaking"
        else:
            return "General"
    
    async def _scrape_proceedings_page(self, endpoint: str, category: str) -> List[CPUCProceeding]:
        """Scrape proceedings from a specific CPUC page"""
        proceedings = []
        url = CPUC_BASE_URL + endpoint
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logging.warning(f"HTTP {response.status} for {url}")
                    return proceedings
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse proceeding entries - adapted from original selectors
                proceeding_elements = self._find_proceeding_elements(soup)
                
                for element in proceeding_elements:
                    try:
                        proceeding = self._parse_proceeding_element(element, category)
                        if proceeding:
                            proceedings.append(proceeding)
                    except Exception as e:
                        logging.warning(f"Failed to parse proceeding element: {str(e)}")
                        continue
                
        except Exception as e:
            logging.error(f"Failed to scrape proceedings page {url}: {str(e)}")
        
        return proceedings
    
    def _find_proceeding_elements(self, soup: BeautifulSoup) -> List:
        """Find proceeding elements - based on original CPUC HTML structure patterns"""
        # Try multiple selectors based on CPUC's changing HTML structure
        selectors = [
            'tr.highlight-row',
            'tr[class*="proceeding"]',
            'tr td:first-child a[href*="proceeding"]',
            '.proceeding-item',
            'table tr:has(td a[href*="A."])',  # Application numbers
            'table tr:has(td a[href*="R."])',  # Rulemaking numbers
        ]
        
        elements = []
        for selector in selectors:
            try:
                found = soup.select(selector)
                if found:
                    elements.extend(found)
                    break
            except Exception:
                continue
        
        # Fallback: find table rows with proceeding-like patterns
        if not elements:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    if self._is_proceeding_row(row):
                        elements.append(row)
        
        return elements
    
    def _is_proceeding_row(self, row) -> bool:
        """Check if table row contains proceeding data"""
        text = row.get_text().strip()
        # Look for proceeding ID patterns (A.XX-XX-XXX or R.XX-XX-XXX)
        proceeding_pattern = r'[AR]\.\d{2}-\d{2}-\d{3}'
        return bool(re.search(proceeding_pattern, text))
    
    def _parse_proceeding_element(self, element, category: str) -> Optional[CPUCProceeding]:
        """Parse individual proceeding element - based on original patterns"""
        try:
            # Extract proceeding ID
            proceeding_id = self._extract_proceeding_id(element)
            if not proceeding_id:
                return None
            
            # Extract title and URL
            title_link = element.find('a', href=True)
            if not title_link:
                return None
            
            title = title_link.get_text().strip()
            proceeding_url = urljoin(CPUC_BASE_URL, title_link['href'])
            
            # Extract status and dates
            cells = element.find_all('td')
            status = self._extract_text_from_cells(cells, 'status', 'Active')
            date_filed = self._extract_text_from_cells(cells, 'date', '')
            
            proceeding = CPUCProceeding(
                proceeding_id=proceeding_id,
                title=title,
                url=proceeding_url,
                status=status,
                date_filed=date_filed,
                last_updated=datetime.now().isoformat(),
                category=category,
                summary=title[:200] + "..." if len(title) > 200 else title
            )
            
            return proceeding
            
        except Exception as e:
            logging.warning(f"Failed to parse proceeding element: {str(e)}")
            return None
    
    def _extract_proceeding_id(self, element) -> Optional[str]:
        """Extract proceeding ID from element"""
        text = element.get_text()
        # Look for standard CPUC proceeding ID patterns
        patterns = [
            r'[AR]\.\d{2}-\d{2}-\d{3}',
            r'[AR]\d{4}\d{2}\d{3}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def _extract_text_from_cells(self, cells: List, field_type: str, default: str) -> str:
        """Extract specific field from table cells"""
        if len(cells) < 2:
            return default
        
        # Heuristic based on cell position and content
        for i, cell in enumerate(cells):
            cell_text = cell.get_text().strip()
            
            if field_type == 'status' and i > 1:
                if any(status in cell_text.lower() for status in ['active', 'closed', 'pending']):
                    return cell_text
            elif field_type == 'date' and i > 0:
                # Look for date patterns
                if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', cell_text):
                    return cell_text
        
        return default

# Global service manager instance
service_manager = AzureServiceManager()

# Azure Function 1: Proceeding Scraper
@app.function_name("ProceedingScraper")
@app.schedule(schedule="0 0 6 * * *", arg_name="timer", run_on_startup=False)
async def proceeding_scraper(timer: func.TimerRequest) -> None:
    """
    CPUC Proceeding Scraper - Part 1 of 3-step pipeline
    Runs daily at 6 AM to discover new and updated proceedings
    """
    logging.info("Starting CPUC Proceeding Scraper - Step 1/3")
    
    try:
        await service_manager.initialize()
        
        # Scrape all proceedings using original patterns
        async with CPUCScraperService(service_manager) as scraper:
            proceedings = await scraper.scrape_all_proceedings()
        
        if not proceedings:
            logging.warning("No proceedings found during scraping")
            return
        
        # Store proceedings metadata in blob storage
        await store_proceedings_metadata(proceedings)
        
        # Create download queue for Document Downloader
        await create_download_queue(proceedings)
        
        logging.info(f"Proceeding scraper completed successfully. Found {len(proceedings)} proceedings")
        
    except Exception as e:
        logging.error(f"Proceeding scraper failed: {str(e)}")
        raise

async def store_proceedings_metadata(proceedings: List[CPUCProceeding]) -> None:
    """Store proceedings metadata in Azure Blob Storage"""
    try:
        container_name = "proceedings-metadata"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f"proceedings_{timestamp}.json"
        
        # Convert proceedings to JSON
        proceedings_data = [asdict(p) for p in proceedings]
        json_data = json.dumps(proceedings_data, indent=2, default=str)
        
        # Upload to blob storage
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(json_data, overwrite=True)
        logging.info(f"Stored proceedings metadata: {blob_name}")
        
    except Exception as e:
        logging.error(f"Failed to store proceedings metadata: {str(e)}")
        raise

async def create_download_queue(proceedings: List[CPUCProceeding]) -> None:
    """Create download queue for Document Downloader"""
    try:
        container_name = "processing-queue"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f"download_queue_{timestamp}.json"
        
        # Create queue items
        queue_data = []
        for proceeding in proceedings:
            queue_item = {
                "proceeding_id": proceeding.proceeding_id,
                "proceeding_url": proceeding.url,
                "title": proceeding.title,
                "category": proceeding.category,
                "status": proceeding.status,
                "queued_at": datetime.now().isoformat(),
                "retry_count": 0
            }
            queue_data.append(queue_item)
        
        # Store queue
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(json.dumps(queue_data, indent=2), overwrite=True)
        logging.info(f"Created download queue with {len(queue_data)} items: {blob_name}")
        
    except Exception as e:
        logging.error(f"Failed to create download queue: {str(e)}")
        raise

# Azure Function 2: Document Downloader
@app.function_name("DocumentDownloader")
@app.schedule(schedule="0 0 */2 * * *", arg_name="timer", run_on_startup=False)
async def document_downloader(timer: func.TimerRequest) -> None:
    """
    Document Downloader - Part 2 of 3-step pipeline
    Runs every 2 hours to download and process documents
    """
    logging.info("Starting Document Downloader - Step 2/3")
    
    try:
        await service_manager.initialize()
        
        # Get queued proceedings for download
        queue_items = await get_download_queue()
        
        if not queue_items:
            logging.info("No items in download queue")
            return
        
        # Process downloads with concurrency control (original pattern)
        processed_documents = await process_download_queue(queue_items)
        
        if processed_documents:
            # Create vector processing queue
            await create_vector_queue(processed_documents)
        
        logging.info(f"Document downloader completed. Processed {len(processed_documents)} documents")
        
    except Exception as e:
        logging.error(f"Document downloader failed: {str(e)}")
        raise

async def get_download_queue() -> List[Dict]:
    """Get items from download queue"""
    try:
        container_name = "processing-queue"
        container_client = service_manager.blob_client.get_container_client(container_name)
        queue_items = []
        
        # Process all download queue files
        async for blob in container_client.list_blobs():
            if blob.name.startswith("download_queue_"):
                blob_client = service_manager.blob_client.get_blob_client(
                    container=container_name,
                    blob=blob.name
                )
                
                download_stream = await blob_client.download_blob()
                queue_data = json.loads(await download_stream.readall())
                queue_items.extend(queue_data)
                
                # Delete processed queue file
                await blob_client.delete_blob()
        
        return queue_items
        
    except Exception as e:
        logging.error(f"Failed to get download queue: {str(e)}")
        return []

async def process_download_queue(queue_items: List[Dict]) -> List[ProcessedDocument]:
    """Process download queue with original concurrency patterns"""
    processed_documents = []
    
    # Process in batches to avoid overwhelming CPUC servers
    batch_size = MAX_CONCURRENT_DOWNLOADS
    
    for i in range(0, len(queue_items), batch_size):
        batch = queue_items[i:i+batch_size]
        
        # Process batch concurrently
        tasks = [process_single_proceeding(item) for item in batch]
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

async def process_single_proceeding(queue_item: Dict) -> List[ProcessedDocument]:
    """Process documents from a single proceeding"""
    documents = []
    
    try:
        proceeding_url = queue_item["proceeding_url"]
        proceeding_id = queue_item["proceeding_id"]
        
        # Download proceeding page and find document links
        async with aiohttp.ClientSession() as session:
            document_urls = await discover_proceeding_documents(session, proceeding_url)
            
            # Download each document
            for doc_url, doc_filename in document_urls:
                try:
                    # Rate limiting between document downloads
                    await asyncio.sleep(REQUEST_DELAY)
                    
                    # Download PDF content
                    pdf_content = await download_pdf_document(session, doc_url)
                    if not pdf_content:
                        continue
                    
                    # Extract text content
                    text_content = extract_text_from_pdf(pdf_content)
                    if not text_content or len(text_content.strip()) < 100:
                        continue
                    
                    # Store PDF in blob storage
                    blob_url = await store_document_blob(pdf_content, proceeding_id, doc_filename)
                    
                    # Create processed document
                    document = ProcessedDocument(
                        document_id=str(uuid.uuid4()),
                        proceeding_id=proceeding_id,
                        filename=doc_filename,
                        blob_url=blob_url,
                        text_content=text_content,
                        file_size=len(pdf_content),
                        processed_timestamp=datetime.now().isoformat(),
                        metadata={
                            "source_url": doc_url,
                            "proceeding_title": queue_item.get("title", ""),
                            "category": queue_item.get("category", ""),
                            "text_length": len(text_content),
                            "status": queue_item.get("status", "")
                        }
                    )
                    
                    documents.append(document)
                    
                except Exception as e:
                    logging.warning(f"Failed to process document {doc_url}: {str(e)}")
                    continue
        
    except Exception as e:
        logging.error(f"Failed to process proceeding {queue_item.get('proceeding_id')}: {str(e)}")
    
    return documents

async def discover_proceeding_documents(session: aiohttp.ClientSession, proceeding_url: str) -> List[tuple]:
    """Discover document links from proceeding page"""
    document_urls = []
    
    try:
        async with session.get(proceeding_url) as response:
            if response.status != 200:
                return document_urls
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find PDF document links - based on original patterns
            pdf_patterns = [
                'a[href$=".pdf"]',
                'a[href*=".pdf"]',
                'a[href*="document"]',
                'a[href*="filing"]'
            ]
            
            for pattern in pdf_patterns:
                links = soup.select(pattern)
                for link in links:
                    href = link.get('href')
                    if href and href.lower().endswith('.pdf'):
                        full_url = urljoin(proceeding_url, href)
                        filename = link.get_text().strip() or href.split('/')[-1]
                        document_urls.append((full_url, filename))
        
    except Exception as e:
        logging.warning(f"Failed to discover documents from {proceeding_url}: {str(e)}")
    
    return document_urls

async def download_pdf_document(session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
    """Download PDF document content"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            if response.status == 200 and response.content_type == 'application/pdf':
                content = await response.read()
                if len(content) > 1000:  # Minimum size check
                    return content
    except Exception as e:
        logging.warning(f"Failed to download PDF from {url}: {str(e)}")
    
    return None

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF - based on original extraction logic"""
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
        
        # Clean and normalize text
        text_content = re.sub(r'\s+', ' ', text_content)
        text_content = text_content.strip()
        
        return text_content
        
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {str(e)}")
        return ""

async def store_document_blob(pdf_content: bytes, proceeding_id: str, filename: str) -> str:
    """Store PDF document in Azure Blob Storage"""
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

async def create_vector_queue(documents: List[ProcessedDocument]) -> None:
    """Create queue for vector processing"""
    try:
        container_name = "processing-queue"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f"vector_queue_{timestamp}.json"
        
        # Prepare queue data
        queue_data = []
        for doc in documents:
            queue_item = {
                "document_id": doc.document_id,
                "proceeding_id": doc.proceeding_id,
                "filename": doc.filename,
                "blob_url": doc.blob_url,
                "text_content": doc.text_content,
                "file_size": doc.file_size,
                "metadata": doc.metadata,
                "queued_at": datetime.now().isoformat()
            }
            queue_data.append(queue_item)
        
        # Store queue
        blob_client = service_manager.blob_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        await blob_client.upload_blob(json.dumps(queue_data, indent=2), overwrite=True)
        logging.info(f"Created vector queue with {len(documents)} items: {blob_name}")
        
    except Exception as e:
        logging.error(f"Failed to create vector queue: {str(e)}")
        raise

# Azure Function 3: Vector Database Builder
@app.function_name("VectorDatabaseBuilder")
@app.schedule(schedule="0 0 */4 * * *", arg_name="timer", run_on_startup=False)
async def vector_database_builder(timer: func.TimerRequest) -> None:
    """
    Vector Database Builder - Part 3 of 3-step pipeline
    Runs every 4 hours to build vector embeddings using Gemini Flash 2.0
    """
    logging.info("Starting Vector Database Builder - Step 3/3")
    
    try:
        await service_manager.initialize()
        
        # Get queued documents for vector processing
        queue_items = await get_vector_queue()
        
        if not queue_items:
            logging.info("No items in vector processing queue")
            return
        
        # Process documents with multithreading (original pattern)
        await process_vector_queue_multithreaded(queue_items)
        
        logging.info(f"Vector database builder completed. Processed {len(queue_items)} documents")
        
    except Exception as e:
        logging.error(f"Vector database builder failed: {str(e)}")
        raise

async def get_vector_queue() -> List[Dict]:
    """Get items from vector processing queue"""
    try:
        container_name = "processing-queue"
        container_client = service_manager.blob_client.get_container_client(container_name)
        queue_items = []
        
        # Process all vector queue files
        async for blob in container_client.list_blobs():
            if blob.name.startswith("vector_queue_"):
                blob_client = service_manager.blob_client.get_blob_client(
                    container=container_name,
                    blob=blob.name
                )
                
                download_stream = await blob_client.download_blob()
                queue_data = json.loads(await download_stream.readall())
                queue_items.extend(queue_data)
                
                # Delete processed queue file
                await blob_client.delete_blob()
        
        return queue_items
        
    except Exception as e:
        logging.error(f"Failed to get vector queue: {str(e)}")
        return []

async def process_vector_queue_multithreaded(queue_items: List[Dict]) -> None:
    """Process vector queue with multithreading - based on original multithreaded_insert.py patterns"""
    
    # Process in batches for better memory management
    batch_size = 10
    
    for i in range(0, len(queue_items), batch_size):
        batch = queue_items[i:i+batch_size]
        
        # Process each document in the batch
        search_documents = []
        for doc_data in batch:
            try:
                # Create text chunks - based on original chunking strategy
                chunks = create_text_chunks(doc_data["text_content"])
                
                # Generate embeddings for each chunk using Gemini Flash 2.0
                for i, chunk in enumerate(chunks):
                    chunk_embeddings = await generate_embeddings_with_retry(chunk)
                    
                    if chunk_embeddings:
                        search_doc = {
                            "id": f"{doc_data['document_id']}_chunk_{i}",
                            "document_id": doc_data["document_id"],
                            "proceeding_id": doc_data["proceeding_id"],
                            "filename": doc_data["filename"],
                            "blob_url": doc_data["blob_url"],
                            "content": chunk,
                            "content_vector": chunk_embeddings,
                            "metadata": json.dumps(doc_data["metadata"]),
                            "file_size": doc_data["file_size"],
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "processed_at": datetime.now().isoformat()
                        }
                        search_documents.append(search_doc)
                
            except Exception as e:
                logging.error(f"Failed to process document {doc_data.get('document_id')}: {str(e)}")
                continue
        
        # Index batch of documents in Azure AI Search
        if search_documents:
            await index_documents_batch(search_documents)

def create_text_chunks(text: str) -> List[str]:
    """Create overlapping text chunks - based on original chunking logic"""
    if not text or len(text.strip()) < 100:
        return []
    
    chunks = []
    words = text.split()
    
    # Calculate chunk size in words
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

async def generate_embeddings_with_retry(text: str, max_retries: int = 3) -> List[float]:
    """Generate embeddings with retry logic for robustness"""
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

async def index_documents_batch(documents: List[Dict]) -> None:
    """Index batch of documents in Azure AI Search"""
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

# Azure Function: Search API with RAG
@app.function_name("SearchAPI")
@app.route(route="search", methods=["GET", "POST"])
async def search_api(req: func.HttpRequest) -> func.HttpResponse:
    """
    RAG-powered search API using Gemini Flash 2.0
    Provides intelligent search across CPUC regulatory documents
    """
    logging.info("Processing RAG search request")
    
    try:
        await service_manager.initialize()
        
        # Parse request parameters
        if req.method == "GET":
            query = req.params.get('q', '').strip()
            top_k = int(req.params.get('top_k', '10'))
        else:
            req_body = req.get_json()
            query = req_body.get('query', '').strip() if req_body else ''
            top_k = int(req_body.get('top_k', 10)) if req_body else 10
        
        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Search query parameter 'q' is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Perform RAG search
        search_results = await perform_rag_search(query, top_k)
        
        # Generate AI response using Gemini Flash 2.0
        ai_response = await generate_rag_response(query, search_results)
        
        # Format response
        response_data = {
            "query": query,
            "ai_response": ai_response,
            "source_documents": search_results,
            "total_results": len(search_results),
            "timestamp": datetime.now().isoformat(),
            "model": GEMINI_MODEL_NAME
        }
        
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Search API failed: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error", 
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

async def perform_rag_search(query: str, top_k: int = 10) -> List[Dict]:
    """Perform RAG search using vector similarity and keyword search"""
    try:
        # Generate query embedding
        query_embedding = await generate_embeddings_with_retry(query)
        
        if not query_embedding:
            # Fallback to keyword search only
            return await perform_keyword_search(query, top_k)
        
        # Perform hybrid search (vector + keyword)
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        search_results = await service_manager.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=[
                "document_id", "proceeding_id", "filename", "content", 
                "metadata", "blob_url", "chunk_index", "total_chunks"
            ],
            top=top_k,
            query_type="semantic"
        )
        
        # Format results
        results = []
        async for result in search_results:
            metadata = json.loads(result.get("metadata", "{}"))
            
            formatted_result = {
                "document_id": result["document_id"],
                "proceeding_id": result["proceeding_id"],
                "filename": result["filename"],
                "content": result["content"],
                "blob_url": result["blob_url"],
                "relevance_score": result.get("@search.score", 0),
                "chunk_info": {
                    "chunk_index": result.get("chunk_index", 0),
                    "total_chunks": result.get("total_chunks", 1)
                },
                "metadata": metadata
            }
            results.append(formatted_result)
        
        return results
        
    except Exception as e:
        logging.error(f"RAG search failed: {str(e)}")
        return []

async def perform_keyword_search(query: str, top_k: int) -> List[Dict]:
    """Fallback keyword search when vector search fails"""
    try:
        search_results = await service_manager.search_client.search(
            search_text=query,
            select=[
                "document_id", "proceeding_id", "filename", "content", 
                "metadata", "blob_url", "chunk_index", "total_chunks"
            ],
            top=top_k,
            query_type="simple"
        )
        
        results = []
        async for result in search_results:
            metadata = json.loads(result.get("metadata", "{}"))
            
            formatted_result = {
                "document_id": result["document_id"],
                "proceeding_id": result["proceeding_id"],
                "filename": result["filename"],
                "content": result["content"],
                "blob_url": result["blob_url"],
                "relevance_score": result.get("@search.score", 0),
                "chunk_info": {
                    "chunk_index": result.get("chunk_index", 0),
                    "total_chunks": result.get("total_chunks", 1)
                },
                "metadata": metadata
            }
            results.append(formatted_result)
        
        return results
        
    except Exception as e:
        logging.error(f"Keyword search failed: {str(e)}")
        return []

async def generate_rag_response(query: str, search_results: List[Dict]) -> str:
    """Generate RAG response using Gemini Flash 2.0 with context"""
    try:
        if not search_results:
            return "I couldn't find any relevant documents to answer your question. Please try rephrasing your query or using different keywords."
        
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results[:5]):  # Use top 5 results
            context_parts.append(
                f"Document {i+1}: {result['filename']}\n"
                f"Proceeding: {result['proceeding_id']}\n"
                f"Content: {result['content'][:800]}...\n"
                f"---"
            )
        
        context = "\n\n".join(context_parts)
        
        # Create RAG prompt
        rag_prompt = f"""
You are an expert assistant for CPUC (California Public Utilities Commission) regulatory document analysis. 
Based on the following regulatory documents, provide a comprehensive, accurate answer to the user's question.

User Question: {query}

Relevant Document Excerpts:
{context}

Instructions:
1. Provide a detailed, professional response that directly answers the question
2. Reference specific documents and proceeding IDs when relevant
3. Explain regulatory implications and context where appropriate
4. If the information is insufficient, clearly state what additional details would be needed
5. Maintain accuracy and cite specific sections when possible
6. Use professional language appropriate for regulatory analysis

Response:
"""
        
        # Generate response using Gemini Flash 2.0
        response = await service_manager.gemini_client.generate_content(rag_prompt)
        
        return response.text
        
    except Exception as e:
        logging.error(f"RAG response generation failed: {str(e)}")
        return "I apologize, but I'm unable to generate a response at this time. Please try again later or contact system administrator."

# Azure Function: Health Check
@app.function_name("HealthCheck")
@app.route(route="health", methods=["GET"])
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint for monitoring system status"""
    try:
        await service_manager.initialize()
        
        # Test connections to all services
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "pipeline_status": {}
        }
        
        # Test Blob Storage
        try:
            container_client = service_manager.blob_client.get_container_client("proceedings-metadata")
            await container_client.get_container_properties()
            health_status["services"]["blob_storage"] = "healthy"
        except Exception as e:
            health_status["services"]["blob_storage"] = f"unhealthy: {str(e)}"
        
        # Test Search Service
        try:
            index_client = SearchIndexClient(
                endpoint=SEARCH_SERVICE_ENDPOINT,
                credential=AzureKeyCredential(SEARCH_API_KEY)
            )
            await index_client.get_index("grc-documents")
            health_status["services"]["search_service"] = "healthy"
        except Exception as e:
            health_status["services"]["search_service"] = f"unhealthy: {str(e)}"
        
        # Test Gemini API
        try:
            test_embedding = await generate_embeddings_with_retry("test", max_retries=1)
            if test_embedding:
                health_status["services"]["gemini_api"] = "healthy"
            else:
                health_status["services"]["gemini_api"] = "unhealthy: no embedding returned"
        except Exception as e:
            health_status["services"]["gemini_api"] = f"unhealthy: {str(e)}"
        
        # Check pipeline status
        try:
            # Check recent proceedings data
            container_client = service_manager.blob_client.get_container_client("proceedings-metadata")
            blobs = []
            async for blob in container_client.list_blobs():
                if blob.name.startswith("proceedings_"):
                    blobs.append(blob)
            
            if blobs:
                latest_blob = max(blobs, key=lambda b: b.last_modified)
                hours_since_update = (datetime.now(latest_blob.last_modified.tzinfo) - latest_blob.last_modified).total_seconds() / 3600
                
                if hours_since_update < 48:  # Updated within 2 days
                    health_status["pipeline_status"]["proceeding_scraper"] = "healthy"
                else:
                    health_status["pipeline_status"]["proceeding_scraper"] = f"stale: {hours_since_update:.1f} hours old"
            else:
                health_status["pipeline_status"]["proceeding_scraper"] = "no data found"
                
        except Exception as e:
            health_status["pipeline_status"]["proceeding_scraper"] = f"error: {str(e)}"
        
        # Determine overall status
        service_issues = [v for v in health_status["services"].values() if not v == "healthy"]
        pipeline_issues = [v for v in health_status["pipeline_status"].values() if not v == "healthy"]
        
        if service_issues or pipeline_issues:
            health_status["status"] = "degraded"
            health_status["issues"] = service_issues + pipeline_issues
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        
        return func.HttpResponse(
            json.dumps(health_status, indent=2),
            status_code=status_code,
            mimetype="application/json"
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
            mimetype="application/json"
        )

# Azure Function: Manual Trigger for Development
@app.function_name("ManualTrigger")
@app.route(route="trigger/{step}", methods=["POST"])
async def manual_trigger(req: func.HttpRequest) -> func.HttpResponse:
    """Manual trigger for pipeline steps during development and testing"""
    try:
        step = req.route_params.get('step')
        
        if step == "scraper":
            await proceeding_scraper(None)
            message = "Proceeding scraper triggered successfully"
        elif step == "downloader":
            await document_downloader(None)
            message = "Document downloader triggered successfully"
        elif step == "vectorizer":
            await vector_database_builder(None)
            message = "Vector database builder triggered successfully"
        else:
            return func.HttpResponse(
                json.dumps({"error": f"Unknown step: {step}. Use 'scraper', 'downloader', or 'vectorizer'"}),
                status_code=400,
                mimetype="application/json"
            )
        
        return func.HttpResponse(
            json.dumps({
                "message": message,
                "timestamp": datetime.now().isoformat()
            }),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Manual trigger failed: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Manual trigger failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )