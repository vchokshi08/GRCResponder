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
    """Search API with Key Vault integration"""
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
        
        response_data = {
            "query": query,
            "ai_response": f"Storage + Key Vault ready! Your query '{query}' can now access secure configuration.",
            "timestamp": datetime.now().isoformat(),
            "status": "keyvault_ready",
            "next_step": "Add Azure Search and Gemini AI"
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
    """Chat API with secure configuration"""
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
        if not message:
            return func.HttpResponse(
                json.dumps({"error": "Message is required"}),
                status_code=400,
                mimetype="application/json",
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        response_data = {
            "response": f"Hello! You said: '{message}'. I now have secure access to configuration via Key Vault!",
            "timestamp": datetime.now().isoformat(),
            "status": "keyvault_integrated"
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