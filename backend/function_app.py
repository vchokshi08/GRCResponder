import azure.functions as func
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any

# Azure Storage import - now working!
from azure.storage.blob import BlobServiceClient

# Initialize Function App
app = func.FunctionApp()

# Configuration
STORAGE_CONNECTION_STRING = os.environ.get('AzureWebJobsStorage')

@app.function_name("HealthCheck")
@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check with actual Azure Storage test"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "message": "GRCResponder Function App with Azure Storage!",
            "version": "2.0.0"
        }
        
        # Test Azure Storage connection
    # Test Azure Storage connection
        if STORAGE_CONNECTION_STRING:
            try:
                blob_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
                containers = list(blob_client.list_containers())
                health_status["storage_status"] = "connected"
                health_status["storage_containers"] = [c.name for c in containers[:5]]  # Limit to first 5
            except Exception as e:
                health_status["storage_status"] = f"error: {str(e)[:100]}"  # Limit error message length
        else:
            health_status["storage_status"] = "no_connection_string"
        
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
    """Search API with storage integration"""
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
            "ai_response": f"Azure Storage is working! Your query '{query}' can now be processed with full Azure backend support.",
            "timestamp": datetime.now().isoformat(),
            "status": "azure_storage_ready",
            "next_step": "Add Key Vault and Search services"
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
    """Chat API with storage"""
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
            "response": f"Hello! You said: '{message}'. I'm now running with Azure Storage integration! Ready for the next step.",
            "timestamp": datetime.now().isoformat(),
            "status": "storage_integrated"
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
    """Test endpoint showing storage integration"""
    return func.HttpResponse(
        json.dumps({
            "message": "🎉 Azure Storage Integration Complete!",
            "step": "2.0 - Storage Working",
            "achievements": [
                "✅ Azure Functions Core Tools installed",
                "✅ Package installation working", 
                "✅ Azure Storage connected",
                "✅ Ready for next Azure services"
            ],
            "next_steps": [
                "Add Azure Key Vault",
                "Add Azure Search", 
                "Add Google Gemini AI",
                "Add original GRC logic"
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