# GRCResponder Backend - Azure Functions

This is the Azure Cloud Native Serverless backend for GRCResponder, implementing the proven 3-part pipeline for CPUC regulatory document analysis using Gemini Flash 2.0 and RAG.

## 🏗️ Architecture Overview

The backend preserves the original battle-tested GRCResponder logic while adapting it to Azure serverless components:

### Original 3-Part Pipeline → Azure Functions
1. **Proceeding Scraper** → `ProceedingScraper` Azure Function (Timer: Daily 6 AM)
2. **Document Downloader** → `DocumentDownloader` Azure Function (Timer: Every 2 hours)  
3. **Vector Database Builder** → `VectorDatabaseBuilder` Azure Function (Timer: Every 4 hours)

### Technology Stack
- **Azure Functions** (Python 3.11) - Serverless compute
- **Azure Blob Storage** - Document and metadata storage
- **Azure AI Search** - Vector database (replaces Qdrant for serverless)
- **Azure Key Vault** - Secure API key storage
- **Gemini Flash 2.0** - AI model for embeddings and RAG responses
- **Application Insights** - Monitoring and logging

## 📂 Project Structure