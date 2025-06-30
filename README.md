# GRCResponder 🏛️⚡

**100% Azure Cloud Native Serverless Infrastructure for PG&E Regulatory Document Analysis with RAG and Gemini Flash 2.0**

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FYOUR_USERNAME%2FGRCResponder%2Fmain%2Finfrastructure%2Fgrcresponder-infrastructure.json)

## 🚀 Quick Start

### Prerequisites
- Azure subscription (Pay-as-you-go or higher)
- Azure CLI installed
- PowerShell 5.1 or higher
- Git

### One-Click Deployment
```powershell
# Clone the repository
git clone https://github.com/YOUR_USERNAME/GRCResponder.git
cd GRCResponder

# Run the deployment script
.\scripts\one-click-deploy.ps1
```

## 🏗️ Architecture

**100% Serverless Azure Cloud Native Solution:**

- **🔧 Azure Functions**: Python-based serverless compute for document processing
- **🌐 Static Web Apps**: React frontend with hourglass animation search interface  
- **🔍 AI Search**: Vector database for RAG (Retrieval Augmented Generation)
- **💾 Blob Storage**: Document storage and caching
- **🔐 Key Vault**: Secure API key management
- **📊 Application Insights**: Monitoring and analytics
- **🤖 Gemini Flash 2.0**: AI-powered document analysis

## 📋 Features

### Core Pipeline (3-Part Architecture)
1. **Proceeding Scraper** - Automated CPUC proceeding discovery
2. **Document Downloader** - Bulk document acquisition and processing  
3. **Vector Database Builder** - RAG pipeline with embeddings and search

### User Interface
- Clean React frontend with animated search interface
- Real-time document search and analysis
- Responsive design for desktop and mobile

### Enterprise Ready
- Infrastructure as Code (ARM templates)
- Secure secret management
- Scalable serverless architecture
- Cost-optimized pay-per-use model

## 🎯 Use Cases

**Perfect for PG&E and Other Utilities:**
- Regulatory compliance document analysis
- CPUC proceeding monitoring
- Legal document search and summarization
- Regulatory filing assistance
- Compliance audit preparation

## 📁 Project Structure

```
GRCResponder/
├── infrastructure/          # ARM templates and deployment scripts
├── backend/                # Azure Functions (Python)
├── frontend/               # React Static Web App
├── docs/                   # Documentation
└── scripts/                # Automation scripts
```

## 🔧 Deployment Options

### Option 1: One-Click Azure Deployment
Click the "Deploy to Azure" button above for instant deployment.

### Option 2: PowerShell Script
```powershell
.\scripts\one-click-deploy.ps1 -SubscriptionId "your-subscription-id"
```

### Option 3: Manual ARM Template
```powershell
az deployment group create \
  --resource-group rg-grcresponder-dev \
  --template-file infrastructure/grcresponder-infrastructure.json \
  --parameters @infrastructure/grcresponder-parameters.json
```

## 🔑 Configuration

### Required Parameters
- **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Azure Subscription**: Pay-as-you-go or higher tier
- **Resource Group**: Will be created if doesn't exist

### Optional Parameters
- **Environment**: dev, staging, prod (default: dev)
- **Location**: Azure region (default: West US 2)
- **Project Name**: Custom prefix for resources (default: grcresponder)

## 💰 Cost Estimation

**Typical Monthly Costs (Pay-as-you-go):**
- Azure Functions: $0-10 (consumption plan)
- Static Web Apps: Free tier
- AI Search: Free tier (15MB storage)
- Blob Storage: $1-5 (depending on document volume)
- Key Vault: $0.03 per 10,000 operations
- Application Insights: $2-5 (first 5GB free)

**Total: ~$3-25/month** depending on usage

## 🛠️ Development

### Local Development Setup
```powershell
# Clone and setup
git clone https://github.com/YOUR_USERNAME/GRCResponder.git
cd GRCResponder

# Install dependencies
cd backend && pip install -r requirements.txt
cd ../frontend && npm install

# Start local development
func start  # Backend
npm start   # Frontend
```

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📚 Documentation

- [Deployment Guide](docs/deployment-guide.md)
- [Architecture Overview](docs/architecture-overview.md)
- [API Documentation](docs/api-documentation.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🏆 Why GRCResponder?

✅ **100% Serverless** - No servers to manage, infinite scaling  
✅ **Cost Optimized** - Pay only for what you use  
✅ **Enterprise Security** - Azure Key Vault integration  
✅ **Infrastructure as Code** - Reproducible deployments  
✅ **Modern Tech Stack** - React + Python + Azure + AI  
✅ **Regulatory Focused** - Built specifically for utility compliance  

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Support

- Create an [Issue](https://github.com/YOUR_USERNAME/GRCResponder/issues) for bugs
- Start a [Discussion](https://github.com/YOUR_USERNAME/GRCResponder/discussions) for questions
- Check [Documentation](docs/) for detailed guides

