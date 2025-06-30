# GRCResponder Backend Deployment Script
# Deploy Azure Functions with battle-tested CPUC scraping logic

param(
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId = "d4d5edc0-d0d0-491c-8d55-4bf5481b5b49",
    
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName = "rg-grcresponder-dev",
    
    [Parameter(Mandatory=$true)]
    [string]$FunctionAppName = "grcresponder-dev-functions",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "West US 2",
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev"
)

Write-Host "Starting GRCResponder Backend Deployment" -ForegroundColor Green
Write-Host "Subscription: $SubscriptionId" -ForegroundColor Cyan
Write-Host "Resource Group: $ResourceGroupName" -ForegroundColor Cyan
Write-Host "Function App: $FunctionAppName" -ForegroundColor Cyan

# Set Azure context
Write-Host "Setting Azure context..." -ForegroundColor Yellow
az account set --subscription $SubscriptionId

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to set Azure subscription context"
    exit 1
}

# Verify resource group exists
Write-Host "Verifying resource group..." -ForegroundColor Yellow
$rgExists = az group exists --name $ResourceGroupName
if ($rgExists -eq "false") {
    Write-Error "Resource group $ResourceGroupName does not exist. Please deploy infrastructure first."
    exit 1
}

# Get Function App details
Write-Host "Getting Function App details..." -ForegroundColor Yellow
$functionApp = az functionapp show --name $FunctionAppName --resource-group $ResourceGroupName --query "{name:name, state:state, location:location}" -o json | ConvertFrom-Json

if (-not $functionApp) {
    Write-Error "Function App $FunctionAppName not found. Please deploy infrastructure first."
    exit 1
}

Write-Host "Function App found: $($functionApp.name) in $($functionApp.location)" -ForegroundColor Green

# Configure Function App settings
Write-Host "Configuring Function App settings..." -ForegroundColor Yellow

$keyVaultUrl = "https://grcresponder-dev-kv.vault.azure.net/"
$searchEndpoint = "https://grcresponder-dev-search.search.windows.net"

# Get Search API key
$searchApiKey = az search admin-key show --service-name "grcresponder-dev-search" --resource-group $ResourceGroupName --query "primaryKey" -o tsv

if (-not $searchApiKey) {
    Write-Error "Failed to retrieve Search API key"
    exit 1
}

# Configure app settings
Write-Host "Setting application configuration..." -ForegroundColor Cyan

az functionapp config appsettings set --name $FunctionAppName --resource-group $ResourceGroupName --settings @(
    "FUNCTIONS_EXTENSION_VERSION=~4",
    "FUNCTIONS_WORKER_RUNTIME=python",
    "PYTHON_VERSION=3.11",
    "SCM_DO_BUILD_DURING_DEPLOYMENT=true",
    "ENABLE_ORYX_BUILD=true",
    "KEY_VAULT_URL=$keyVaultUrl",
    "SEARCH_SERVICE_ENDPOINT=$searchEndpoint",
    "SEARCH_API_KEY=$searchApiKey",
    "WEBSITE_RUN_FROM_PACKAGE=1"
) | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to configure Function App settings"
    exit 1
}

# Enable system-assigned managed identity
Write-Host "Enabling managed identity..." -ForegroundColor Cyan
az functionapp identity assign --name $FunctionAppName --resource-group $ResourceGroupName | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to enable managed identity"
    exit 1
}

# Get the managed identity principal ID
$principalId = az functionapp identity show --name $FunctionAppName --resource-group $ResourceGroupName --query "principalId" -o tsv

# Grant Key Vault access to managed identity
Write-Host "Granting Key Vault access..." -ForegroundColor Cyan
az keyvault set-policy --name "grcresponder-dev-kv" --object-id $principalId --secret-permissions get list | Out-Null

# Grant Storage access
Write-Host "Granting Storage access..." -ForegroundColor Cyan
$storageAccountId = az storage account show --name "grcresponderdevstorage" --resource-group $ResourceGroupName --query "id" -o tsv
az role assignment create --assignee $principalId --role "Storage Blob Data Contributor" --scope $storageAccountId | Out-Null

# Grant Search access
Write-Host "Granting Search access..." -ForegroundColor Cyan
$searchServiceId = az search service show --name "grcresponder-dev-search" --resource-group $ResourceGroupName --query "id" -o tsv
az role assignment create --assignee $principalId --role "Search Index Data Contributor" --scope $searchServiceId | Out-Null
az role assignment create --assignee $principalId --role "Search Service Contributor" --scope $searchServiceId | Out-Null

# Create storage containers
Write-Host "Creating storage containers..." -ForegroundColor Yellow

$containerNames = @(
    "proceedings-metadata",
    "processing-queue", 
    "documents"
)

foreach ($containerName in $containerNames) {
    Write-Host "  Creating container: $containerName" -ForegroundColor Gray
    az storage container create --name $containerName --account-name "grcresponderdevstorage" --auth-mode login | Out-Null
}

# Package and deploy functions
Write-Host "Packaging Function App..." -ForegroundColor Yellow

# Create deployment package structure
$deploymentPath = "deployment-package"
if (Test-Path $deploymentPath) {
    Remove-Item -Recurse -Force $deploymentPath
}
New-Item -ItemType Directory -Path $deploymentPath | Out-Null

# Copy function files
Copy-Item "function_app.py" -Destination "$deploymentPath\"
Copy-Item "requirements.txt" -Destination "$deploymentPath\"
Copy-Item "host.json" -Destination "$deploymentPath\"

# Create .funcignore file
$funcignoreContent = @"
.git*
.vscode
.pytest_cache
__pycache__
*.pyc
.env
local.settings.json
test_*.py
"@

$funcignorePath = Join-Path $deploymentPath ".funcignore"
Set-Content -Path $funcignorePath -Value $funcignoreContent -Encoding UTF8

# Create deployment zip
Write-Host "Creating deployment package..." -ForegroundColor Cyan
Compress-Archive -Path "$deploymentPath\*" -DestinationPath "grcresponder-backend.zip" -Force

# Deploy to Azure Functions
Write-Host "Deploying to Azure Functions..." -ForegroundColor Yellow
az functionapp deployment source config-zip --name $FunctionAppName --resource-group $ResourceGroupName --src "grcresponder-backend.zip" | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Error "Function deployment failed"
    exit 1
}

# Clean up
Remove-Item -Recurse -Force $deploymentPath
Remove-Item "grcresponder-backend.zip"

Write-Host "Deployment completed successfully!" -ForegroundColor Green
Write-Host "Function App URL: https://$FunctionAppName.azurewebsites.net" -ForegroundColor Cyan
Write-Host "Health Check: https://$FunctionAppName.azurewebsites.net/api/health" -ForegroundColor Cyan