# GRCResponder Infrastructure Deployment Script for New Azure Account
param(
    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroupName = "rg-grcresponder-dev",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "West US 2"
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "Starting GRCResponder Infrastructure Deployment" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Yellow
Write-Host "Resource Group: $ResourceGroupName" -ForegroundColor Cyan
Write-Host "Location: $Location" -ForegroundColor Cyan

try {
    # Get current subscription if not provided
    if (-not $SubscriptionId) {
        $currentAccount = az account show --query id -o tsv
        $SubscriptionId = $currentAccount
        Write-Host "Using current subscription: $SubscriptionId" -ForegroundColor Yellow
    }

    # Validate ARM template
    Write-Host "Validating ARM template..." -ForegroundColor Blue
    az deployment group validate --resource-group $ResourceGroupName --template-file "grcresponder-infrastructure.json" --parameters "@grcresponder-parameters.json"

    if ($LASTEXITCODE -ne 0) {
        throw "ARM template validation failed"
    }

    Write-Host "ARM template validation successful" -ForegroundColor Green

    # Deploy infrastructure
    Write-Host "Deploying infrastructure... This may take 10-15 minutes" -ForegroundColor Blue
    $deploymentName = "grcresponder-deployment-$(Get-Date -Format 'yyyyMMddHHmmss')"

    az deployment group create --resource-group $ResourceGroupName --name $deploymentName --template-file "grcresponder-infrastructure.json" --parameters "@grcresponder-parameters.json"

    if ($LASTEXITCODE -ne 0) {
        throw "Infrastructure deployment failed"
    }

    Write-Host "Infrastructure deployment successful" -ForegroundColor Green

    # Get deployment outputs
    Write-Host "Retrieving deployment outputs..." -ForegroundColor Blue
    $outputs = az deployment group show --resource-group $ResourceGroupName --name $deploymentName --query properties.outputs --output json | ConvertFrom-Json

    # Display results
    Write-Host ""
    Write-Host "Deployment Complete!" -ForegroundColor Green
    Write-Host "========================" -ForegroundColor Yellow
    Write-Host "Function App Name: $($outputs.functionAppName.value)" -ForegroundColor Cyan
    Write-Host "Function App URL: $($outputs.functionAppUrl.value)" -ForegroundColor Cyan
    Write-Host "Static Web App URL: https://$($outputs.staticWebAppUrl.value)" -ForegroundColor Cyan
    Write-Host "Storage Account: $($outputs.storageAccountName.value)" -ForegroundColor Cyan
    Write-Host "Key Vault: $($outputs.keyVaultName.value)" -ForegroundColor Cyan
    Write-Host "Search Service: $($outputs.searchServiceName.value)" -ForegroundColor Cyan

    # Save deployment info
    $deploymentInfo = @{
        DeploymentName = $deploymentName
        ResourceGroup = $ResourceGroupName
        Timestamp = Get-Date
        Outputs = $outputs
    }

    $deploymentInfo | ConvertTo-Json -Depth 3 | Out-File "deployment-info.json"
    Write-Host "Deployment info saved to: deployment-info.json" -ForegroundColor Yellow

    Write-Host ""
    Write-Host "100% Azure Cloud Native Serverless Infrastructure Ready!" -ForegroundColor Green

} catch {
    Write-Host "Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}