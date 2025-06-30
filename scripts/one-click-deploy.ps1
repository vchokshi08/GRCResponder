# GRCResponder One-Click Deployment Script
# 100% Azure Cloud Native Serverless Infrastructure as Code
param(
    [Parameter(Mandatory=$false)]
    [string]$SubscriptionId,
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroupName = "rg-grcresponder-dev",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "West US 2",
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev",
    
    [Parameter(Mandatory=$true)]
    [string]$GeminiApiKey
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🏛️⚡ GRCResponder One-Click Deployment" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Yellow
Write-Host "🎯 Target: 100% Azure Cloud Native Serverless" -ForegroundColor Cyan
Write-Host "📍 Resource Group: $ResourceGroupName" -ForegroundColor Cyan
Write-Host "🌍 Location: $Location" -ForegroundColor Cyan
Write-Host "🏷️  Environment: $Environment" -ForegroundColor Cyan

try {
    # Check prerequisites
    Write-Host ""
    Write-Host "🔍 Checking Prerequisites..." -ForegroundColor Blue
    
    # Check Azure CLI
    $azVersion = az version --query '."azure-cli"' -o tsv 2>$null
    if (-not $azVersion) {
        throw "Azure CLI is not installed. Please install from https://aka.ms/installazurecliwindows"
    }
    Write-Host "✅ Azure CLI version: $azVersion" -ForegroundColor Green
    
    # Check login status
    $currentAccount = az account show --query id -o tsv 2>$null
    if (-not $currentAccount) {
        Write-Host "🔑 Please login to Azure..." -ForegroundColor Yellow
        az login
        $currentAccount = az account show --query id -o tsv
    }
    
    # Set subscription if provided
    if ($SubscriptionId) {
        az account set --subscription $SubscriptionId
        Write-Host "✅ Using subscription: $SubscriptionId" -ForegroundColor Green
    } else {
        $SubscriptionId = $currentAccount
        Write-Host "✅ Using current subscription: $SubscriptionId" -ForegroundColor Green
    }
    
    # Verify subscription type
    $subscriptionInfo = az account show --query "{Name:name, State:state}" -o json | ConvertFrom-Json
    Write-Host "✅ Subscription: $($subscriptionInfo.Name) ($($subscriptionInfo.State))" -ForegroundColor Green
    
    # Create resource group
    Write-Host ""
    Write-Host "🏗️  Creating Resource Group..." -ForegroundColor Blue
    az group create --name $ResourceGroupName --location $Location --output none
    Write-Host "✅ Resource group created: $ResourceGroupName" -ForegroundColor Green
    
    # Register required providers
    Write-Host ""
    Write-Host "📋 Registering Azure Resource Providers..." -ForegroundColor Blue
    $providers = @(
        "Microsoft.Web",
        "Microsoft.Storage", 
        "Microsoft.KeyVault",
        "Microsoft.Search",
        "Microsoft.Insights"
    )
    
    foreach ($provider in $providers) {
        Write-Host "   Registering $provider..." -ForegroundColor Yellow
        az provider register --namespace $provider --output none
    }
    
    # Wait for registration
    Write-Host "⏳ Waiting for provider registration..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    foreach ($provider in $providers) {
        $state = az provider show --namespace $provider --query "registrationState" -o tsv
        Write-Host "   $provider`: $state" -ForegroundColor $(if ($state -eq "Registered") { "Green" } else { "Yellow" })
    }
    
    # Prepare parameters
    Write-Host ""
    Write-Host "📝 Preparing Deployment Parameters..." -ForegroundColor Blue
    
    $parametersPath = "infrastructure/grcresponder-parameters.json"
    $templatePath = "infrastructure/grcresponder-infrastructure.json"
    
    # Check if files exist
    if (-not (Test-Path $parametersPath)) {
        throw "Parameters file not found: $parametersPath"
    }
    if (-not (Test-Path $templatePath)) {
        throw "Template file not found: $templatePath"
    }
    
    # Update parameters with provided values
    $parameters = Get-Content $parametersPath | ConvertFrom-Json
    $parameters.parameters.environment.value = $Environment
    $parameters.parameters.location.value = $Location
    $parameters.parameters.geminiApiKey.value = $GeminiApiKey
    
    # Save updated parameters
    $tempParamsPath = "temp-parameters.json"
    $parameters | ConvertTo-Json -Depth 10 | Set-Content $tempParamsPath -Encoding UTF8
    
    Write-Host "✅ Parameters configured" -ForegroundColor Green
    
    # Validate ARM template
    Write-Host ""
    Write-Host "🔍 Validating ARM Template..." -ForegroundColor Blue
    az deployment group validate --resource-group $ResourceGroupName --template-file $templatePath --parameters "@$tempParamsPath" --output none
    
    if ($LASTEXITCODE -ne 0) {
        throw "ARM template validation failed"
    }
    Write-Host "✅ ARM template validation successful" -ForegroundColor Green
    
    # Deploy infrastructure
    Write-Host ""
    Write-Host "🚀 Deploying Infrastructure..." -ForegroundColor Blue
    Write-Host "   This may take 10-15 minutes for complete deployment" -ForegroundColor Yellow
    
    $deploymentName = "grcresponder-deployment-$(Get-Date -Format 'yyyyMMddHHmmss')"
    
    az deployment group create --resource-group $ResourceGroupName --name $deploymentName --template-file $templatePath --parameters "@$tempParamsPath" --output none
    
    if ($LASTEXITCODE -ne 0) {
        throw "Infrastructure deployment failed"
    }
    
    Write-Host "✅ Infrastructure deployment successful" -ForegroundColor Green
    
    # Get deployment outputs
    Write-Host ""
    Write-Host "📊 Retrieving Deployment Information..." -ForegroundColor Blue
    $outputs = az deployment group show --resource-group $ResourceGroupName --name $deploymentName --query properties.outputs --output json | ConvertFrom-Json
    
    # Clean up temp file
    Remove-Item $tempParamsPath -Force
    
    # Display results
    Write-Host ""
    Write-Host "🎉 DEPLOYMENT COMPLETE!" -ForegroundColor Green
    Write-Host "========================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "🔗 Your GRCResponder URLs:" -ForegroundColor Cyan
    Write-Host "   Frontend: https://$($outputs.staticWebAppUrl.value)" -ForegroundColor White
    Write-Host "   API: $($outputs.functionAppUrl.value)" -ForegroundColor White
    Write-Host ""
    Write-Host "📦 Azure Resources Created:" -ForegroundColor Cyan
    Write-Host "   Function App: $($outputs.functionAppName.value)" -ForegroundColor White
    Write-Host "   Storage Account: $($outputs.storageAccountName.value)" -ForegroundColor White
    Write-Host "   Key Vault: $($outputs.keyVaultName.value)" -ForegroundColor White
    Write-Host "   Search Service: $($outputs.searchServiceName.value)" -ForegroundColor White
    Write-Host ""
    Write-Host "💰 Estimated Monthly Cost: ~`$3-25 (pay-per-use)" -ForegroundColor Green
    Write-Host ""
    
    # Save deployment info
    $deploymentInfo = @{
        DeploymentName = $deploymentName
        ResourceGroup = $ResourceGroupName
        Location = $Location
        Environment = $Environment
        Timestamp = Get-Date
        Outputs = $outputs
        SubscriptionId = $SubscriptionId
    }
    
    $deploymentInfo | ConvertTo-Json -Depth 3 | Out-File "deployment-info.json" -Encoding UTF8
    Write-Host "📋 Deployment info saved to: deployment-info.json" -ForegroundColor Yellow
    
    Write-Host ""
    Write-Host "🏆 100% Azure Cloud Native Serverless Infrastructure Ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📚 Next Steps:" -ForegroundColor Cyan
    Write-Host "   1. Deploy backend functions: .\scripts\deploy-backend.ps1" -ForegroundColor White
    Write-Host "   2. Deploy frontend app: .\scripts\deploy-frontend.ps1" -ForegroundColor White
    Write-Host "   3. Configure document processing pipeline" -ForegroundColor White
    Write-Host ""
    Write-Host "🎯 Ready for PG&E Regulatory Document Analysis!" -ForegroundColor Green

} catch {
    Write-Host ""
    Write-Host "❌ Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "🔧 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   1. Ensure you have Azure subscription with sufficient quota" -ForegroundColor White
    Write-Host "   2. Verify Gemini API key is valid" -ForegroundColor White
    Write-Host "   3. Check Azure CLI is logged in: az account show" -ForegroundColor White
    Write-Host "   4. Try different Azure region if quota issues persist" -ForegroundColor White
    Write-Host ""
    
    # Clean up temp file if it exists
    if (Test-Path $tempParamsPath) {
        Remove-Item $tempParamsPath -Force
    }
    
    exit 1
}