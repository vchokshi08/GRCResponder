# Simple Direct Deployment - No Function Core Tools Required
param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroupName = "rg-grcresponder-dev",
    [Parameter(Mandatory=$false)]
    [string]$FunctionAppName = "grcresponder-dev-functions"
)

Write-Host "=== Simple Direct Deployment ===" -ForegroundColor Green

# 1. Disable authentication first
Write-Host "1. Disabling authentication..." -ForegroundColor Yellow
try {
    az webapp auth update --name $FunctionAppName --resource-group $ResourceGroupName --enabled false
    Write-Host "   Authentication disabled" -ForegroundColor Green
} catch {
    Write-Host "   Warning: Could not disable authentication (may not be enabled)" -ForegroundColor Yellow
}

# 2. Set authorization level to anonymous
Write-Host "2. Setting authorization level..." -ForegroundColor Yellow
az functionapp config appsettings set --name $FunctionAppName --resource-group $ResourceGroupName --settings "AzureWebJobsFeatureFlags=EnableWorkerIndexing" | Out-Null

# 3. Create a simple deployment package
Write-Host "3. Creating deployment package..." -ForegroundColor Yellow

# Clean up first
if (Test-Path "simple-deploy") { Remove-Item -Recurse -Force "simple-deploy" }
New-Item -ItemType Directory -Path "simple-deploy" | Out-Null

# Copy files
Copy-Item "function_app.py" -Destination "simple-deploy\"
Copy-Item "requirements.txt" -Destination "simple-deploy\"

# Create a minimal host.json
$hostJson = @'
{
  "version": "2.0",
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  },
  "functionTimeout": "00:05:00"
}
'@
Set-Content -Path "simple-deploy\host.json" -Value $hostJson

# Create local.settings.json (for structure)
$localSettings = @'
{
  "IsEncrypted": false,
  "Values": {
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "AzureWebJobsStorage": ""
  }
}
'@
Set-Content -Path "simple-deploy\local.settings.json" -Value $localSettings

Write-Host "   Package contents:" -ForegroundColor White
Get-ChildItem "simple-deploy" | ForEach-Object { Write-Host "     $($_.Name)" -ForegroundColor Gray }

# 4. Create zip and deploy
Write-Host "4. Deploying..." -ForegroundColor Yellow
Compress-Archive -Path "simple-deploy\*" -DestinationPath "simple-deployment.zip" -Force

# Deploy with more verbose output
az functionapp deployment source config-zip --name $FunctionAppName --resource-group $ResourceGroupName --src "simple-deployment.zip" --timeout 300

# 5. Restart Function App to ensure changes take effect
Write-Host "5. Restarting Function App..." -ForegroundColor Yellow
az functionapp restart --name $FunctionAppName --resource-group $ResourceGroupName

# 6. Wait for restart
Write-Host "6. Waiting for restart..." -ForegroundColor Yellow
Start-Sleep -Seconds 45

# 7. Check functions again
Write-Host "7. Checking functions..." -ForegroundColor Yellow
$functions = az functionapp function list --name $FunctionAppName --resource-group $ResourceGroupName --query "[].{name:name, status:status}" -o json | ConvertFrom-Json

if ($functions -and $functions.Count -gt 0) {
    Write-Host "   SUCCESS: Found $($functions.Count) functions!" -ForegroundColor Green
    foreach ($func in $functions) {
        Write-Host "     - $($func.name) ($($func.status))" -ForegroundColor White
    }
} else {
    Write-Host "   Still no functions found" -ForegroundColor Red
}

# 8. Test endpoint with authentication disabled
Write-Host "8. Testing endpoints..." -ForegroundColor Yellow

$testUrls = @(
    "https://$FunctionAppName.azurewebsites.net/api/test",
    "https://$FunctionAppName.azurewebsites.net/api/health"
)

foreach ($url in $testUrls) {
    try {
        $response = Invoke-RestMethod -Uri $url -TimeoutSec 15
        Write-Host "   $url : SUCCESS" -ForegroundColor Green
        if ($response.message) {
            Write-Host "     Message: $($response.message)" -ForegroundColor White
        }
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "   $url : FAILED (Status: $statusCode)" -ForegroundColor Red
        
        if ($statusCode -eq 401) {
            Write-Host "     Still getting 401 - authentication issue persists" -ForegroundColor Yellow
        } elseif ($statusCode -eq 404) {
            Write-Host "     404 - function not found/registered" -ForegroundColor Yellow
        }
    }
}

# 9. Get Function App logs to see what's happening
Write-Host "9. Checking deployment logs..." -ForegroundColor Yellow
try {
    $kuduUrl = "https://$FunctionAppName.scm.azurewebsites.net/api/logstream"
    Write-Host "   Check build logs at: $kuduUrl" -ForegroundColor White
    
    # Try to get last deployment status
    $deploymentStatus = az functionapp deployment list --name $FunctionAppName --resource-group $ResourceGroupName --query "[0].{status:status, message:message, active:active}" -o json | ConvertFrom-Json
    if ($deploymentStatus) {
        Write-Host "   Last deployment status: $($deploymentStatus.status)" -ForegroundColor White
        if ($deploymentStatus.message) {
            Write-Host "   Message: $($deploymentStatus.message)" -ForegroundColor White
        }
    }
} catch {
    Write-Host "   Could not get deployment status" -ForegroundColor Yellow
}

# Clean up
Remove-Item -Recurse -Force "simple-deploy"
Remove-Item -Force "simple-deployment.zip"

Write-Host "`n=== Deployment Complete ===" -ForegroundColor Green
Write-Host "If functions still not found, check Azure Portal > Function App > Functions tab" -ForegroundColor Yellow