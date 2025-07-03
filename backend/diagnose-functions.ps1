# GRCResponder Function App Diagnostic Script
param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroupName = "rg-grcresponder-dev",
    [Parameter(Mandatory=$false)]
    [string]$FunctionAppName = "grcresponder-dev-functions"
)

Write-Host "=== GRCResponder Function App Diagnostics ===" -ForegroundColor Green

# 1. Check Function App status
Write-Host "1. Checking Function App status..." -ForegroundColor Yellow
$functionApp = az functionapp show --name $FunctionAppName --resource-group $ResourceGroupName --query "{name:name, state:state, kind:kind, defaultHostName:defaultHostName}" -o json | ConvertFrom-Json

if ($functionApp) {
    Write-Host "   Function App: $($functionApp.name)" -ForegroundColor White
    Write-Host "   State: $($functionApp.state)" -ForegroundColor White
    Write-Host "   Kind: $($functionApp.kind)" -ForegroundColor White
    Write-Host "   URL: https://$($functionApp.defaultHostName)" -ForegroundColor White
} else {
    Write-Host "   ERROR: Function App not found!" -ForegroundColor Red
    exit 1
}

# 2. Check deployment status
Write-Host "`n2. Checking deployment status..." -ForegroundColor Yellow
$deployments = az functionapp deployment list --name $FunctionAppName --resource-group $ResourceGroupName --query "[0].{id:id, status:status, message:message, active:active}" -o json | ConvertFrom-Json

if ($deployments) {
    Write-Host "   Latest Deployment ID: $($deployments.id)" -ForegroundColor White
    Write-Host "   Status: $($deployments.status)" -ForegroundColor White
    Write-Host "   Active: $($deployments.active)" -ForegroundColor White
    if ($deployments.message) {
        Write-Host "   Message: $($deployments.message)" -ForegroundColor White
    }
} else {
    Write-Host "   No deployments found" -ForegroundColor Red
}

# 3. Check function list
Write-Host "`n3. Checking functions..." -ForegroundColor Yellow
$functions = az functionapp function list --name $FunctionAppName --resource-group $ResourceGroupName --query "[].{name:name, language:language}" -o json | ConvertFrom-Json

if ($functions -and $functions.Count -gt 0) {
    Write-Host "   Found $($functions.Count) functions:" -ForegroundColor Green
    foreach ($func in $functions) {
        Write-Host "     - $($func.name) ($($func.language))" -ForegroundColor White
    }
} else {
    Write-Host "   ERROR: No functions found!" -ForegroundColor Red
}

# 4. Check app settings
Write-Host "`n4. Checking critical app settings..." -ForegroundColor Yellow
$settings = az functionapp config appsettings list --name $FunctionAppName --resource-group $ResourceGroupName --query "[?name=='FUNCTIONS_WORKER_RUNTIME' || name=='PYTHON_VERSION' || name=='SCM_DO_BUILD_DURING_DEPLOYMENT'].{name:name, value:value}" -o json | ConvertFrom-Json

foreach ($setting in $settings) {
    Write-Host "   $($setting.name): $($setting.value)" -ForegroundColor White
}

# 5. Check logs
Write-Host "`n5. Checking recent logs..." -ForegroundColor Yellow
try {
    $logs = az functionapp log tail --name $FunctionAppName --resource-group $ResourceGroupName --timeout 5 2>$null
    if ($logs) {
        Write-Host "   Recent log entries found (showing last few lines):" -ForegroundColor White
        $logs | Select-Object -Last 10 | ForEach-Object { Write-Host "     $_" -ForegroundColor Gray }
    } else {
        Write-Host "   No recent logs available" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   Could not retrieve logs" -ForegroundColor Yellow
}

# 6. Test basic connectivity
Write-Host "`n6. Testing basic connectivity..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "https://$($functionApp.defaultHostName)" -Method GET -TimeoutSec 10 -ErrorAction Stop
    Write-Host "   Basic connectivity: OK (Status: $($response.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "   Basic connectivity: FAILED" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== Diagnosis Complete ===" -ForegroundColor Green
Write-Host "`nRecommended Actions:" -ForegroundColor Yellow
Write-Host "1. If no functions found: Redeploy with correct function_app.py" -ForegroundColor White
Write-Host "2. If deployment failed: Check build logs in Azure Portal" -ForegroundColor White
Write-Host "3. If connectivity fails: Check Function App is running" -ForegroundColor White