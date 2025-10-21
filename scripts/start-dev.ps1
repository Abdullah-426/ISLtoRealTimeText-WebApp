# Windows PowerShell script to start all services
Write-Host "Starting ISL Real-time Text Application..." -ForegroundColor Green

# Function to start a service in background
function Start-Service {
    param(
        [string]$Name,
        [string]$Path,
        [string]$Command,
        [int]$Port
    )
    
    Write-Host "Starting $Name on port $Port..." -ForegroundColor Yellow
    
    # Check if port is already in use
    $portInUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($portInUse) {
        Write-Host "Port $Port is already in use. Stopping existing process..." -ForegroundColor Red
        $process = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*$Port*" }
        if ($process) {
            $process | Stop-Process -Force
            Start-Sleep -Seconds 2
        }
    }
    
    # Start the service
    $job = Start-Job -ScriptBlock {
        param($p, $c)
        Set-Location $p
        Invoke-Expression $c
    } -ArgumentList $Path, $Command
    
    Start-Sleep -Seconds 3
    return $job
}

# Start Inference Service (Port 8001)
$inferJob = Start-Service -Name "Inference Service" -Path "services/infer" -Command ".venv\Scripts\activate; uvicorn main:app --host 0.0.0.0 --port 8001" -Port 8001

# Start Postprocess Service (Port 8000)  
$postJob = Start-Service -Name "Postprocess Service" -Path "services/postprocess" -Command ".venv\Scripts\activate; uvicorn main:app --host 0.0.0.0 --port 8000" -Port 8000

# Wait a moment for services to start
Start-Sleep -Seconds 5

# Check if services are running
Write-Host "`nChecking services..." -ForegroundColor Cyan

try {
    $inferResponse = Invoke-WebRequest -Uri "http://localhost:8001/" -UseBasicParsing -TimeoutSec 5
    Write-Host "✅ Inference Service: Running on port 8001" -ForegroundColor Green
} catch {
    Write-Host "❌ Inference Service: Not responding on port 8001" -ForegroundColor Red
}

try {
    $postResponse = Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing -TimeoutSec 5
    Write-Host "✅ Postprocess Service: Running on port 8000" -ForegroundColor Green
} catch {
    Write-Host "❌ Postprocess Service: Not responding on port 8000" -ForegroundColor Red
}

# Start Frontend
Write-Host "`nStarting Frontend..." -ForegroundColor Yellow
Set-Location "frontend"

# Check if node_modules exists, if not install dependencies
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "Starting Next.js development server..." -ForegroundColor Yellow
Write-Host "`n🚀 Application will be available at: http://localhost:3000" -ForegroundColor Green
Write-Host "`nPress Ctrl+C to stop all services" -ForegroundColor Cyan

# Start frontend (this will block)
npm run dev
