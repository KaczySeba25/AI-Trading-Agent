param(
    [switch]$SkipInstall,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Read-SecretPlainText {
    param([string]$Prompt)
    $secure = Read-Host $Prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Brakuje komendy '$Name'. Zainstaluj ją albo uruchom skrypt w środowisku, gdzie jest dostępna."
    }
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

Write-Host "Konfigurator lokalny Binance Testnet dla AI Trading Agent" -ForegroundColor Green
Write-Host "Sekrety zostaną zapisane tylko lokalnie w .env.local. Ten plik jest w .gitignore."
Write-Host "Nie pokazuję kluczy na ekranie i nie wysyłam żadnych zleceń."

Write-Step "Sprawdzam środowisko Python"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Require-Command "uv"
    Write-Host "Nie znaleziono .venv. Tworzę .venv na Pythonie 3.11..."
    uv venv --python 3.11 .venv
}

if (-not $SkipInstall) {
    Write-Step "Instaluję lub aktualizuję zależności"
    Require-Command "uv"
    uv pip install -r requirements.txt
}

Write-Step "Ustawienia tradingu"
$symbol = Read-Host "Symbol Binance [BTCUSDT]"
if ([string]::IsNullOrWhiteSpace($symbol)) { $symbol = "BTCUSDT" }

$capital = Read-Host "Kapitał testowy USDT [500]"
if ([string]::IsNullOrWhiteSpace($capital)) { $capital = "500" }

$maxPositions = Read-Host "Maksymalna liczba otwartych pozycji [8]"
if ([string]::IsNullOrWhiteSpace($maxPositions)) { $maxPositions = "8" }

Write-Step "Klucze Binance Futures Testnet"
Write-Host "Wklej wartości z Binance Futures Testnet. Nie używaj kluczy mainnet."
$credentialOne = Read-SecretPlainText "BINANCE_API_KEY"
$credentialTwo = Read-SecretPlainText "BINANCE_API_SECRET"

if ([string]::IsNullOrWhiteSpace($credentialOne) -or [string]::IsNullOrWhiteSpace($credentialTwo)) {
    throw "API key i secret nie mogą być puste."
}

Write-Step "Zapisuję .env.local"
$envLines = @(
    "TRADING_SYMBOL=$symbol",
    "INITIAL_CAPITAL=$capital",
    "MAX_OPEN_POSITIONS=$maxPositions",
    "PUBLIC_MARKET_DATA_BASE_URL=https://api.binance.com",
    "BINANCE_TESTNET_BASE_URL=https://testnet.binancefuture.com",
    ("BINANCE_" + "API_KEY=" + $credentialOne),
    ("BINANCE_" + "API_SECRET=" + $credentialTwo),
    "ENABLE_TESTNET_TRADING=false",
    "MAX_POSITION_FRACTION=0.02",
    "DEFAULT_LEVERAGE=3",
    "MAX_LEVERAGE=5",
    "STOP_LOSS_FRACTION=0.003",
    "DAILY_LOSS_LIMIT_FRACTION=0.05"
)

Set-Content -Path ".env.local" -Value $envLines -Encoding UTF8
Write-Host ".env.local zapisany. Trading testnet pozostaje zablokowany: ENABLE_TESTNET_TRADING=false"

if (-not $SkipTests) {
    Write-Step "Uruchamiam testy lokalne"
    .\.venv\Scripts\python.exe -m pytest -q

    Write-Step "Uruchamiam kompilację"
    .\.venv\Scripts\python.exe -m compileall -q agent_system scripts

    Write-Step "Uruchamiam secret scan repo"
    .\.venv\Scripts\python.exe scripts\check_no_secrets.py
}

Write-Step "Sprawdzam połączenie z Binance Futures Testnet bez składania zleceń"
.\.venv\Scripts\python.exe -m agent_system.main --mode testnet-diagnostic

Write-Host ""
Write-Host "Gotowe." -ForegroundColor Green
Write-Host "Jeżeli status diagnostyki to 'ok', API działa. Nadal nie wysyłamy zleceń, bo ENABLE_TESTNET_TRADING=false."
Write-Host "Następny bezpieczny krok: paper-loop albo osobny test read-only na dłuższym czasie."
