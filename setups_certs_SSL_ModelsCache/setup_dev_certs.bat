@echo off
REM ==============================================================================
REM TranscrevAI - Development SSL Certificates Setup
REM ==============================================================================
REM This script installs mkcert and generates local SSL certificates for HTTPS
REM development on localhost. Required for getUserMedia() API (live recording).
REM ==============================================================================

echo.
echo ========================================
echo TranscrevAI - SSL Setup for Development
echo ========================================
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Este script requer privilegios de Administrador!
    echo Por favor, execute como Administrador ^(botao direito -^> Executar como administrador^)
    echo.
    pause
    exit /b 1
)

echo [1/5] Verificando instalacao do mkcert...
where mkcert >nul 2>&1
if %errorLevel% equ 0 (
    echo [OK] mkcert ja esta instalado!
    goto :generate_certs
)

echo [INFO] mkcert nao encontrado. Instalando...
echo.

REM Check if Chocolatey is installed
where choco >nul 2>&1
if %errorLevel% neq 0 (
    echo [2/5] Instalando Chocolatey ^(gerenciador de pacotes^)...
    echo [INFO] Isso pode levar alguns minutos...
    echo.

    REM Install Chocolatey
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

    if %errorLevel% neq 0 (
        echo [ERROR] Falha ao instalar Chocolatey!
        echo.
        echo Instalacao manual:
        echo 1. Visite: https://github.com/FiloSottile/mkcert/releases
        echo 2. Baixe mkcert-v*-windows-amd64.exe
        echo 3. Renomeie para mkcert.exe e adicione ao PATH
        echo 4. Execute este script novamente
        pause
        exit /b 1
    )

    REM Refresh environment
    refreshenv >nul 2>&1
    echo [OK] Chocolatey instalado com sucesso!
) else (
    echo [OK] Chocolatey ja esta instalado!
)

echo.
echo [3/5] Instalando mkcert via Chocolatey...
choco install mkcert -y

if %errorLevel% neq 0 (
    echo [ERROR] Falha ao instalar mkcert!
    echo.
    echo Instalacao manual:
    echo 1. Visite: https://github.com/FiloSottile/mkcert/releases
    echo 2. Baixe mkcert-v*-windows-amd64.exe
    echo 3. Renomeie para mkcert.exe e adicione ao PATH
    pause
    exit /b 1
)

REM Refresh environment to get mkcert in PATH
refreshenv >nul 2>&1
echo [OK] mkcert instalado com sucesso!
echo.

:generate_certs
echo [4/5] Configurando Certificate Authority local...
mkcert -install

if %errorLevel% neq 0 (
    echo [WARNING] Falha ao instalar CA local. Certificados podem nao ser confiaveis.
    echo Isso pode exigir confirmacao manual no navegador.
)

echo.
echo [5/5] Gerando certificados SSL para localhost...

REM Store the script directory
set SCRIPT_DIR=%~dp0

REM Create certs directory if it doesn't exist in the project folder
if not exist "%SCRIPT_DIR%certs" mkdir "%SCRIPT_DIR%certs"

REM Change to certs directory and generate certificates
cd /d "%SCRIPT_DIR%certs"
mkcert localhost 127.0.0.1 ::1

if %errorLevel% neq 0 (
    echo [ERROR] Falha ao gerar certificados!
    cd /d "%SCRIPT_DIR%"
    pause
    exit /b 1
)

REM Return to script directory
cd /d "%SCRIPT_DIR%"
echo [OK] Certificados gerados com sucesso!
echo.

REM Display generated files
echo ========================================
echo Certificados gerados:
echo ========================================
dir /b "%SCRIPT_DIR%certs\localhost*.pem"
echo.
echo Localizacao: %SCRIPT_DIR%certs\
echo.

echo ========================================
echo [SUCESSO] Configuracao concluida!
echo ========================================
echo.
echo Proximos passos:
echo 1. Certifique-se de que APP_ENV=development no arquivo .env
echo 2. Execute: python main.py
echo 3. Acesse: https://localhost:8000
echo.
echo Notas:
echo - Os certificados sao validos apenas para desenvolvimento local
echo - Seu navegador confiara automaticamente nos certificados
echo - Para producao, use Caddy com certificados reais
echo.
pause
