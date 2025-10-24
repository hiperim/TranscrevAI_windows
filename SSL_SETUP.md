# TranscrevAI - Configuração SSL

Guia completo para configurar HTTPS no TranscrevAI em ambientes de desenvolvimento e produção.

## Por que HTTPS é necessário?

A API `getUserMedia()` do navegador (usada para captura de áudio ao vivo) **requer HTTPS** para funcionar, exceto em `localhost` não-SSL. Para garantir funcionamento consistente e testar em ambiente real, usamos HTTPS mesmo em desenvolvimento.

---

## 🔧 Configuração para Desenvolvimento (mkcert)

### 1. Executar o script de setup

**Como Administrador**, execute:

```batch
setup_dev_certs.bat
```

Este script irá:
- ✅ Instalar Chocolatey (se necessário)
- ✅ Instalar mkcert via Chocolatey
- ✅ Criar uma Certificate Authority (CA) local confiável
- ✅ Gerar certificados SSL para localhost
- ✅ Salvar certificados em `certs/localhost.pem` e `certs/localhost-key.pem`

### 2. Verificar o arquivo .env

Certifique-se de que:

```bash
APP_ENV=development
```

### 3. Executar a aplicação

```batch
python main.py
```

### 4. Acessar

Abra o navegador em: **https://localhost:8000**

O navegador confiará automaticamente no certificado! ✅

---

## 📦 Configuração para Produção (Caddy)

### Arquitetura de Produção

```
Internet → Caddy (HTTPS :443) → TranscrevAI (HTTP :8080)
          ↓
     SSL automático via Let's Encrypt
```

### 1. Arquivo de configuração .env

```bash
APP_ENV=production
```

### 2. Criar Caddyfile

Já criado em `Caddyfile` (raiz do projeto):

```caddy
seudominio.com {
    reverse_proxy localhost:8080
}
```

### 3. Instalar Caddy

**Windows (Chocolatey):**
```batch
choco install caddy -y
```

**Linux:**
```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy
```

### 4. Executar

**Terminal 1 - TranscrevAI:**
```batch
python main.py
```

**Terminal 2 - Caddy:**
```batch
caddy run
```

### 5. Acessar

Abra o navegador em: **https://seudominio.com**

Caddy gerencia automaticamente certificados Let's Encrypt! ✅

---

## 🔄 Alternando entre Ambientes

Edite o arquivo `.env`:

```bash
# Desenvolvimento
APP_ENV=development

# Produção
APP_ENV=production
```

Ou use variável de ambiente temporária:

```batch
REM Windows
set APP_ENV=production
python main.py
```

```bash
# Linux/Mac
APP_ENV=production python main.py
```

---

## 🐛 Solução de Problemas

### Certificados não encontrados

**Problema:** `Development SSL certificates not found`

**Solução:**
1. Execute `setup_dev_certs.bat` como Administrador
2. Verifique se os arquivos existem em `certs/`
3. Reinicie a aplicação

### Navegador não confia no certificado

**Problema:** Aviso de certificado não confiável

**Solução:**
1. Execute: `mkcert -install` como Administrador
2. Reinicie o navegador
3. Acesse novamente

### mkcert não encontrado

**Problema:** `'mkcert' is not recognized as an internal or external command`

**Solução - Instalação Manual:**
1. Visite: https://github.com/FiloSottile/mkcert/releases
2. Baixe `mkcert-v*-windows-amd64.exe`
3. Renomeie para `mkcert.exe`
4. Adicione ao PATH ou coloque na pasta do projeto
5. Execute `setup_dev_certs.bat` novamente

### Porta 8000 já em uso

**Problema:** `Error binding to address`

**Solução:**
```batch
REM Windows - Encontrar processo usando porta 8000
netstat -ano | findstr :8000

REM Matar processo (substitua PID)
taskkill /PID <PID> /F
```

---

## 📚 Referências

- **mkcert:** https://github.com/FiloSottile/mkcert
- **Caddy:** https://caddyserver.com/docs/
- **getUserMedia API:** https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia

---

## 🔐 Segurança

### Desenvolvimento
- ✅ Certificados válidos apenas para `localhost`
- ✅ CA local confiável apenas no seu sistema
- ⚠️ Não compartilhe certificados de desenvolvimento

### Produção
- ✅ Certificados Let's Encrypt automaticamente renováveis
- ✅ Caddy gerencia toda a configuração SSL/TLS
- ✅ Grade A+ em SSL Labs por padrão
