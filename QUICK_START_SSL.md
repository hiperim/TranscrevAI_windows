# 🚀 Quick Start - SSL Setup

Guia rápido para começar a usar HTTPS no TranscrevAI.

## Para Desenvolvimento (Windows)

### 1️⃣ Execute o setup (como Administrador)

```batch
setup_dev_certs.bat
```

### 2️⃣ Inicie a aplicação

```batch
python main.py
```

### 3️⃣ Acesse

Abra: **https://localhost:8000**

✅ Pronto! O navegador confiará automaticamente no certificado.

---

## Para Produção

### 1️⃣ Configure o ambiente

Edite `.env`:
```bash
APP_ENV=production
```

### 2️⃣ Configure o Caddyfile

Edite `Caddyfile` e substitua `seudominio.com` pelo seu domínio:
```caddy
meusite.com {
    reverse_proxy localhost:8080
}
```

### 3️⃣ Inicie os serviços

**Terminal 1:**
```batch
python main.py
```

**Terminal 2:**
```batch
caddy run
```

### 4️⃣ Acesse

Abra: **https://meusite.com**

✅ Caddy gerencia SSL automaticamente!

---

## 🔄 Alternar Ambientes

Edite `.env`:

```bash
# Desenvolvimento (HTTPS local com mkcert)
APP_ENV=development

# Produção (HTTP interno, Caddy gerencia SSL)
APP_ENV=production
```

---

## ❓ Problemas?

Consulte: `SSL_SETUP.md` para guia completo de solução de problemas.

---

## 📝 Notas Importantes

- **Development:** Certificados válidos apenas para `localhost`
- **Production:** Caddy usa Let's Encrypt (renovação automática)
- **HTTPS é necessário:** Para `getUserMedia()` funcionar (captura de áudio)
