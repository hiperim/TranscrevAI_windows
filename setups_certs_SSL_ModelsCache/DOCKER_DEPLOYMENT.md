# TranscrevAI - Docker Deployment

Aplicacao de transcricao e diarizacao de audio com modelos ML embedded.

**Suporte Multi-Arquitetura:** AMD64 (Intel/AMD) e ARM64 (Apple Silicon)

---

## Como Rodar

### Pre-requisitos

- Docker e Docker Compose
- ~25GB de espaco em disco

### Comando

```bash
# Clone o repositorio
git clone https://github.com/hiperim/transcrevai.git
cd transcrevai

# Rode (auto-detecta hardware)
docker compose -f docker-compose.pull.yml up
```

### Acesso

http://localhost:8000

---

## Detalhes

- **Imagem:** ~20GB (modelos ML embedded)
- **Download:** Automatico do Docker Hub
- **Token HF:** Nao necessario
- **Internet:** Apenas para download inicial
- **Performance:** Auto-detecta CPU/RAM disponiveis

### Comandos Uteis

```bash
# Rodar em background
docker compose -f docker-compose.pull.yml up -d

# Ver logs
docker compose -f docker-compose.pull.yml logs -f

# Parar
docker compose -f docker-compose.pull.yml down
```

---

## Build Local (Opcional - Para Desenvolvedores)

Se precisar modificar a imagem:

1. Crie `.env` com token HF:
   ```
   HUGGING_FACE_HUB_TOKEN=hf_xxx
   ```

2. Execute:
   ```bash
   # Windows
   .\SETUPs_certs_SSL_ModelsCache\build-multiarch.ps1

   # Linux/Mac
   ./SETUPs_certs_SSL_ModelsCache/build-multiarch.sh
   ```

Para detalhes: [ARM_COMPATIBILITY.md](./ARM_COMPATIBILITY.md)
