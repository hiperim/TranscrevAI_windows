# TranscrevAI - Guia de Deployment com Docker

Este guia fornece instruções para configurar e rodar a aplicação TranscrevAI usando Docker. A abordagem utiliza um cache de modelos local para garantir que a aplicação seja autossuficiente e funcione 100% offline no runtime.

**✅ Suporte Multi-Arquitetura:** TranscrevAI suporta AMD64 (Intel/AMD) e ARM64 (Apple Silicon) através de imagens Docker multi-arch.

## Pré-requisitos

- Docker e Docker Compose (Docker Desktop recomendado para multi-arch builds)
- Git
- Python 3.11+
- Um token de acesso do Hugging Face (para o download inicial dos modelos)

## 🌍 Opções de Build

### Opção A: Build Multi-Arquitetura (Recomendado)

Para criar imagens que funcionam em **Intel/AMD (x86_64)** e **Apple Silicon (ARM64)**:

**Windows:**
```powershell
.\SETUPs_certs_SSL_ModelsCache\build-multiarch.ps1
```

**Linux/Mac:**
```bash
chmod +x ./SETUPs_certs_SSL_ModelsCache/build-multiarch.sh
./SETUPs_certs_SSL_ModelsCache/build-multiarch.sh
```

**Nota:** Requer Docker Desktop e faz push automático para Docker Hub.

Para mais detalhes, consulte: [ARM_COMPATIBILITY.md](./ARM_COMPATIBILITY.md)

### Opção B: Build Local Simples

Para build local em sua arquitetura nativa (sem push para Docker Hub):

```bash
docker-compose up -d --build
```

---

## 🚀 Passo 1: Setup Inicial (Apenas uma vez)

Após clonar o repositório, o passo mais importante é popular o cache de modelos local. Este cache viverá dentro do seu projeto na pasta `models/.cache/`, tornando a aplicação totalmente portátil.

1.  **Navegue até a pasta do projeto:**
    ```bash
    cd TranscrevAI_windows
    ```

2.  **Crie um arquivo `.env`:**
    Crie um arquivo chamado `.env` na raiz do projeto e adicione seu token do Hugging Face:
    ```
    HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

3.  **Execute o script de download:**
    Este comando irá baixar todos os modelos necessários para a pasta `models/.cache/`.
    ```bash
    python SETUPs_certs_SSL_ModelsCache/download_models.py
    ```

Com os modelos baixados localmente, você está pronto para rodar a aplicação em qualquer um dos modos abaixo.

--- 

## 📦 Modo 1: Rodando em Produção

Este é o modo padrão para usar a aplicação. Ele usa a imagem Docker otimizada.

```bash
# Constrói a imagem (se não existir) e inicia o container em background
docker-compose up -d --build

# Para ver os logs
docker-compose logs -f

# Para parar o container
docker-compose down
```

**Acesse a aplicação em:** [http://localhost:8000](http://localhost:8000)

--- 

## 💻 Modo 2: Rodando em Desenvolvimento (com Hot-Reload)

Este modo é ideal para desenvolvimento. Ele monta o seu código local dentro do container, então qualquer mudança que você fizer nos arquivos `.py` será refletida automaticamente sem precisar reconstruir a imagem.

```bash
# Constrói a imagem base e inicia o container de desenvolvimento
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# Para ver os logs com hot-reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# Para parar os containers
docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
```

--- 

## 🧪 Modo 3: Rodando a Suíte de Testes (Pytest)

Este modo usa um ambiente Docker específico para testes, que inclui o `pytest` e outras dependências de desenvolvimento. Ele garante que os testes rodem em um ambiente Linux limpo, idêntico ao de produção.

**1. Construa a imagem de teste:**
Este comando precisa ser executado apenas uma vez ou sempre que o `Dockerfile.test` mudar.
```bash
docker-compose -f docker-compose.test.yml build
```

**2. Execute os testes:**
Este comando inicia um container temporário, roda o `pytest`, e remove o container ao finalizar.
```bash
# Para rodar a suíte de testes completa
docker-compose -f docker-compose.test.yml run --rm transcrevai-test python -m pytest tests/test_unit.py -v

# Para rodar um teste específico (ex: o de performance)
docker-compose -f docker-compose.test.yml run --rm transcrevai-test python -m pytest tests/test_unit.py::test_pipeline_quality_and_performance -v
```
