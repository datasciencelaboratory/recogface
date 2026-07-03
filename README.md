# recogface 👤

Sistema completo e otimizado de reconhecimento facial em tempo real desenvolvido em Python. Utiliza Programação Orientada a Objetos (POO), detecção ultraleve com MediaPipe, extração de embeddings com MobileFaceNet (via OpenCV DNN) e busca vetorial de alta performance com Qdrant.

O sistema é 100% conteinerizado e pronto para rodar sem dependências locais no host.

---

## 🛠️ Tecnologias e Arquitetura

- **Detecção de Faces:** [MediaPipe Face Detection](https://mediapipe.dev/) (otimizado para CPU).
- **Extração de Embeddings:** [MobileFaceNet](https://github.com/deepinsight/insightface) (modelo ONNX de 128 dimensões executado no OpenCV DNN).
- **Banco de Dados Vetorial:** [Qdrant](https://qdrant.tech/) (banco de dados vetorial de alta performance).
- **Orquestração e Interface:** OpenCV (visualização do frame em tempo real) e `argparse` (CLI amigável).
- **DevOps:** Docker & Docker Compose com mapeamento nativo de webcam USB e X11 Display.

---

## 📐 Estrutura de Classes (POO)

A lógica está organizada em módulos de responsabilidade única dentro do pacote [recogface](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface):

1. **[`CameraManager`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/camera.py):** Gerencia a captura dos frames da webcam USB via OpenCV, garantindo liberação segura dos recursos físicos.
2. **[`FaceDetector`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/detector.py):** Processa os frames e recorta os rostos utilizando soluções leves adequadas para CPU (MediaPipe).
3. **[`EmbeddingExtractor`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/extractor.py):** Carrega o modelo ONNX da MobileFaceNet e extrai o vetor de 128 dimensões a partir do recorte facial.
4. **[`VectorDBClient`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/db.py):** Interface com o Qdrant. Cria coleções automaticamente, insere e busca embeddings utilizando métrica de Cosseno.
5. **[`PipelineController`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/pipeline.py):** Orquestra o fluxo em tempo real (Captura ➔ Detecção ➔ Extração ➔ Busca no Banco ➔ Feedback Visual).

---

## 🚀 Como Executar

### 1. Pré-requisitos
- **Sistema Operacional:** Linux (com servidor X11 ativo para exibir a janela do OpenCV).
- **Hardware:** Webcam USB conectada no host no caminho `/dev/video0` e pelo menos 4GB de RAM livre.
- **Software:** Docker e Docker Compose instalados.

### 2. Permitir Acesso ao Servidor X11 (Host)
Para que o container Docker possa renderizar a tela do OpenCV diretamente no seu monitor, você precisa liberar o acesso X11 rodando o seguinte comando **no terminal do seu host**:

```bash
xhost +local:docker
```

### 3. Iniciar a Infraestrutura com Docker Compose
Suba o banco de dados Qdrant e prepare a imagem da aplicação Python (o modelo MobileFaceNet ONNX será baixado e embutido automaticamente no build):

```bash
docker compose up -d
```

---

## ⌨️ Comandos da CLI (Dentro do Container)

Você pode executar o utilitário CLI `recogface` diretamente usando o `docker compose exec`.

### 1. Cadastrar uma Nova Pessoa
Para registrar um novo rosto no banco vetorial, utilize o comando `add`. Você precisa passar o nome e o caminho de uma imagem válida (que pode estar mapeada dentro da pasta do projeto):

```bash
docker compose exec recogface-app recogface add --name "Nome da Pessoa" --image "caminho/da/foto.jpg"
```

*Exemplo:*
```bash
docker compose exec recogface-app recogface add --name "Allan Queiroz" --image "allan.jpg"
```

### 2. Iniciar Reconhecimento Facial em Tempo Real
Inicia o pipeline que abre a webcam, detecta os rostos e compara os embeddings com o banco de dados Qdrant em tempo real:

```bash
docker compose exec recogface-app recogface run
```

- **Sair do Modo de Reconhecimento:** Pressione a tecla `q` ou `ESC` na janela de vídeo para encerrar e liberar a webcam com segurança.

---

## 🏗️ Estrutura do Repositório

O projeto possui a seguinte estrutura organizada:

- [`recogface/`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface): Módulos e classes do sistema.
- [`main.py`](file:///home/allan-queiroz/allan/repos/testes/recogface/main.py): Ponto de entrada alternativo para execução.
- [`setup.py`](file:///home/allan-queiroz/allan/repos/testes/recogface/setup.py): Empacotamento Python que expõe o comando `recogface` no PATH.
- [`Dockerfile`](file:///home/allan-queiroz/allan/repos/testes/recogface/Dockerfile): Definição de imagem e instalação de dependências do OpenCV/GTK.
- [`docker-compose.yml`](file:///home/allan-queiroz/allan/repos/testes/recogface/docker-compose.yml): Orquestração de serviços e volumes.
- [`download_model.py`](file:///home/allan-queiroz/allan/repos/testes/recogface/download_model.py): Utilitário de build para pré-carregar o modelo.
- [`requirements.txt`](file:///home/allan-queiroz/allan/repos/testes/recogface/requirements.txt): Dependências Python da aplicação.
