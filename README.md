# recogface 👤

Sistema completo, moderno e altamente otimizado de reconhecimento facial em tempo real desenvolvido em Python. Utiliza Programação Orientada a Objetos (POO), detecção baseada em MediaPipe, extração de embeddings de alta precisão com **FaceNet 512-D** (via OpenCV DNN) e busca vetorial de alta performance com **Qdrant**.

O sistema foi estruturado e conteinerizado para rodar 100% isolado no Docker, fornecendo compatibilidade nativa com **servidores X11 e Wayland (Xwayland)**, inclusive sob ambientes onde o Docker foi instalado via **Snap** (Ubuntu).

---

## 🛠️ Tecnologias e Arquitetura

- **Detecção de Faces:** [MediaPipe Face Detection](https://mediapipe.dev/) (usando o modelo `Full-Range` configurado para até 5 metros de distância, otimizado para CPU).
- **Extração de Embeddings:** [FaceNet](https://huggingface.co/haikalmumtaz/facenet-onnx) (modelo ONNX de 512 dimensões executado no OpenCV DNN, com pré-processamento de margem de 25% ao redor do rosto para máxima acurácia).
- **Banco de Dados Vetorial:** [Qdrant](https://qdrant.tech/) (banco de dados vetorial oficial rodando na porta 6333, com suporte a busca de cosseno de 512 dimensões).
- **Interface e HUD:** OpenCV para manipulação gráfica e desenho de bounding boxes, e `argparse` para a CLI.
- **DevOps:** Docker e Docker Compose com script facilitador de permissões X11 e bind-mounts dinâmicos.

---

## 📐 Estrutura de Classes (POO)

A lógica está organizada em módulos de responsabilidade única dentro do pacote [recogface](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface):

1. **[`CameraManager`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/camera.py):** Inicializa e gerencia a captura física de frames da webcam host (`/dev/video0`) via OpenCV, com suporte a gerenciadores de contexto (`with`).
2. **[`FaceDetector`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/detector.py):** Detecta rostos no frame usando MediaPipe (Full-Range) e extrai recortes faciais aplicando uma margem de segurança de 25% para melhorar a precisão da rede de embeddings.
3. **[`EmbeddingExtractor`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/extractor.py):** Carrega o modelo ONNX do FaceNet (512-D) e gera vetores normalizados L2 prontos para buscas de Cosseno.
4. **[`VectorDBClient`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/db.py):** Interface com o Qdrant. Possui um mecanismo inteligente de auto-migração que detecta incompatibilidades de tamanho de vetores antigos e recria a coleção automaticamente se necessário.
5. **[`PipelineController`](file:///home/allan-queiroz/allan/repos/testes/recogface/recogface/pipeline.py):** Orquestra o fluxo em tempo real (Frames ➔ Detecção ➔ Recorte Padded ➔ Embedding 512D ➔ Busca por Cosseno ➔ HUD na tela).

---

## 🚀 Como Inicializar e Executar

O projeto foi projetado para rodar com apenas um comando no Host, sem dependências locais de sistema ou bibliotecas Python.

### 1. Pré-requisitos
- **Sistema Operacional:** Linux (X11 ou Wayland com Xwayland ativo).
- **Hardware:** Webcam USB conectada e mapeada em `/dev/video0`.
- **Software:** Docker e Docker Compose instalados.

### 2. Executando o Sistema (Script Unificado)
Criei o script executável [`run.sh`](file:///home/allan-queiroz/allan/repos/testes/recogface/run.sh) que automatiza todas as configurações de permissões de tela, resolve limitações de sandbox do Docker Snap e inicia a aplicação:

No terminal do seu **Host**, execute:
```bash
./run.sh
```

**O que o script faz por debaixo dos panos?**
1. Libera permissões de tela no Host (`xhost +`).
2. Cria e faz o bind-mount do socket do X11 (`/tmp/.X11-unix`) para dentro de uma pasta local do projeto (`./.X11-unix`). Isso permite que o Docker instalado por **Snap** contorne o isolamento e consiga renderizar a tela gráfica de vídeo.
3. Sobe os containers em segundo plano.
4. Inicia o pipeline de reconhecimento facial automaticamente.

- **Para Sair da Câmera:** Clique na janela de vídeo e pressione a tecla **`q`** ou **`ESC`**.

---

## ⌨️ Comandos da CLI (Dentro do Container)

Você pode executar o utilitário CLI `recogface` diretamente usando o `docker compose exec`.

### 1. Cadastrar uma Pessoa no Banco Vetorial
Devido ao isolamento do container, a imagem de cadastro deve estar **dentro da pasta do projeto** (onde é montado o volume do container).

1. Coloque a foto desejada na raiz do projeto (ex: `allan.jpg`).
2. Execute o comando `add` da CLI:
   ```bash
   docker compose exec recogface-app recogface add --name "Nome da Pessoa" --image "imagem.jpg"
   ```

*Caso a imagem contenha mais de uma face, a CLI exibirá um aviso e registrará automaticamente a maior face encontrada.*

### 2. Iniciar Reconhecimento Facial Manualmente
Caso você queira rodar o reconhecimento em tempo real sem passar pelo script principal `run.sh` (e os containers já estejam rodando com `docker compose up -d`):
```bash
docker compose exec recogface-app recogface run --threshold 0.65
```

- **`--threshold`:** Limiar de similaridade de cosseno (padrão calibrado: `0.65` para FaceNet 512).
- **`--camera`:** ID do dispositivo de vídeo a ser utilizado no container (padrão: `0` para `/dev/video0`).

---

## 🧹 Limpeza e Manutenção

Para parar e remover todos os containers criados pelo Docker Compose, execute na raiz do projeto:
```bash
docker compose down
```
