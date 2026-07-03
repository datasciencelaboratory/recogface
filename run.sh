#!/bin/bash
# recogface convenience runner script

# Ensure local X11 permissions are open on the host
xhost +local:docker >/dev/null 2>&1 || xhost + >/dev/null 2>&1

# Workaround for Snap-installed Docker:
# Snap confinement blocks access to /tmp/.X11-unix, causing it to appear empty.
# We mount it inside the project directory (which is under the user's home folder)
# so Snap Docker is allowed to read it and pass the X11 sockets to the container.
mkdir -p .X11-unix
if ! mountpoint -q .X11-unix; then
    echo "[Info] Configurando redirecionamento do socket X11 para o sandbox do Docker..."
    sudo mount --bind /tmp/.X11-unix .X11-unix
fi

# Spin up containers
docker compose up -d

echo "[Info] Iniciando o reconhecimento facial em tempo real..."
echo "[Info] Pressione a tecla 'q' ou 'ESC' na janela de vídeo para encerrar."
echo ""

# Execute the CLI pipeline inside the container
docker compose exec recogface-app recogface run
