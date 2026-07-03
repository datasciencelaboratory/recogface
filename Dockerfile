FROM python:3.10-slim

# Set environment variables to optimize Python runtime and display configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QT_X11_NO_MITSHM=1

# Install system dependencies for OpenCV (GUI/GTK/QT) and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libasound2 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xfixes0 \
    libxkbcommon-x11-0 \
    libsm6 \
    libice6 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files required to install dependencies
COPY requirements.txt setup.py /app/

# Install python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and download script
COPY download_model.py main.py /app/
COPY recogface/ /app/recogface/

# Pre-download the MobileFaceNet ONNX model during docker build
RUN python download_model.py

# Install the application as a package to register the 'recogface' CLI command in PATH
RUN pip install --no-cache-dir .

# Keep the container running by default to allow CLI execution
CMD ["tail", "-f", "/dev/null"]
