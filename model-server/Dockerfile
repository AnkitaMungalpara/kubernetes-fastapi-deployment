FROM python:3.12-slim

WORKDIR /opt/src

# Copy only requirements.txt first for better Docker caching
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        libpng-dev \
        && rm -rf /var/lib/apt/lists/*

# Install specific PyTorch CPU version and torchvision
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy the rest of the code and assets
COPY . .

# Expose port (optional for documentation)
EXPOSE 80

# Run FastAPI app using uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
