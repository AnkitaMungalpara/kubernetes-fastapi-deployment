# Kubernetes Deployment with Ingress

This project demonstrates how to deploy a **FastAPI-based model server** and a **web server** backed by a **Redis cache** on **Kubernetes (K8S)**. We use **Docker** to containerize services, **Minikube** as a local Kubernetes cluster, and **Ingress** to expose the application to the outside world.

Think of it as building a small production-like environment:

* **Model Server** → Runs an ONNX model for image classification.
* **Web Server** → Handles requests and communicates with the model server + Redis.
* **Redis** → Provides caching so duplicate requests don’t waste compute.
* **Kubernetes** → Orchestrates the services.
* **Ingress** → Provides access through a hostname.

## Key Terminologies

* **Pod**: The smallest deployable unit in Kubernetes that runs containers.
* **Deployment**: Ensures a defined number of pods are always running.
* **Service**: Provides networking between pods and stable endpoints.
* **Ingress**: Exposes HTTP/HTTPS routes to internal services.
* **Minikube**: Local tool for running Kubernetes clusters.
* **Redis**: An in-memory database used here for caching.
* **FastAPI**: A Python framework for building APIs.

## Step 1: Docker Compose (Local Test)

Before Kubernetes, let’s test our services using Docker Compose.

👉 Let's Build the images:

## Model Server

### model-server/server.py

```python
import urllib
import io
import os
import zlib
import json
import socket
import logging
import requests

import redis.asyncio as redis
import torch
import onnxruntime as ort
import numpy as np

from PIL import Image
from fastapi import FastAPI, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Dict, Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - ModelServer - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mamba Model Server")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "resnet50_imagenet_1k_model.onnx")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.txt")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
HOSTNAME = socket.gethostname()

@app.on_event("startup")
async def initialize():
    global model, device, categories, redis_pool, onnx_session

    logger.info(f"Initializing model server on host {HOSTNAME}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading ONNX model: {MODEL_PATH}")
    onnx_session = ort.InferenceSession(MODEL_PATH)
    logger.info("ONNX model loaded successfully")

    # Load ImageNet labels
    logger.info("Loading local label file")
    with open(LABELS_PATH, "r") as f:
        categories = [line.strip() for line in f.readlines()]
    logger.info(f"Loaded {len(categories)} categories")

    # Redis setup
    logger.info(f"Creating Redis connection pool: host={REDIS_HOST}, port={REDIS_PORT}")
    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=0,
        decode_responses=True,
    )
    logger.info("Model server initialization complete")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup connection pool on shutdown"""
    logger.info("Shutting down model server")
    await redis_pool.aclose()
    logger.info("Cleanup complete")

def get_redis():
    return redis.Redis(connection_pool=redis_pool)

def predict(inp_img: Image) -> Dict[str, float]:
    logger.debug("Starting prediction")

    # Convert to RGB and resize to expected size (e.g., 224x224 for ResNet50)
    img = inp_img.convert("RGB").resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0

    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)  # NCHW + dtype fix


    # Run ONNX inference
    logger.debug("Running ONNX inference")
    ort_inputs = {onnx_session.get_inputs()[0].name: img_tensor}
    ort_outs = onnx_session.run(None, ort_inputs)

    # Convert to torch tensor and apply softmax
    probabilities = torch.nn.functional.softmax(torch.tensor(ort_outs[0][0]), dim=0)

    # Get top predictions
    top_prob, top_catid = torch.topk(probabilities, 5)
    confidences = {
        categories[idx.item()]: float(prob)
        for prob, idx in zip(top_prob, top_catid)
    }

    logger.debug(f"Prediction complete. Top class: {list(confidences.keys())[0]}")
    return confidences


async def write_to_cache(file: bytes, result: Dict[str, float]) -> None:
    cache = get_redis()
    hash = str(zlib.adler32(file))
    logger.debug(f"Writing prediction to cache with hash: {hash}")
    await cache.set(hash, json.dumps(result))
    logger.debug("Cache write complete")

@app.post("/infer")
async def infer(image: Annotated[bytes, File()]):
    logger.info("Received inference request")
    img: Image.Image = Image.open(io.BytesIO(image))

    logger.debug("Running prediction")
    predictions = predict(img)

    logger.debug("Writing results to cache")
    await write_to_cache(image, predictions)

    logger.info("Inference complete")
    return predictions

@app.get("/health")
async def health_check():
    try:
        redis_client = get_redis()
        redis_connected = await redis_client.ping()
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        redis_connected = False

    return {
        "status": "healthy",
        "hostname": HOSTNAME,
        "model": MODEL_PATH,
        "device": str(device) if "device" in globals() else None,
        "redis": {
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "connected": redis_connected,
        },
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### model-server/requirements.txt

```python
fastapi[all]==0.115.6
redis==5.2.1
requests
onnxruntime
uvicorn
pillow
```

### Dockerfile

```python
FROM python:3.12-slim

WORKDIR /opt/src

# Copy only requirements.txt first for better Docker caching
COPY requirements.txt .

# Install specific PyTorch CPU version and torchvision
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Copy the rest of the code and assets
COPY . .

# Expose port
EXPOSE 80

# Run FastAPI app using uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
```

## Web Server

### web-server/server.py

```python
import os
import zlib
import json
import logging
import socket

import redis.asyncio as redis
import httpx

from fastapi import FastAPI, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - WebServer - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Web Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add constants
HOSTNAME = socket.gethostname()
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", "6379")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "<http://localhost:8000>")

@app.on_event("startup")
async def initialize():
    global redis_pool
    logger.info(f"Initializing web server on host {HOSTNAME}")
    logger.info(f"Creating Redis connection pool: host={REDIS_HOST}, port={REDIS_PORT}")
    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=0,
        decode_responses=True,
    )
    logger.info("Web server initialization complete")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup connection pool on shutdown"""
    logger.info("Shutting down web server")
    await redis_pool.aclose()
    logger.info("Cleanup complete")

def get_redis():
    return redis.Redis(connection_pool=redis_pool)

async def check_cached(image: bytes):
    hash = zlib.adler32(image)
    cache = get_redis()

    logger.debug(f"Checking cache for image hash: {hash}")
    data = await cache.get(hash)

    if data:
        logger.info(f"Cache hit for image hash: {hash}")
    else:
        logger.info(f"Cache miss for image hash: {hash}")

    return json.loads(data) if data else None

@app.post("/classify-imagenet")
async def classify_imagenet(image: Annotated[bytes, File()]):
    logger.info("Received classification request")
    infer_cache = await check_cached(image)

    if infer_cache == None:
        logger.info("Making request to model server")
        async with httpx.AsyncClient() as client:
            try:
                url = f"{MODEL_SERVER_URL}/infer"
                files = {"image": image}

                logger.debug(f"Sending request to model server: {url}")
                response = await client.post(url, files=files)
                response.raise_for_status()

                logger.info("Successfully received model prediction")
                return response.json()
            except Exception as e:
                logger.error(f"Model server request failed: {str(e)}")
                raise HTTPException(status_code=500, detail="Error from Model Endpoint")

    return infer_cache

@app.get("/health")
async def health_check():
    """Health check endpoint for kubernetes readiness/liveness probes"""
    try:
        # Test Redis connection
        redis_client = get_redis()
        redis_connected = await redis_client.ping()
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        redis_connected = False

    try:
        # Test Model Server connection
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MODEL_SERVER_URL}/health")
            response.raise_for_status()
            model_health = response.json()
            model_connected = True
    except Exception as e:
        logger.error(f"Model server health check failed: {str(e)}")
        model_connected = False
        model_health = None

    health_status = {
        "status": "healthy" if (redis_connected and model_connected) else "degraded",
        "hostname": HOSTNAME,
        "redis": {"host": REDIS_HOST, "port": REDIS_PORT, "connected": redis_connected},
        "model_server": {
            "url": MODEL_SERVER_URL,
            "connected": model_connected,
            "health": model_health,
        },
    }

    logger.info(f"Health check status: {health_status['status']}")
    return health_status
```

### web-server/requirements.txt

```
fastapi[all]==0.115.6
redis==5.2.1
httpx==0.28.1
requests
```

### Dockerfile

```python
FROM python:3.12

WORKDIR /opt/src

COPY requirements.txt .

RUN pip install -r requirements.txt \
	&& rm -rf /root/.cache/pip

COPY server.py .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
```

```python
docker compose build
```

```
[+] Building 3.2s (26/26) FINISHED                                                                                                              
 => [internal] load local bake definitions                                                                                                 0.0s
 => => reading from stdin 929B                                                                                                             0.0s
 => [web-server internal] load build definition from Dockerfile                                                                            0.0s
 => => transferring dockerfile: 251B                                                                                                       0.0s
 => [model-server internal] load build definition from Dockerfile                                                                          0.0s
 => => transferring dockerfile: 869B                                                                                                       0.0s
 => [web-server internal] load metadata for docker.io/library/python:3.12                                                                  0.9s
 => [model-server internal] load metadata for docker.io/library/python:3.12-slim                                                           0.9s
 => [auth] library/python:pull token for registry-1.docker.io                                                                              0.0s
 => [web-server internal] load .dockerignore                                                                                               0.0s
 => => transferring context: 2B                                                                                                            0.0s
 => [model-server internal] load .dockerignore                                                                                             0.0s
 => => transferring context: 2B                                                                                                            0.0s
 => [web-server 1/5] FROM docker.io/library/python:3.12@sha256:645df645815f1403566b103b2a2bb07f6a01516bbb15078ed004e41d198ba194            0.0s
 => => resolve docker.io/library/python:3.12@sha256:645df645815f1403566b103b2a2bb07f6a01516bbb15078ed004e41d198ba194                       0.0s
 => [web-server internal] load build context                                                                                               0.0s
 => => transferring context: 66B                                                                                                           0.0s
 => [model-server 1/7] FROM docker.io/library/python:3.12-slim@sha256:9c1d9ed7593f2552a4ea47362ec0d2ddf5923458a53d0c8e30edf8b398c94a31     0.0s
 => => resolve docker.io/library/python:3.12-slim@sha256:9c1d9ed7593f2552a4ea47362ec0d2ddf5923458a53d0c8e30edf8b398c94a31                  0.0s
 => [model-server internal] load build context                                                                                             0.0s
 => => transferring context: 5.21kB                                                                                                        0.0s
 => CACHED [model-server 2/7] WORKDIR /opt/src                                                                                             0.0s
 => CACHED [model-server 3/7] COPY requirements.txt .                                                                                      0.0s
 => CACHED [model-server 4/7] RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         libjpeg-de  0.0s
 => CACHED [model-server 5/7] RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1     -f https://download.pytorch.org/whl/cpu  0.0s
 => CACHED [model-server 6/7] RUN pip install --no-cache-dir -r requirements.txt     && rm -rf /root/.cache/pip                            0.0s
 => [model-server 7/7] COPY . .                                                                                                            0.2s
 => CACHED [web-server 2/5] WORKDIR /opt/src                                                                                               0.0s
 => CACHED [web-server 3/5] COPY requirements.txt .                                                                                        0.0s
 => CACHED [web-server 4/5] RUN pip install -r requirements.txt  && rm -rf /root/.cache/pip                                                0.0s
 => CACHED [web-server 5/5] COPY server.py .                                                                                               0.0s
 => [web-server] exporting to image                                                                                                        0.1s
 => => exporting layers                                                                                                                    0.0s
 => => exporting manifest sha256:992486908d0a1db3a4f5a4225059957aedc45615f1b3d63310094a2d041e1133                                          0.0s
 => => exporting config sha256:dbe319f78d420f20b22a588e45ee9141f886157aa8fd534b7a77a13b141fe3ca                                            0.0s
 => => exporting attestation manifest sha256:b571b3ee7d41838c3eaf180926c89eb0fbe50aa45af6340abd090e0c64706214                              0.0s
 => => exporting manifest list sha256:40014ab56c238b9f5178b76f5c638063921bdd111ff74509b59e05434a237aa2                                     0.0s
 => => naming to docker.io/library/s14-k8s-web-server:latest                                                                               0.0s
 => [web-server] resolving provenance for metadata file                                                                                    0.0s
 => [model-server] exporting to image                                                                                                      2.0s
 => => exporting layers                                                                                                                    2.0s
 => => exporting manifest sha256:ea0c59eaf737cc473cb7ab010f1eb521d2c37536ddfefdcb66873a726271b4a0                                          0.0s
 => => exporting config sha256:90101bb0dd957d16460dbbc77ba61db5d4cde415ba78f9b6da01a88825f76c3e                                            0.0s
 => => exporting attestation manifest sha256:a4e3f1df4ea7809547870e963812dfa858ecff6f53a7501aae265653a32999af                              0.0s
 => => exporting manifest list sha256:5054c2d8d304c1ed7fca87e2bf0ae91d2d2db0718dea4f717c97816147b492b7                                     0.0s
 => => naming to docker.io/library/s14-k8s-model-server:latest                                                                             0.0s
 => [model-server] resolving provenance for metadata file                                                                                  0.0s
[+] Building 2/2
 ✔ model-server  Built                                                                                                                     0.0s 
 ✔ web-server    Built  
```


*This builds container images for the model server and web server.*

✅ Output shows images being built successfully.

👉 Start the containers:

```python
docker compose up
```
```
[+] Running 8/8
 ✔ redis Pulled                                                                                                                             2.0s 
   ✔ 2493b15fc5cc Pull complete                                                                                                             0.2s 
   ✔ 9d28ed3f68f2 Pull complete                                                                                                             0.8s 
   ✔ ad0336d6fb70 Pull complete                                                                                                             0.3s 
   ✔ 4f4fb700ef54 Pull complete                                                                                                             0.3s 
   ✔ 731ddbd3b0ef Pull complete                                                                                                             0.3s 
   ✔ 266b5a3707a6 Pull complete                                                                                                             0.3s 
   ✔ 059ac13d98f2 Pull complete                                                                                                             0.4s 
[+] Running 4/4
 ✔ Network s14-k8s_default           Created                                                                                                0.0s 
 ✔ Container s14-k8s-redis-1         Created                                                                                                0.1s 
 ✔ Container s14-k8s-model-server-1  Created                                                                                               28.2s 
 ✔ Container s14-k8s-web-server-1    Created                                                                                                0.5s 
Attaching to model-server-1, redis-1, web-server-1
redis-1         | 1:C 06 Aug 2025 01:14:38.229 * oO0OoO0OoO0Oo Redis is starting oO0OoO0OoO0Oo
redis-1         | 1:C 06 Aug 2025 01:14:38.229 * Redis version=7.4.5, bits=64, commit=00000000, modified=0, pid=1, just started
redis-1         | 1:C 06 Aug 2025 01:14:38.229 * Configuration loaded
redis-1         | 1:M 06 Aug 2025 01:14:38.229 * monotonic clock: POSIX clock_gettime
redis-1         | 1:M 06 Aug 2025 01:14:38.230 * Running mode=standalone, port=6379.
redis-1         | 1:M 06 Aug 2025 01:14:38.230 * Server initialized
redis-1         | 1:M 06 Aug 2025 01:14:38.231 * Ready to accept connections tcp
web-server-1    | INFO:     Started server process [1]
web-server-1    | INFO:     Waiting for application startup.
web-server-1    | 2025-08-06 01:14:44,785 - WebServer - INFO - Initializing web server on host dc3e47a3bf69
web-server-1    | 2025-08-06 01:14:44,785 - WebServer - INFO - Creating Redis connection pool: host=redis, port=6379
web-server-1    | 2025-08-06 01:14:44,785 - WebServer - INFO - Web server initialization complete
web-server-1    | INFO:     Application startup complete.
web-server-1    | INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
model-server-1  | INFO:     Started server process [1]
model-server-1  | INFO:     Waiting for application startup.
model-server-1  | 2025-08-06 01:14:46,886 - ModelServer - INFO - Initializing model server on host c96204071edf
model-server-1  | 2025-08-06 01:14:46,888 - ModelServer - INFO - Using device: cpu
model-server-1  | 2025-08-06 01:14:46,888 - ModelServer - INFO - Loading ONNX model: resnet50_imagenet_1k_model.onnx
model-server-1  | 2025-08-06 01:14:47,159 - ModelServer - INFO - ONNX model loaded successfully
model-server-1  | 2025-08-06 01:14:47,159 - ModelServer - INFO - Loading local label file
model-server-1  | 2025-08-06 01:14:47,159 - ModelServer - INFO - Loaded 1000 categories
model-server-1  | 2025-08-06 01:14:47,159 - ModelServer - INFO - Creating Redis connection pool: host=redis, port=6379
model-server-1  | 2025-08-06 01:14:47,159 - ModelServer - INFO - Model server initialization complete
model-server-1  | INFO:     Application startup complete.
model-server-1  | INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
model-server-1  | INFO:     192.168.65.1:34390 - "GET /health HTTP/1.1" 200 OK

```
*This runs Redis, Model Server, and Web Server containers together.*

✅ Output: Redis starts, model server loads ONNX model, web server initializes.



## Step 2: Start Minikube (Local Kubernetes)

👉 Start cluster:

Initializes a local Kubernetes cluster using Docker as the virtualization driver. Minikube creates a virtual environment for testing Kubernetes deployments locally.

```python
 minikube start --driver=docker
```
```
😄  minikube v1.36.0 on Darwin 14.5 (arm64)
✨  Using the docker driver based on existing profile
👍  Starting "minikube" primary control-plane node in "minikube" cluster
🚜  Pulling base image v0.0.47 ...
🤷  docker "minikube" container is missing, will recreate.
🔥  Creating docker container (CPUs=2, Memory=4000MB) ...
🐳  Preparing Kubernetes v1.33.1 on Docker 28.1.1 ...
    ▪ Generating certificates and keys ...
    ▪ Booting up control plane ...
    ▪ Configuring RBAC rules ...
🔗  Configuring bridge CNI (Container Networking Interface) ...
🔎  Verifying Kubernetes components...
💡  After the addon is enabled, please run "minikube tunnel" and your ingress resources would be available at "127.0.0.1"
    ▪ Using image gcr.io/k8s-minikube/storage-provisioner:v5
    ▪ Using image docker.io/kubernetesui/dashboard:v2.7.0
    ▪ Using image registry.k8s.io/metrics-server/metrics-server:v0.7.2
    ▪ Using image registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.5.3
    ▪ Using image registry.k8s.io/ingress-nginx/controller:v1.12.2
    ▪ Using image docker.io/kubernetesui/metrics-scraper:v1.0.8
    ▪ Using image registry.k8s.io/ingress-nginx/kube-webhook-certgen:v1.5.3
🔎  Verifying ingress addon...
💡  Some dashboard features require the metrics-server addon. To enable all features please run:

	minikube addons enable metrics-server

🌟  Enabled addons: storage-provisioner, metrics-server, default-storageclass, dashboard, ingress
🏄  Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default

```
*This starts a Kubernetes cluster inside Docker with addons like ingress enabled.*

👉 Check Minikube container:

Lists running Docker containers, showing the Minikube container that's hosting the Kubernetes cluster.

```python
docker ps
```
```
CONTAINER ID   IMAGE                                 COMMAND                  CREATED         STATUS         PORTS                                                                                                                                  NAMES
73f06b46c537   gcr.io/k8s-minikube/kicbase:v0.0.47   "/usr/local/bin/entr…"   2 minutes ago   Up 2 minutes   127.0.0.1:50672->22/tcp, 127.0.0.1:50673->2376/tcp, 127.0.0.1:50675->5000/tcp, 127.0.0.1:50671->8443/tcp, 127.0.0.1:50674->32443/tcp   minikube
```

*Shows that Minikube itself is running as a container.*

## Step 3: Explore Cluster State

👉 Get all resources in default namespace:

Lists all resources in the default namespace. Initially shows only the Kubernetes API service.


```python
kubectl get all
```

```
NAME                 TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
service/kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   9m
```

👉 Get resources across all namespaces:

Lists all resources across all namespaces, showing system components like CoreDNS, etcd, and the Kubernetes dashboard.


```python
kubectl get all -A
```
```
NAMESPACE              NAME                                             READY   STATUS      RESTARTS   AGE
ingress-nginx          pod/ingress-nginx-admission-create-5p68t         0/1     Completed   0          3m32s
ingress-nginx          pod/ingress-nginx-admission-patch-x64lt          0/1     Completed   1          3m32s
ingress-nginx          pod/ingress-nginx-controller-67c5cb88f-bwz7p     1/1     Running     0          3m31s
kube-system            pod/coredns-674b8bbfcf-5m2qz                     1/1     Running     0          3m31s
kube-system            pod/coredns-674b8bbfcf-x6n9w                     1/1     Running     0          3m31s
kube-system            pod/etcd-minikube                                1/1     Running     0          3m36s
kube-system            pod/kube-apiserver-minikube                      1/1     Running     0          3m36s
kube-system            pod/kube-controller-manager-minikube             1/1     Running     0          3m36s
kube-system            pod/kube-proxy-7pvrw                             1/1     Running     0          3m32s
kube-system            pod/kube-scheduler-minikube                      1/1     Running     0          3m36s
kube-system            pod/metrics-server-7fbb699795-qp99h              1/1     Running     0          3m31s
kube-system            pod/storage-provisioner                          1/1     Running     0          3m35s
kubernetes-dashboard   pod/dashboard-metrics-scraper-5d59dccf9b-nnrsx   1/1     Running     0          3m31s
kubernetes-dashboard   pod/kubernetes-dashboard-7779f9b69b-9km7f        1/1     Running     0          3m31s

NAMESPACE              NAME                                         TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                      AGE
default                service/kubernetes                           ClusterIP   10.96.0.1        <none>        443/TCP                      3m38s
ingress-nginx          service/ingress-nginx-controller             NodePort    10.102.131.187   <none>        80:31580/TCP,443:32676/TCP   3m35s
ingress-nginx          service/ingress-nginx-controller-admission   ClusterIP   10.99.45.1       <none>        443/TCP                      3m35s
kube-system            service/kube-dns                             ClusterIP   10.96.0.10       <none>        53/UDP,53/TCP,9153/TCP       3m36s
kube-system            service/metrics-server                       ClusterIP   10.98.134.104    <none>        443/TCP                      3m35s
kubernetes-dashboard   service/dashboard-metrics-scraper            ClusterIP   10.97.104.142    <none>        8000/TCP                     3m35s
kubernetes-dashboard   service/kubernetes-dashboard                 ClusterIP   10.109.202.231   <none>        80/TCP                       3m35s

NAMESPACE     NAME                        DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR            AGE
kube-system   daemonset.apps/kube-proxy   1         1         1       1            1           kubernetes.io/os=linux   3m36s

NAMESPACE              NAME                                        READY   UP-TO-DATE   AVAILABLE   AGE
ingress-nginx          deployment.apps/ingress-nginx-controller    1/1     1            1           3m35s
kube-system            deployment.apps/coredns                     2/2     2            2           3m36s
kube-system            deployment.apps/metrics-server              1/1     1            1           3m35s
kubernetes-dashboard   deployment.apps/dashboard-metrics-scraper   1/1     1            1           3m35s
kubernetes-dashboard   deployment.apps/kubernetes-dashboard        1/1     1            1           3m35s

NAMESPACE              NAME                                                   DESIRED   CURRENT   READY   AGE
ingress-nginx          replicaset.apps/ingress-nginx-controller-67c5cb88f     1         1         1       3m32s
kube-system            replicaset.apps/coredns-674b8bbfcf                     2         2         2       3m32s
kube-system            replicaset.apps/metrics-server-7fbb699795              1         1         1       3m32s
kubernetes-dashboard   replicaset.apps/dashboard-metrics-scraper-5d59dccf9b   1         1         1       3m32s
kubernetes-dashboard   replicaset.apps/kubernetes-dashboard-7779f9b69b        1         1         1       3m32s

NAMESPACE       NAME                                       STATUS     COMPLETIONS   DURATION   AGE
ingress-nginx   job.batch/ingress-nginx-admission-create   Complete   1/1           8s         3m35s
ingress-nginx   job.batch/ingress-nginx-admission-patch    Complete   1/1           9s         3m35s

```
*This shows system pods like CoreDNS, metrics-server, ingress controller, etc.*

## Step 4: Use Minikube’s Docker Daemon

👉 Point Docker CLI to Minikube:

Configures your terminal to use Minikube's Docker daemon instead of your local one. This allows Kubernetes to access images you build locally without pushing to a registry.

```
eval $(minikube docker-env)
```

Now any `docker build` goes into Minikube’s environment.

👉 List available images:

Shows Docker images available in Minikube's Docker environment, including system images for Kubernetes components.


```python
 docker image ls
```

```
REPOSITORY                                           TAG        IMAGE ID       CREATED         SIZE
registry.k8s.io/kube-apiserver                       v1.33.1    9a2b7cf4f854   2 months ago    97.1MB
registry.k8s.io/kube-scheduler                       v1.33.1    014094c90caa   2 months ago    70.5MB
registry.k8s.io/kube-controller-manager              v1.33.1    674996a72aa5   2 months ago    90.5MB
registry.k8s.io/kube-proxy                           v1.33.1    3e58848989f5   2 months ago    99.7MB
registry.k8s.io/ingress-nginx/controller             <none>     84591f30b7f1   3 months ago    301MB
registry.k8s.io/ingress-nginx/kube-webhook-certgen   <none>     43a738c76a31   3 months ago    65MB
registry.k8s.io/etcd                                 3.5.21-0   31747a36ce71   4 months ago    146MB
registry.k8s.io/coredns/coredns                      v1.12.0    f72407be9e08   8 months ago    68.4MB
registry.k8s.io/metrics-server/metrics-server        <none>     5548a49bb60b   11 months ago   65.5MB
registry.k8s.io/pause                                3.10       afb61768ce38   14 months ago   514kB
kubernetesui/dashboard                               <none>     20b332c9a70d   2 years ago     244MB
kubernetesui/metrics-scraper                         <none>     a422e0e98235   3 years ago     42.3MB
gcr.io/k8s-minikube/storage-provisioner              v5         ba04bb24b957   4 years ago     29MB

```

*Verifies images inside Minikube’s Docker.*

## Step 5: Build Custom Images in Minikube

👉 Build the model server:

Builds the model-server Docker image specifically for the amd64 platform (for compatibility) and tags it as `model-server`.


```python
docker build --platform linux/amd64 -t model-server model-server
```

```
[+] Building 142.7s (14/14) FINISHED                                                                                                               docker:default
 => [internal] load build definition from Dockerfile                                                                                                         0.0s
 => => transferring dockerfile: 869B                                                                                                                         0.0s
 => [internal] load metadata for docker.io/library/python:3.12-slim                                                                                          1.2s
 => [auth] library/python:pull token for registry-1.docker.io                                                                                                0.0s
 => [internal] load .dockerignore                                                                                                                            0.0s
 => => transferring context: 2B                                                                                                                              0.0s
 => [1/7] FROM docker.io/library/python:3.12-slim@sha256:9c1d9ed7593f2552a4ea47362ec0d2ddf5923458a53d0c8e30edf8b398c94a31                                    2.4s
 => => resolve docker.io/library/python:3.12-slim@sha256:9c1d9ed7593f2552a4ea47362ec0d2ddf5923458a53d0c8e30edf8b398c94a31                                    0.3s
 => => sha256:bc31e3bfcc822213057d459f03536b30705f8708b53e166bbe3c3a53d41d8ae5 5.57kB / 5.57kB                                                               0.0s
 => => sha256:59e22667830bf04fb35e15ed9c70023e9d121719bb87f0db7f3159ee7c7e0b8d 28.23MB / 28.23MB                                                             0.7s
 => => sha256:4c665aba06d1c52829be84ca62e1030e27b8a3aa0f922666cbe74d24234ff227 3.51MB / 3.51MB                                                               0.6s
 => => sha256:e3586b415667d044c3e5c7c91023d29d7db667b73a8082068a1b7f36c1962c34 13.66MB / 13.66MB                                                             0.8s
 => => sha256:9c1d9ed7593f2552a4ea47362ec0d2ddf5923458a53d0c8e30edf8b398c94a31 9.13kB / 9.13kB                                                               0.0s
 => => sha256:c5247bc3787c809810386c289942bdf3b6e9437dc7436e1acfe7c822f446e3b5 1.75kB / 1.75kB                                                               0.0s
 => => sha256:f5cc5422ebcbbf01f9cd227d36de9dd7e133e1fc6d852f3b0c65260ab58f99f3 250B / 250B                                                                   0.8s
 => => extracting sha256:59e22667830bf04fb35e15ed9c70023e9d121719bb87f0db7f3159ee7c7e0b8d                                                                    0.8s
 => => extracting sha256:4c665aba06d1c52829be84ca62e1030e27b8a3aa0f922666cbe74d24234ff227                                                                    0.1s
 => => extracting sha256:e3586b415667d044c3e5c7c91023d29d7db667b73a8082068a1b7f36c1962c34                                                                    0.4s
 => => extracting sha256:f5cc5422ebcbbf01f9cd227d36de9dd7e133e1fc6d852f3b0c65260ab58f99f3                                                                    0.0s
 => [internal] load build context                                                                                                                            0.8s
 => => transferring context: 102.63MB                                                                                                                        0.8s
 => [auth] library/python:pull token for registry-1.docker.io                                                                                                0.0s
 => [2/7] WORKDIR /opt/src                                                                                                                                   0.3s
 => [3/7] COPY requirements.txt .                                                                                                                            0.0s
 => [4/7] RUN apt-get update && apt-get install -y --no-install-recommends         build-essential         libjpeg-dev         libpng-dev         && rm -r  13.6s
 => [5/7] RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1     -f https://download.pytorch.org/whl/cpu/torch_stable.html                    104.8s 
 => [6/7] RUN pip install --no-cache-dir -r requirements.txt     && rm -rf /root/.cache/pip                                                                  9.5s 
 => [7/7] COPY . .                                                                                                                                           0.1s 
 => exporting to image                                                                                                                                      10.8s 
 => => exporting layers                                                                                                                                     10.8s 
 => => writing image sha256:143c9762e8f8807cbdcc62fa4c1e1e1a536cf862a12ac093cf6a872c8d8c25ee                                                                 0.0s 
 => => naming to docker.io/library/model-server                                                                                                              0.0s 
                                                                      
```
👉 Build the web server:

Builds the web-server Docker image for the amd64 platform and tags it as `web-server`.

```python
docker build --platform linux/amd64 -t web-server web-server
```

```
[+] Building 21.4s (10/10) FINISHED                                                                                                                docker:default
 => [internal] load build definition from Dockerfile                                                                                                         0.0s
 => => transferring dockerfile: 251B                                                                                                                         0.0s
 => [internal] load metadata for docker.io/library/python:3.12                                                                                               0.8s
 => [internal] load .dockerignore                                                                                                                            0.0s
 => => transferring context: 2B                                                                                                                              0.0s
 => [1/5] FROM docker.io/library/python:3.12@sha256:645df645815f1403566b103b2a2bb07f6a01516bbb15078ed004e41d198ba194                                        10.4s
 => => resolve docker.io/library/python:3.12@sha256:645df645815f1403566b103b2a2bb07f6a01516bbb15078ed004e41d198ba194                                         0.0s
 => => sha256:b3d245705d9c006d10d35b888f58e42d57b803388d3e06d885a050520782daf7 6.46kB / 6.46kB                                                               0.0s
 => => sha256:c2e76af9483f2d17a3e370639403df2c53a3da1480d533116a8694cd91f15d5a 24.02MB / 24.02MB                                                             0.8s
 => => sha256:37f838b71c6b82c581b7543a313255b8c99c23cc9d96c1ad6f9f5f208c942553 64.40MB / 64.40MB                                                             2.0s
 => => sha256:645df645815f1403566b103b2a2bb07f6a01516bbb15078ed004e41d198ba194 9.08kB / 9.08kB                                                               0.0s
 => => sha256:6228f48a3d363d0d2e8a95f7b356857861a14550f86aa1456835b6a24b1a9a1f 2.33kB / 2.33kB                                                               0.0s
 => => sha256:ebed137c7c18cb1906fb8314eabc10611ddf49a281f8c1b5eab987a7137f749f 48.49MB / 48.49MB                                                             1.5s
 => => sha256:873a4c80287477653c01b20948fc34bb1bacf0f826bcc2ddc3bd2fe25b342d45 211.36MB / 211.36MB                                                           5.2s
 => => extracting sha256:ebed137c7c18cb1906fb8314eabc10611ddf49a281f8c1b5eab987a7137f749f                                                                    1.5s
 => => sha256:ffd592b5cf92171a279d48e22387a596c940352327e0c1973972a38f41042b19 6.16MB / 6.16MB                                                               2.0s
 => => sha256:34af509f605c821f54b5815829279b93b3832d6895c2b6e8ea2d0c19a1f25af2 25.64MB / 25.64MB                                                             3.0s
 => => sha256:b7db6ad1f83721354c4011a0fe015c2a140851c5e722bc65fbdf0e9c38172546 249B / 249B                                                                   2.2s
 => => extracting sha256:c2e76af9483f2d17a3e370639403df2c53a3da1480d533116a8694cd91f15d5a                                                                    0.4s
 => => extracting sha256:37f838b71c6b82c581b7543a313255b8c99c23cc9d96c1ad6f9f5f208c942553                                                                    1.7s
 => => extracting sha256:873a4c80287477653c01b20948fc34bb1bacf0f826bcc2ddc3bd2fe25b342d45                                                                    4.1s
 => => extracting sha256:ffd592b5cf92171a279d48e22387a596c940352327e0c1973972a38f41042b19                                                                    0.2s
 => => extracting sha256:34af509f605c821f54b5815829279b93b3832d6895c2b6e8ea2d0c19a1f25af2                                                                    0.5s
 => => extracting sha256:b7db6ad1f83721354c4011a0fe015c2a140851c5e722bc65fbdf0e9c38172546                                                                    0.0s
 => [internal] load build context                                                                                                                            0.0s
 => => transferring context: 4.45kB                                                                                                                          0.0s
 => [2/5] WORKDIR /opt/src                                                                                                                                   0.1s
 => [3/5] COPY requirements.txt .                                                                                                                            0.0s
 => [4/5] RUN pip install -r requirements.txt  && rm -rf /root/.cache/pip                                                                                    9.9s
 => [5/5] COPY server.py .                                                                                                                                   0.0s 
 => exporting to image                                                                                                                                       0.2s 
 => => exporting layers                                                                                                                                      0.2s 
 => => writing image sha256:b859aed237498cefba8eedae7c68da72c430a1366aa76318722f77b11432a537                                                                 0.0s 
 => => naming to docker.io/library/web-server                                                                                                                0.0s 
                                                         
```
*Both images are now available directly inside Minikube.*

## Step 6: Switch Back to Local Docker

👉 Exit Minikube’s Docker daemon:

Resets your terminal to use your local Docker daemon again instead of Minikube's.

```
eval $(minikube docker-env -u)
```

*Restores Docker CLI to your local machine.*

## Step 7: Apply Kubernetes Manifests

We define Deployments and Services in YAML files.

### model-server.deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: model-server:latest
        imagePullPolicy: Never
        resources:
          limits:
            memory: "2Gi"  
            cpu: "1000m"
        ports:
        - containerPort: 80
        env:
          - name: REDIS_HOST
            value: redis-db-service
          - name: REDIS_PORT
            value: "6379"
          - name: REDIS_PASSWORD
            value: <redis-password>
```

### model-server.service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server-service
spec:
  selector:
    app: model-server
  ports:
  - port: 80
    targetPort: 80 
```

### redis.deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
    name: redis-db
spec:
    replicas: 1
    selector:
        matchLabels:
            app: redis
            role: master
    template:
        metadata:
            labels:
                app: redis
                role: master
        spec:
            volumes:
                - name: redis-storage
                  persistentVolumeClaim:
                    claimName: redis-pvc
            containers:
                - name: redis-master
                  image: redis:7.4.1
                  resources:
                      limits:
                          cpu: 200m
                          memory: 200Mi
                  command:
                      - redis-server
                  args:
                      - --requirepass
                      - $(REDIS_PASSWORD)
                  ports:
                      - containerPort: 6379
                  volumeMounts:
                      - name: redis-storage
                        mountPath: /data
                  env:
                    - name: REDIS_PASSWORD
                      value: <redis-password>
```

### redis.service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
    name: redis-db-service
    labels:
        app: redis
        role: master
spec:
    ports:
        - port: 6379
          targetPort: 6379
    selector:
        app: redis
        role: master 
```

### web-server.deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-server
  labels:
    app: web-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: web-server
  template:
    metadata:
      labels:
        app: web-server
    spec:
      containers:
      - name: web-server
        image: web-server:latest
        imagePullPolicy: Never
        resources:
          limits:
            memory: "200Mi"
            cpu: "500m"
        ports:
          - containerPort: 80
        env:
          - name: REDIS_HOST
            value: redis-db-service
          - name: REDIS_PORT
            value: "6379"
          - name: REDIS_PASSWORD
            value: <redos-password>
          - name: MODEL_SERVER_URL
            value: "<http://model-server-service>" 
```


### web-server.service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-server-service
spec:
  selector:
    app: web-server
  ports:
  - port: 80
    targetPort: 80
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-server-ingress
spec:
  rules:
    - host: fastapi.localhost
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web-server-service
                port:
                  number: 80 
```

👉 Applying Kubernetes Manifests:

Creates all Kubernetes resources defined in YAML files in the current directory, including deployments, services, and ingress rules.

```python
kubectl apply -f .
```

```
deployment.apps/model-server created
service/model-server-service created
deployment.apps/redis-db created
service/redis-db-service created
deployment.apps/web-server created
service/web-server-service created
ingress.networking.k8s.io/web-server-ingress created
```

*This creates model-server, web-server, redis deployments, services, and ingress.*

👉 Checking Deployed Resources:

Lists all resources in the default namespace after deployment, showing the pods, services, deployments, and replicasets that were created.

```python
kubectl get all
```

```
NAME                                READY   STATUS    RESTARTS   AGE
pod/model-server-678c7f478c-dv7sv   1/1     Running   0          48s
pod/redis-db-6864b584bb-p28nt       1/1     Running   0          48s
pod/web-server-5d9db84fbf-fmc2p     1/1     Running   0          48s

NAME                           TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)    AGE
service/kubernetes             ClusterIP   10.96.0.1       <none>        443/TCP    15m
service/model-server-service   ClusterIP   10.107.44.85    <none>        80/TCP     48s
service/redis-db-service       ClusterIP   10.104.173.91   <none>        6379/TCP   48s
service/web-server-service     ClusterIP   10.102.56.29    <none>        80/TCP     48s

NAME                           READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/model-server   1/1     1            1           48s
deployment.apps/redis-db       1/1     1            1           48s
deployment.apps/web-server     1/1     1            1           48s

NAME                                      DESIRED   CURRENT   READY   AGE
replicaset.apps/model-server-678c7f478c   1         1         1       48s
replicaset.apps/redis-db-6864b584bb       1         1         1       48s
replicaset.apps/web-server-5d9db84fbf     1         1         1       48s
```

*Shows pods, deployments, and services are up.*

## Step 8: Expose with Ingress

👉 Checking Ingress Resources:

Lists all Ingress resources, showing the hostname (fastapi.localhost) that will route external traffic to the web-server service.

```python
kubectl get ing
```

```
NAME                 CLASS   HOSTS               ADDRESS        PORTS   AGE
web-server-ingress   nginx   fastapi.localhost   192.168.49.2   80      2m2s
```

*Lists the ingress mapping (e.g., `fastapi.localhost`).*

👉 Creating Network Tunnel for Local Access:

Creates a network tunnel that allows accessing services through their defined Ingress rules. This makes the application accessible at `http://fastapi.localhost` from your local machine.


```python
minikube tunnel
```

```
✅  Tunnel successfully started

📌  NOTE: Please do not close this terminal as this process must stay alive for the tunnel to be accessible ...

❗  The service/ingress web-server-ingress requires privileged ports to be exposed: [80 443]
🔑  sudo permission will be asked for it.
🏃  Starting tunnel for service web-server-ingress.
```

⚠️ Keep this terminal open for tunnel to stay active.

👉 Access app in browser:

```
http://fastapi.localhost/health
```
```
{
  "status": "healthy",
  "hostname": "web-server-5d9db84fbf-fmc2p",
  "redis": {
    "host": "redis-db-service",
    "port": "6379",
    "connected": true
  },
  "model_server": {
    "url": "http://model-server-service",
    "connected": true,
    "health": {
      "status": "healthy",
      "hostname": "model-server-678c7f478c-dv7sv",
      "model": "resnet50_imagenet_1k_model.onnx",
      "device": "cpu",
      "redis": {
        "host": "redis-db-service",
        "port": "6379",
        "connected": true
      }
    }
  }
}
```

*Returns JSON showing health status of web server, model server, and Redis.*

## Step 9: Check Namespaces

👉 Checking Kubernetes Namespaces:

Lists all namespaces in the Kubernetes cluster, showing system namespaces and any custom namespaces that have been created.

```python
kubectl get namespaces
NAME                   STATUS   AGE
default                Active   112m
ingress-nginx          Active   111m
kube-node-lease        Active   112m
kube-public            Active   112m
kube-system            Active   112m
kubernetes-dashboard   Active   111m
```

*Shows all namespaces like default, kube-system, ingress-nginx, etc.*

At the end of this workflow, you have successfully deployed a FastAPI application backed by a model server and Redis cache on Kubernetes. All services are running inside a Minikube cluster, exposed through an Ingress. Requests now flow from the browser to the Ingress, into the Web Server, then to the Model Server (with Redis providing caching along the way). This gives you a working, end-to-end, production-style setup entirely on your local machine.

Thank You for Reading!
