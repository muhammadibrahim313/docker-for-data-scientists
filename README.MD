# Iris Classifier API (Dockerized)
[![Build and Deploy ML API](https://github.com/muhammadibrahim313/docker-for-data-scientists/actions/workflows/deploy.yml/badge.svg)](https://github.com/muhammadibrahim313/docker-for-data-scientists/actions/workflows/deploy.yml)
A containerized ML API that predicts Iris flower species.

## Live Demo

**API Endpoint:** https://YOUR_RENDER_URL.onrender.com

## Tech Stack

- Python
- FastAPI
- scikit-learn
- Docker
- Render

## Run Locally

```bash
docker pull YOUR_DOCKERHUB_USERNAME/iris-classifier:latest
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/iris-classifier:latest
```

Then open http://localhost:8000

## Author

Muhammad Ibrahim Qasmi
