# 🚀 Bucket Brigade Deployment Guide

This guide covers deploying the Bucket Brigade platform to various environments.

## 📋 Prerequisites

- Python 3.9+ with uv package manager
- Node.js 18+ with pnpm
- Docker (optional, for containerized deployment)
- Web server (nginx, Apache, etc.) for production

## 🏗️ Build Process

### Frontend Build

```bash
# Install dependencies
pnpm install

# Build for production
pnpm run build

# Preview production build locally
pnpm run preview
```

The built files will be in `web/dist/` ready for deployment.

### Python Package Build (Optional)

```bash
# Build wheel for Python package
cd bucket-brigade-core
python -m build

# Install locally for testing
pip install dist/*.whl
```

## 🌐 Deployment Options

> 📝 **Note**: The backend API deployment option described below is for future implementation. The current version can be deployed as a static site only (Option 2). See API.md for details on the planned backend API.

### Option 1: Static Site + API (Future - Not Yet Implemented)

Deploy the frontend as static files and run the Python API separately.

#### Frontend Deployment

**Netlify**
```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy from web directory
cd web
netlify deploy --prod --dir=dist
```

**Vercel**
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy from web directory
cd web
vercel --prod
```

**GitHub Pages**
```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages
on:
  push:
    branches: [ main ]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - run: npm install -g pnpm
      - run: pnpm install
      - run: pnpm run build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./web/dist
```

#### Backend API Deployment

**Railway**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy Python API
railway login
railway init
railway up
```

**Render**
```yaml
# render.yaml
services:
  - type: web
    name: bucket-brigade-api
    runtime: python3
    buildCommand: pip install -r requirements.txt
    startCommand: python -m bucket_brigade.api
```

**Heroku**
```yaml
# Procfile
web: python -m bucket_brigade.api

# requirements.txt
flask
bucket-brigade-core
# ... other dependencies
```

### Option 2: Docker Deployment

#### Single Container (Development)

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install Node.js for building frontend
RUN apt-get update && apt-get install -y nodejs npm
RUN npm install -g pnpm

# Copy and setup backend
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install -e .

# Build frontend
WORKDIR /app/web
RUN pnpm install && pnpm run build

# Expose port and run
EXPOSE 8000
WORKDIR /app
CMD ["python", "-m", "bucket_brigade.api"]
```

#### Multi-Container (Production)

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data

  web:
    build: ./web
    ports:
      - "3000:3000"
    depends_on:
      - api
```

### Option 3: Cloud Platform Deployment

#### AWS

**API Gateway + Lambda (Serverless)**
```python
# lambda_function.py
from bucket_brigade.api import app

def lambda_handler(event, context):
    return app(event, context)
```

**ECS Fargate**
```yaml
# task-definition.json
{
  "family": "bucket-brigade",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "your-registry/bucket-brigade:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000
        }
      ]
    }
  ]
}
```

#### Google Cloud

**Cloud Run**
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/bucket-brigade
gcloud run deploy --image gcr.io/PROJECT-ID/bucket-brigade --platform managed
```

**App Engine**
```yaml
# app.yaml
runtime: python311
entrypoint: python -m bucket_brigade.api

handlers:
- url: /.*
  script: auto
```

## ⚙️ Configuration

### Environment Variables

```bash
# API Configuration
BUCKET_BRIGADE_PORT=8000
BUCKET_BRIGADE_HOST=0.0.0.0
BUCKET_BRIGADE_DEBUG=false

# Database (if using persistent storage)
DATABASE_URL=postgresql://user:pass@localhost/bucket_brigade

# CORS (for web deployment)
CORS_ORIGINS=https://your-frontend-domain.com
```

### Runtime Configuration

```python
# config.py
import os

class Config:
    PORT = int(os.getenv('BUCKET_BRIGADE_PORT', 8000))
    HOST = os.getenv('BUCKET_BRIGADE_HOST', 'localhost')
    DEBUG = os.getenv('BUCKET_BRIGADE_DEBUG', 'false').lower() == 'true'
    DATABASE_URL = os.getenv('DATABASE_URL')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
```

## 🔒 Security Considerations

### API Security
- Use HTTPS in production
- Implement rate limiting
- Validate all user inputs
- Use secure headers (CSP, HSTS, etc.)

### Agent Security
- Sandbox agent execution
- Limit resource usage per agent
- Validate agent code statically
- Implement timeouts for agent actions

### Data Security
- Encrypt sensitive data at rest
- Use secure database connections
- Implement proper authentication if needed
- Regular security updates

## 📊 Monitoring & Logging

### Health Checks

```python
# health.py
from flask import Blueprint, jsonify
import psutil
import time

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'memory_usage': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent()
    })
```

### Logging Configuration

```python
# logging.py
import logging
import os

def setup_logging():
    level = logging.INFO if os.getenv('ENVIRONMENT') == 'production' else logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
```

## 🔄 Updates & Rollbacks

### Blue-Green Deployment

```bash
# Deploy new version
kubectl set image deployment/bucket-brigade api=new-version

# Verify health
curl https://api.example.com/health

# Switch traffic (if using load balancer)
kubectl patch service bucket-brigade -p '{"spec":{"selector":{"version":"v2"}}}'

# Rollback if needed
kubectl rollout undo deployment/bucket-brigade
```

### Rolling Updates

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      containers:
      - name: api
        image: your-registry/bucket-brigade:latest
```

## 🧪 Testing Deployments

### Pre-deployment Checks

```bash
# Run full test suite
npm run test

# Test API endpoints
curl -X POST http://localhost:8000/api/games \
  -H "Content-Type: application/json" \
  -d '{"scenario": "trivial_cooperation", "agents": ["firefighter", "coordinator"]}'

# Test frontend build
cd web && pnpm run build && pnpm run preview
```

### Production Validation

```bash
# Health check
curl https://your-api.com/health

# Load test
ab -n 1000 -c 10 https://your-api.com/api/games

# Monitor logs
tail -f /var/log/bucket-brigade/app.log
```

## 🚨 Troubleshooting

### Common Issues

**Frontend not loading**
- Check build process completed successfully
- Verify static file serving configuration
- Check CORS settings

**API returning 500 errors**
- Check application logs
- Verify database connectivity
- Confirm environment variables

**Performance issues**
- Monitor resource usage
- Check database query performance
- Consider caching strategies

**Agent timeouts**
- Increase timeout limits
- Optimize agent code
- Check for infinite loops

## 📈 Scaling

### Horizontal Scaling

```yaml
# kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bucket-brigade-api
spec:
  replicas: 3  # Scale as needed
  selector:
    matchLabels:
      app: bucket-brigade-api
  template:
    metadata:
      labels:
        app: bucket-brigade-api
    spec:
      containers:
      - name: api
        image: your-registry/bucket-brigade:latest
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

### Database Scaling

- Use connection pooling
- Implement read replicas for analytics
- Consider sharding for large datasets
- Use caching (Redis/Memcached)

---

For questions about deployment, check the [CONTRIBUTING.md](CONTRIBUTING.md) guide or create an issue.
