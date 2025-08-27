# 🚀 Deployment Guide - Information Retrieval System

A comprehensive guide to deploy your unified Academic Search Engine + Document Classifier system.

---

## 📋 Quick Deployment Options

### 🎯 **Recommended for Beginners: Railway**
- ✅ **Free tier available**
- ✅ **Automatic deployments from GitHub**
- ✅ **Built-in database support**
- ✅ **Custom domain support**

### 🔧 **For Advanced Users: DigitalOcean/Linode**
- ✅ **Full control over server**
- ✅ **Cost-effective for production**
- ✅ **Scalable resources**

---

## 🛠️ Pre-Deployment Preparation

### 1. **Create Production Configuration**

```bash
# Create production requirements file
cd ui
cp requirements.txt requirements-prod.txt
```

Add to `requirements-prod.txt`:
```txt
# Production extras
gunicorn==21.2.0
uvloop==0.19.0  # Better performance on Linux
httptools==0.6.1  # Better HTTP parsing
```

### 2. **Environment Variables Setup**

Create `.env` file in the root directory:
```env
# Application Settings
APP_NAME="Information Retrieval by Smaran"
APP_VERSION="1.0.0"
ENVIRONMENT="production"

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Database (if using external DB)
DATABASE_URL=sqlite:///./publications.db

# Security
SECRET_KEY=your-secret-key-here-change-this-in-production

# CORS Settings (if needed for API access)
ALLOWED_ORIGINS=["https://yourdomain.com"]
```

### 3. **Create Production Startup Script**

Create `start.sh`:
```bash
#!/bin/bash
cd ui
gunicorn main:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${WORKERS:-4} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keepalive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile - \
    --error-logfile -
```

Make it executable:
```bash
chmod +x start.sh
```

---

## 🌐 Hosting Option 1: Railway (Recommended)

### **Step 1: Prepare for Railway**

Create `railway.toml` in root:
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "./start.sh"
healthcheckPath = "/api/health"
healthcheckTimeout = 300
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3

[env]
PORT = { default = "8000" }
ENVIRONMENT = { default = "production" }
```

Create `Procfile`:
```
web: ./start.sh
```

### **Step 2: Deploy to Railway**

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial deployment setup"
   git branch -M main
   git remote add origin https://github.com/yourusername/information-retrieval-system.git
   git push -u origin main
   ```

2. **Deploy via Railway:**
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect and deploy!

3. **Access your app:**
   - Railway provides a URL like: `https://your-app.railway.app`

### **Step 3: Configure Domain (Optional)**
- Go to Railway dashboard → Settings → Domains
- Add custom domain: `your-domain.com`
- Update DNS records as instructed

---

## ☁️ Hosting Option 2: DigitalOcean Droplet

### **Step 1: Create Droplet**
```bash
# Create a $5/month Ubuntu droplet
# SSH into your server
ssh root@your-server-ip
```

### **Step 2: Server Setup**
```bash
# Update system
apt update && apt upgrade -y

# Install Python and dependencies
apt install python3 python3-pip python3-venv nginx git -y

# Create app user
useradd -m -s /bin/bash appuser
su - appuser

# Clone your repository
git clone https://github.com/yourusername/information-retrieval-system.git
cd information-retrieval-system
```

### **Step 3: Setup Application**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
cd ui
pip install -r requirements-prod.txt
cd ..

# Test the application
./start.sh
```

### **Step 4: Configure Nginx**

Create `/etc/nginx/sites-available/ir-system`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
}
```

Enable the site:
```bash
ln -s /etc/nginx/sites-available/ir-system /etc/nginx/sites-enabled/
systemctl restart nginx
```

### **Step 5: Setup Systemd Service**

Create `/etc/systemd/system/ir-system.service`:
```ini
[Unit]
Description=Information Retrieval System
After=network.target

[Service]
User=appuser
Group=appuser
WorkingDirectory=/home/appuser/information-retrieval-system
Environment="PATH=/home/appuser/information-retrieval-system/venv/bin"
ExecStart=/home/appuser/information-retrieval-system/start.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

Start the service:
```bash
systemctl enable ir-system
systemctl start ir-system
systemctl status ir-system
```

---

## 🔒 Hosting Option 3: Vercel (Static + Serverless)

**Note**: Requires splitting into frontend (static) and backend (serverless functions)

### **Frontend Deployment**

1. **Extract static files:**
   ```bash
   mkdir vercel-frontend
   cp -r ui/templates/* vercel-frontend/
   cp -r ui/static/* vercel-frontend/
   ```

2. **Create vercel.json:**
   ```json
   {
     "builds": [
       {
         "src": "index.html",
         "use": "@vercel/static"
       }
     ],
     "routes": [
       {
         "src": "/api/(.*)",
         "dest": "https://your-backend-api.com/api/$1"
       },
       {
         "src": "/(.*)",
         "dest": "/$1"
       }
     ]
   }
   ```

3. **Deploy:**
   ```bash
   npx vercel --prod
   ```

---

## 📦 Hosting Option 4: Docker Deployment

### **Create Dockerfile**

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start application
CMD ["./start.sh"]
```

### **Create docker-compose.yml**

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - PORT=8000
    volumes:
      - ./data:/app/data  # For persistent data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl  # For SSL certificates
    depends_on:
      - web
    restart: unless-stopped
```

### **Deploy with Docker**

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Update deployment
docker-compose pull && docker-compose up -d
```

---

## 🔧 Production Optimizations

### **1. Enable Compression**

Add to main.py:
```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### **2. Add Rate Limiting**

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/search")
@limiter.limit("10/minute")  # 10 requests per minute
async def search_documents(request: Request, search_request: SearchRequest):
    # ... existing code
```

### **3. Add Logging**

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = datetime.now() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time}")
    return response
```

### **4. Database Optimization**

For production with many users, consider:
```python
# Use PostgreSQL instead of SQLite
# pip install asyncpg databases[postgresql]

DATABASE_URL = "postgresql://user:password@localhost/dbname"
```

---

## 📊 Monitoring & Maintenance

### **1. Health Monitoring**

```python
import psutil
from datetime import datetime

@app.get("/api/health")
async def detailed_health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "search_engine": search_engine is not None,
            "classifier": classifier is not None,
            "database": True,  # Add actual DB check
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
```

### **2. Backup Strategy**

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_${DATE}.tar.gz" \
    task-1/publications.db \
    task-2/*.txt \
    logs/ \
    config/
    
# Upload to cloud storage (optional)
# aws s3 cp "backup_${DATE}.tar.gz" s3://your-backup-bucket/
```

---

## 🚦 Deployment Checklist

### **Pre-Deployment:**
- [ ] Test all functionality locally
- [ ] Update dependencies to latest stable versions
- [ ] Set up environment variables
- [ ] Configure production database
- [ ] Set up logging and monitoring
- [ ] Prepare backup strategy

### **Deployment:**
- [ ] Choose hosting platform
- [ ] Configure domain and SSL
- [ ] Set up continuous deployment
- [ ] Configure load balancing (if needed)
- [ ] Test in production environment

### **Post-Deployment:**
- [ ] Monitor application logs
- [ ] Set up uptime monitoring
- [ ] Configure automated backups
- [ ] Document maintenance procedures
- [ ] Plan scaling strategy

---

## 💡 Quick Start Commands

### **Railway (Easiest):**
```bash
# Push to GitHub, then deploy via Railway dashboard
git push origin main
```

### **Digital Ocean:**
```bash
# One-time server setup
./deploy-scripts/setup-server.sh
./deploy-scripts/deploy.sh
```

### **Docker:**
```bash
# Local production test
docker-compose up -d
```

### **Manual Server:**
```bash
# Start production server
./start.sh
```

---

## 🆘 Troubleshooting

### **Common Issues:**

1. **Port already in use:**
   ```bash
   sudo lsof -ti:8000 | xargs sudo kill -9
   ```

2. **Permission denied:**
   ```bash
   chmod +x start.sh
   sudo chown -R appuser:appuser /path/to/app
   ```

3. **Database not found:**
   ```bash
   # Ensure database files are in correct location
   ls -la task-1/publications.db
   ls -la task-2/*.txt
   ```

4. **Memory issues:**
   ```bash
   # Reduce workers in start.sh
   --workers 2  # Instead of 4
   ```

---

🎉 **Your Information Retrieval System is now ready for the world!** 

Choose the deployment option that best fits your needs and budget. Railway is recommended for beginners, while DigitalOcean/Linode offers more control for advanced users.