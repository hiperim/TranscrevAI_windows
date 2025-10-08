# TranscrevAI - Nginx + SSL Deployment Guide
**Complete Production Deployment with Let's Encrypt SSL**

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [System Preparation](#system-preparation)
3. [Nginx Installation & Configuration](#nginx-installation--configuration)
4. [SSL Certificate Setup (Let's Encrypt)](#ssl-certificate-setup-lets-encrypt)
5. [Systemd Service Configuration](#systemd-service-configuration)
6. [Security Headers & Best Practices](#security-headers--best-practices)
7. [Testing & Verification](#testing--verification)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

**Required:**
- Ubuntu 20.04+ or Debian 11+ server
- Domain name pointing to your server IP (e.g., `transcrevai.example.com`)
- Root or sudo access
- Python 3.11+ installed
- TranscrevAI app cloned to `/opt/transcrevai`

**Verify Domain DNS:**
```bash
# Check if your domain resolves to server IP
dig +short transcrevai.example.com
# Should return your server's public IP
```

---

## System Preparation

### 1. Update System & Install Dependencies

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y nginx certbot python3-certbot-nginx \
    python3-pip python3-venv git curl ufw

# Verify installations
nginx -v
certbot --version
python3 --version
```

### 2. Firewall Configuration

```bash
# Enable UFW firewall
sudo ufw allow 22/tcp       # SSH
sudo ufw allow 80/tcp       # HTTP (for Let's Encrypt validation)
sudo ufw allow 443/tcp      # HTTPS
sudo ufw enable

# Verify rules
sudo ufw status
```

### 3. Setup Application Directory

```bash
# Create application directory
sudo mkdir -p /opt/transcrevai
sudo chown -R $USER:$USER /opt/transcrevai

# Clone or copy your TranscrevAI application
cd /opt/transcrevai
# (If not already present, git clone your repo here)

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p /opt/transcrevai/data/temp
mkdir -p /opt/transcrevai/data/models_cache
mkdir -p /opt/transcrevai/logs

# Set permissions
chmod -R 755 /opt/transcrevai
```

---

## Nginx Installation & Configuration

### 1. Remove Default Configuration

```bash
# Backup and remove default site
sudo mv /etc/nginx/sites-enabled/default /etc/nginx/sites-available/default.bak
```

### 2. Create TranscrevAI Nginx Configuration

**File:** `/etc/nginx/sites-available/transcrevai`

```bash
sudo nano /etc/nginx/sites-available/transcrevai
```

**Configuration (HTTP-only, before SSL):**

```nginx
# TranscrevAI - Initial HTTP Configuration (Pre-SSL)
# This will be upgraded to HTTPS after certbot setup

upstream transcrevai_backend {
    server 127.0.0.1:8000 fail_timeout=0;
    keepalive 32;
}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=2r/s;

server {
    listen 80;
    listen [::]:80;

    server_name transcrevai.example.com;  # REPLACE WITH YOUR DOMAIN

    # Client upload size limit (100MB for audio files)
    client_max_body_size 100M;
    client_body_timeout 300s;

    # Logging
    access_log /var/log/nginx/transcrevai_access.log;
    error_log /var/log/nginx/transcrevai_error.log warn;

    # Root location for Let's Encrypt validation
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # Redirect all other traffic to HTTPS (will be added after SSL setup)
    location / {
        return 301 https://$server_name$request_uri;
    }
}
```

### 3. Enable Configuration & Test

```bash
# Create symbolic link to enable site
sudo ln -s /etc/nginx/sites-available/transcrevai /etc/nginx/sites-enabled/

# Test nginx configuration
sudo nginx -t

# If test passes, reload nginx
sudo systemctl reload nginx
```

---

## SSL Certificate Setup (Let's Encrypt)

### 1. Obtain SSL Certificate with Certbot

```bash
# Replace transcrevai.example.com with your actual domain
sudo certbot --nginx -d transcrevai.example.com

# Follow prompts:
# - Enter email for urgent renewal and security notices
# - Agree to Terms of Service (Y)
# - Choose whether to redirect HTTP to HTTPS (recommended: 2)
```

**Certbot will automatically:**
- Obtain SSL certificate from Let's Encrypt
- Modify your nginx configuration to use SSL
- Setup automatic certificate renewal

### 2. Update Nginx Configuration for Production

**File:** `/etc/nginx/sites-available/transcrevai` (Updated with Full SSL + WebSocket Support)

```bash
sudo nano /etc/nginx/sites-available/transcrevai
```

**Complete Production Configuration:**

```nginx
# TranscrevAI - Production Nginx Configuration with SSL
# WebSocket-enabled reverse proxy for FastAPI application

upstream transcrevai_backend {
    server 127.0.0.1:8000 fail_timeout=0;
    keepalive 32;
}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=2r/s;
limit_req_zone $binary_remote_addr zone=ws_limit:10m rate=5r/s;

# HTTP - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name transcrevai.example.com;  # REPLACE WITH YOUR DOMAIN

    # Let's Encrypt validation
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # Redirect all HTTP to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS - Main Application
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name transcrevai.example.com;  # REPLACE WITH YOUR DOMAIN

    # SSL Configuration (Certbot managed)
    ssl_certificate /etc/letsencrypt/live/transcrevai.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/transcrevai.example.com/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # Enhanced SSL Security
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Client upload limits
    client_max_body_size 100M;
    client_body_timeout 300s;
    client_header_timeout 60s;

    # Logging
    access_log /var/log/nginx/transcrevai_access.log;
    error_log /var/log/nginx/transcrevai_error.log warn;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;

    # Root location - Main application
    location / {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://transcrevai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # WebSocket endpoints - Critical for live audio recording
    location ~ ^/ws/live/(.+)$ {
        limit_req zone=ws_limit burst=10 nodelay;

        proxy_pass http://transcrevai_backend;
        proxy_http_version 1.1;

        # WebSocket upgrade headers
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Standard proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Extended timeouts for long-running connections
        proxy_connect_timeout 7200s;
        proxy_send_timeout 7200s;
        proxy_read_timeout 7200s;

        # Disable buffering for real-time communication
        proxy_buffering off;
    }

    location ~ ^/ws/(.+)$ {
        limit_req zone=ws_limit burst=10 nodelay;

        proxy_pass http://transcrevai_backend;
        proxy_http_version 1.1;

        # WebSocket upgrade headers
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Standard proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Extended timeouts
        proxy_connect_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;

        # Disable buffering
        proxy_buffering off;
    }

    # File upload endpoint - Higher rate limit tolerance
    location /upload {
        limit_req zone=upload_limit burst=5 nodelay;

        proxy_pass http://transcrevai_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Extended timeouts for file uploads
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;

        # Large body support
        client_max_body_size 100M;
        client_body_buffer_size 128k;
    }

    # Static files - Cached and optimized
    location /static/ {
        alias /opt/transcrevai/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # Health check endpoint (optional)
    location /health {
        proxy_pass http://transcrevai_backend;
        access_log off;
    }

    # Deny access to hidden files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
}
```

**Replace `transcrevai.example.com` with your actual domain in 3 locations:**
1. Line 16: `server_name`
2. Line 29: `server_name`
3. Lines 35-36: SSL certificate paths

### 3. Test and Reload Nginx

```bash
# Test configuration
sudo nginx -t

# If successful, reload
sudo systemctl reload nginx
```

### 4. Verify Auto-Renewal

```bash
# Test certificate renewal (dry run)
sudo certbot renew --dry-run

# Check renewal timer status
sudo systemctl status certbot.timer
```

**Certificates will auto-renew every 60 days via systemd timer.**

---

## Systemd Service Configuration

### 1. Create TranscrevAI Systemd Service

**File:** `/etc/systemd/system/transcrevai.service`

```bash
sudo nano /etc/systemd/system/transcrevai.service
```

**Service Configuration:**

```ini
[Unit]
Description=TranscrevAI FastAPI Application
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/transcrevai

# Environment variables
Environment="PATH=/opt/transcrevai/venv/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/opt/transcrevai/.env

# Execute application with Uvicorn
ExecStart=/opt/transcrevai/venv/bin/uvicorn main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 2 \
    --log-level info \
    --access-log \
    --use-colors

# Restart policy
Restart=always
RestartSec=10
StartLimitInterval=200
StartLimitBurst=5

# Resource limits (adjust based on your hardware)
MemoryMax=4G
CPUQuota=200%

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/transcrevai/data /opt/transcrevai/logs /opt/transcrevai/temp
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictRealtime=true

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=transcrevai

[Install]
WantedBy=multi-user.target
```

### 2. Configure Environment Variables

```bash
# Create .env file if it doesn't exist
sudo nano /opt/transcrevai/.env
```

**Example `.env` file:**

```bash
# TranscrevAI Production Environment
ENVIRONMENT=production
LOG_LEVEL=info
HOST=127.0.0.1
PORT=8000

# API Keys (if using external services)
# OPENAI_API_KEY=your_key_here

# Hardware settings
USE_GPU=true  # Set to false if no GPU
COMPUTE_TYPE=int8  # or float16 for GPU

# Model cache
MODELS_CACHE_DIR=/opt/transcrevai/data/models_cache
```

### 3. Set Correct Permissions

```bash
# Set ownership to www-data
sudo chown -R www-data:www-data /opt/transcrevai

# Set proper file permissions
sudo chmod 640 /opt/transcrevai/.env
sudo chmod -R 755 /opt/transcrevai/static
sudo chmod -R 755 /opt/transcrevai/templates
sudo chmod -R 775 /opt/transcrevai/data
sudo chmod -R 775 /opt/transcrevai/logs
sudo chmod -R 775 /opt/transcrevai/temp
```

### 4. Enable and Start Service

```bash
# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable transcrevai

# Start service
sudo systemctl start transcrevai

# Check status
sudo systemctl status transcrevai
```

### 5. Service Management Commands

```bash
# View logs
sudo journalctl -u transcrevai -f

# Restart service
sudo systemctl restart transcrevai

# Stop service
sudo systemctl stop transcrevai

# Check if service is running
sudo systemctl is-active transcrevai
```

---

## Security Headers & Best Practices

### 1. Additional Nginx Security Configuration

**File:** `/etc/nginx/nginx.conf` (Global settings)

```bash
sudo nano /etc/nginx/nginx.conf
```

**Add to `http {}` block:**

```nginx
http {
    # ... existing configuration ...

    # Hide nginx version
    server_tokens off;

    # Buffer overflow protection
    client_body_buffer_size 1k;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 16k;

    # Request timeout protection
    client_body_timeout 10;
    client_header_timeout 10;
    keepalive_timeout 5 5;
    send_timeout 10;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript
               application/x-javascript application/xml+rss
               application/json application/javascript;

    # Connection limits
    limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
    limit_conn conn_limit_per_ip 10;
}
```

### 2. Fail2Ban Configuration (Optional but Recommended)

```bash
# Install Fail2Ban
sudo apt install fail2ban -y

# Create custom jail for nginx
sudo nano /etc/fail2ban/jail.d/nginx-custom.conf
```

**Fail2Ban Configuration:**

```ini
[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/transcrevai_error.log

[nginx-noscript]
enabled = true
port = http,https
logpath = /var/log/nginx/transcrevai_access.log
maxretry = 3

[nginx-badbots]
enabled = true
port = http,https
logpath = /var/log/nginx/transcrevai_access.log
maxretry = 2

[nginx-noproxy]
enabled = true
port = http,https
logpath = /var/log/nginx/transcrevai_access.log
maxretry = 2
```

```bash
# Restart Fail2Ban
sudo systemctl restart fail2ban
sudo systemctl enable fail2ban

# Check status
sudo fail2ban-client status
```

---

## Testing & Verification

### 1. Test SSL Configuration

```bash
# Test SSL certificate
curl -I https://transcrevai.example.com

# Check SSL Labs rating
# Visit: https://www.ssllabs.com/ssltest/analyze.html?d=transcrevai.example.com
```

### 2. Test Application Endpoints

```bash
# Test main page
curl -I https://transcrevai.example.com/

# Test health check (if implemented)
curl https://transcrevai.example.com/health

# Test static files
curl -I https://transcrevai.example.com/static/css/style.css
```

### 3. Test WebSocket Connection

**Create test HTML file:** `ws_test.html`

```html
<!DOCTYPE html>
<html>
<head><title>WebSocket Test</title></head>
<body>
<h1>TranscrevAI WebSocket Test</h1>
<div id="output"></div>
<script>
    const sessionId = 'test-' + Date.now();
    const ws = new WebSocket('wss://transcrevai.example.com/ws/' + sessionId);

    ws.onopen = () => {
        document.getElementById('output').innerHTML += '<p>✓ Connected</p>';
    };

    ws.onmessage = (event) => {
        document.getElementById('output').innerHTML += '<p>Message: ' + event.data + '</p>';
    };

    ws.onerror = (error) => {
        document.getElementById('output').innerHTML += '<p>✗ Error: ' + error + '</p>';
    };
</script>
</body>
</html>
```

**Open in browser and verify connection.**

### 4. Performance Testing

```bash
# Install Apache Bench
sudo apt install apache2-utils -y

# Test concurrent connections
ab -n 100 -c 10 https://transcrevai.example.com/

# Test upload endpoint (requires test file)
# ab -n 10 -c 2 -p test_audio.wav -T 'multipart/form-data' https://transcrevai.example.com/upload
```

### 5. Monitor Application Logs

```bash
# Real-time application logs
sudo journalctl -u transcrevai -f

# Real-time nginx access logs
sudo tail -f /var/log/nginx/transcrevai_access.log

# Real-time nginx error logs
sudo tail -f /var/log/nginx/transcrevai_error.log

# Check for errors in last hour
sudo journalctl -u transcrevai --since "1 hour ago" | grep -i error
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. 502 Bad Gateway Error

**Cause:** FastAPI app not running or not accessible.

```bash
# Check if app is running
sudo systemctl status transcrevai

# Check if port 8000 is listening
sudo netstat -tlnp | grep 8000

# Check application logs
sudo journalctl -u transcrevai -n 50

# Restart application
sudo systemctl restart transcrevai
```

#### 2. WebSocket Connection Fails

**Cause:** Missing WebSocket upgrade headers or timeout issues.

```bash
# Verify nginx WebSocket configuration
sudo nginx -T | grep -A 20 "location.*ws"

# Test WebSocket from command line
wscat -c wss://transcrevai.example.com/ws/test-123

# Check nginx error logs
sudo tail -f /var/log/nginx/transcrevai_error.log
```

#### 3. SSL Certificate Issues

**Cause:** Certificate expired or renewal failed.

```bash
# Check certificate expiry
sudo certbot certificates

# Force renewal
sudo certbot renew --force-renewal

# Verify nginx SSL config
sudo nginx -t
```

#### 4. Upload Fails (413 Request Entity Too Large)

**Cause:** `client_max_body_size` too small.

```bash
# Edit nginx config
sudo nano /etc/nginx/sites-available/transcrevai

# Increase limit:
# client_max_body_size 100M;

# Reload nginx
sudo nginx -t && sudo systemctl reload nginx
```

#### 5. Application Not Starting

**Cause:** Permission issues or missing dependencies.

```bash
# Check permissions
ls -la /opt/transcrevai

# Verify virtual environment
source /opt/transcrevai/venv/bin/activate
python -c "import fastapi; print(fastapi.__version__)"

# Check systemd service logs
sudo journalctl -u transcrevai -xe

# Try running manually to see errors
cd /opt/transcrevai
source venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

#### 6. High Memory Usage

**Cause:** Large model loading or memory leaks.

```bash
# Monitor memory usage
htop

# Check application memory
sudo systemctl status transcrevai

# Adjust systemd memory limits
sudo nano /etc/systemd/system/transcrevai.service
# Change: MemoryMax=4G

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart transcrevai
```

### Useful Diagnostic Commands

```bash
# Check all listening ports
sudo netstat -tlnp

# Check nginx configuration validity
sudo nginx -t

# View full nginx configuration
sudo nginx -T

# Check SSL certificate details
openssl s_client -connect transcrevai.example.com:443 -servername transcrevai.example.com

# Test DNS resolution
dig transcrevai.example.com

# Check firewall rules
sudo ufw status verbose

# Monitor system resources
htop  # or: top, glances

# Check disk space
df -h

# View all systemd services
sudo systemctl list-units --type=service --state=running
```

---

## Maintenance & Monitoring

### Daily Checks

```bash
# Check application status
sudo systemctl status transcrevai

# Check nginx status
sudo systemctl status nginx

# Review error logs
sudo journalctl -u transcrevai --since today | grep -i error
```

### Weekly Tasks

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Check SSL certificate expiry
sudo certbot certificates

# Rotate logs if needed
sudo logrotate -f /etc/logrotate.d/nginx

# Review disk space usage
df -h
du -sh /opt/transcrevai/data/*
```

### Monthly Tasks

```bash
# Update Python dependencies
cd /opt/transcrevai
source venv/bin/activate
pip list --outdated
# Review and update critical packages

# Review fail2ban banned IPs
sudo fail2ban-client status nginx-http-auth

# Performance audit
# Check response times, memory usage trends
```

---

## Quick Reference Commands

```bash
# Restart everything
sudo systemctl restart transcrevai nginx

# View live logs
sudo journalctl -u transcrevai -f

# Test configuration
sudo nginx -t

# Check service status
sudo systemctl status transcrevai nginx certbot.timer

# Renew SSL certificate manually
sudo certbot renew

# Enable/disable service
sudo systemctl enable transcrevai
sudo systemctl disable transcrevai
```

---

## Additional Security Recommendations

1. **Setup Regular Backups**
```bash
# Backup critical directories
sudo tar -czf /backup/transcrevai_$(date +%Y%m%d).tar.gz \
    /opt/transcrevai \
    /etc/nginx/sites-available/transcrevai \
    /etc/systemd/system/transcrevai.service
```

2. **Enable Automatic Security Updates**
```bash
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

3. **Monitor with External Services** (Optional)
- Setup UptimeRobot or similar for uptime monitoring
- Configure Sentry for error tracking
- Use Cloudflare for additional DDoS protection

4. **Regular Security Audits**
```bash
# Check for vulnerable packages
pip list --outdated | grep -i security

# Scan for open ports
sudo nmap -sT -O localhost
```

---

## Conclusion

Your TranscrevAI application is now deployed with:
- ✅ Nginx reverse proxy with HTTP/2
- ✅ Let's Encrypt SSL certificate with auto-renewal
- ✅ WebSocket support for live audio streaming
- ✅ Systemd service for auto-start and process management
- ✅ Security headers and rate limiting
- ✅ Production-grade configuration and monitoring

**Access your application:**
- **HTTPS:** `https://transcrevai.example.com`
- **WebSocket (Live):** `wss://transcrevai.example.com/ws/live/{session_id}`
- **WebSocket (Upload):** `wss://transcrevai.example.com/ws/{session_id}`

**Support & Documentation:**
- Nginx: https://nginx.org/en/docs/
- Let's Encrypt: https://letsencrypt.org/docs/
- FastAPI: https://fastapi.tiangolo.com/deployment/

---

**Author:** TranscrevAI Deployment Guide
**Last Updated:** 2025-10-07
**Version:** 1.0
