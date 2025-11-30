# INSTALLATION & DEPLOYMENT GUIDE
# Stock Signalling Bot v4.0 (Institutional Grade)
# ================================================

## TABLE OF CONTENTS

1. Prerequisites
2. Local Installation (Development)
3. Configuration Setup
4. API Credentials
5. Testing & Validation
6. Production Deployment (VPS)
7. Docker Deployment
8. Systemd Service Setup
9. Monitoring & Logs
10. Troubleshooting

---

## 1. PREREQUISITES

### Minimum Requirements
- Python 3.8+
- pip (Python package manager)
- Git (for version control)
- ~500MB disk space
- Internet connection (for API access)

### Recommended Setup
- Ubuntu 20.04 LTS or CentOS 8+
- 1GB RAM minimum
- 2GB RAM recommended (for multiple indicators)
- VPS/Cloud server for 24/7 operation

### Check Python Version
```bash
python --version
# Should output Python 3.8 or higher
```

---

## 2. LOCAL INSTALLATION (DEVELOPMENT)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/stock-signalling-bot.git
cd stock-signalling-bot
```

### Step 2: Create Virtual Environment
```bash
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Upgrade pip
```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation
```bash
python -c "import pandas; import numpy; print('✓ Core dependencies installed')"
```

---

## 3. CONFIGURATION SETUP

### Step 1: Create .env File
```bash
# Copy example to .env
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

### Step 2: Fill Required Fields

**REQUIRED - Get from Upstox:**
```
UPSTOX_ACCESS_TOKEN=xxx
UPSTOX_API_KEY=xxx
UPSTOX_API_SECRET=xxx
```

**REQUIRED - Get from Telegram:**
```
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
```

### Step 3: Validate Configuration
```bash
python config.py
# Should output: ✓ Configuration loaded and validated successfully
```

---

## 4. API CREDENTIALS

### 4A. Upstox Setup

1. **Create Account**
   - Visit https://upstox.com
   - Sign up with PAN/Aadhar
   - Complete KYC verification

2. **Generate API Credentials**
   - Login to https://upstox.com/account/api
   - Click "Create New App"
   - Enter app name: "Stock Signal Bot"
   - Accept terms and create
   - Copy API Key and Secret

3. **Get Access Token**
   - Use Upstox OAuth flow or
   - Use refresh token method
   - Store access token in .env

4. **Whitelist IP (if needed)**
   - In API settings, add your VPS IP
   - Prevents unauthorized access

### 4B. Telegram Bot Setup

1. **Create Bot**
   - Open Telegram
   - Search for @BotFather
   - Send `/start`
   - Send `/newbot`
   - Follow prompts to create bot
   - Copy bot token

2. **Get Chat ID**
   - Open Telegram
   - Search for @userinfobot
   - Start conversation
   - Your chat ID will be displayed
   - Or check bot in your account settings

3. **Add Bot to Chat**
   - Create private group (or use direct message)
   - Add bot to group
   - Bot can now send alerts

### 4C. Test Credentials
```bash
# Test Upstox connection
python -c "from config import get_config; c=get_config(); print('✓ Upstox configured')"

# Test Telegram connection
python telegram_notifier.py
# Should output: ✓ Telegram connection successful
```

---

## 5. TESTING & VALIDATION

### Step 1: Run Configuration Test
```bash
python config.py
# Validates all parameters and shows warnings
```

### Step 2: Test Analyzer
```bash
python market_analyzer.py
# Loads sample data and runs analysis
```

### Step 3: Test Validator
```bash
python signal_validator.py
# Tests 4-stage validation pipeline
```

### Step 4: Backtest Mode
```bash
BOT_MODE=BACKTEST python main.py
# Analyzes all stocks once, exports results
```

### Step 5: Paper Trading Mode
```bash
BOT_MODE=PAPER python main.py
# Fetches live data, generates signals
# No real execution
```

### Step 6: Adhoc Validation
```bash
BOT_MODE=ADHOC python main.py
# Interactive mode for manual validation
# Commands: [d]ashboard [v]alidate [h]istory [s]tats [q]uit
```

---

## 6. PRODUCTION DEPLOYMENT (VPS)

### Step 1: Choose VPS Provider
- AWS EC2 (t2.micro free tier)
- DigitalOcean ($5/month)
- Linode ($5/month)
- Azure VM (pay-as-you-go)

### Step 2: Initial VPS Setup

```bash
# SSH into server
ssh root@your_vps_ip

# Update system
apt update && apt upgrade -y

# Install Python and dependencies
apt install -y python3 python3-pip python3-venv git

# Create app directory
mkdir -p /opt/bot
cd /opt/bot

# Clone repository
git clone https://github.com/yourusername/stock-signalling-bot.git .
```

### Step 3: Install Application

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Create .env file
nano .env

# Add your credentials:
# UPSTOX_ACCESS_TOKEN=xxx
# TELEGRAM_BOT_TOKEN=xxx
# TELEGRAM_CHAT_ID=xxx
# BOT_MODE=LIVE

# Save and exit
```

### Step 5: Test Deployment

```bash
# Activate venv
source venv/bin/activate

# Test configuration
python config.py

# Run backtest
BOT_MODE=BACKTEST python main.py

# Check logs
tail -f logs/bot_*.log
```

### Step 6: Start Bot

```bash
# Activate venv
source venv/bin/activate

# Run in production
nohup python main.py > bot.log 2>&1 &

# Or use screen for interactive control
screen -S bot
python main.py
# Ctrl+A then D to detach
```

---

## 7. DOCKER DEPLOYMENT

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create logs directory
RUN mkdir -p logs

# Run bot
CMD ["python", "main.py"]
```

### Step 2: Create .dockerignore

```
.git
.env
.gitignore
__pycache__
*.pyc
.pytest_cache
logs/
signals_export.json
signals_history.json
venv/
```

### Step 3: Build Image

```bash
docker build -t stock-bot:latest .
```

### Step 4: Run Container

```bash
# Create .env file first
cp .env.example .env
# Edit .env with your credentials

# Run container
docker run -d \
  --name stock-bot \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/signals_export.json:/app/signals_export.json \
  stock-bot:latest

# View logs
docker logs -f stock-bot

# Stop container
docker stop stock-bot
```

### Step 5: Docker Compose (Optional)

```yaml
version: '3.8'

services:
  bot:
    build: .
    container_name: stock-signal-bot
    environment:
      - BOT_MODE=LIVE
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./signals_export.json:/app/signals_export.json
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

Run with: `docker-compose up -d`

---

## 8. SYSTEMD SERVICE SETUP

### Step 1: Create Service File

```bash
sudo nano /etc/systemd/system/stock-bot.service
```

### Step 2: Add Configuration

```ini
[Unit]
Description=Stock Signalling Bot
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=trading
WorkingDirectory=/opt/bot
Environment="PATH=/opt/bot/venv/bin"
EnvironmentFile=/opt/bot/.env
ExecStart=/opt/bot/venv/bin/python main.py
Restart=always
RestartSec=10

# Log output
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Step 3: Create Bot User

```bash
# Create non-root user
sudo useradd -m -s /bin/bash trading

# Set permissions
sudo chown -R trading:trading /opt/bot
sudo chmod 700 /opt/bot/.env
```

### Step 4: Enable Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable stock-bot

# Start service
sudo systemctl start stock-bot

# Check status
sudo systemctl status stock-bot

# View logs
sudo journalctl -u stock-bot -f
```

### Step 5: Service Commands

```bash
# Start
sudo systemctl start stock-bot

# Stop
sudo systemctl stop stock-bot

# Restart
sudo systemctl restart stock-bot

# View recent logs
sudo journalctl -u stock-bot -n 100

# Follow logs in real-time
sudo journalctl -u stock-bot -f
```

---

## 9. MONITORING & LOGS

### Log Files Location

```
logs/
├── bot_20251130.log      # Daily main log
├── signals_20251130.json # Signals with metadata
└── performance_20251130.json # Daily statistics
```

### View Logs

```bash
# Last 50 lines
tail -50 logs/bot_*.log

# Follow in real-time
tail -f logs/bot_*.log

# Search for errors
grep ERROR logs/bot_*.log

# Count signals generated
grep "Signal validated" logs/bot_*.log | wc -l
```

### Log Rotation Setup

```bash
# Create logrotate config
sudo nano /etc/logrotate.d/stock-bot
```

Add:
```
/opt/bot/logs/bot_*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 trading trading
    sharedscripts
}
```

### Monitoring Commands

```bash
# Check bot is running
ps aux | grep main.py

# Check memory usage
free -h

# Check disk usage
df -h

# Check network connections
netstat -an | grep ESTABLISHED

# Monitor in real-time
top -u trading
```

---

## 10. TROUBLESHOOTING

### Issue: "API client not initialized"

**Cause:** Missing or invalid UPSTOX_ACCESS_TOKEN

**Solution:**
1. Check .env file has token: `cat .env | grep UPSTOX`
2. Verify token is valid: `python config.py`
3. Regenerate token from Upstox dashboard
4. Update .env and restart bot

### Issue: "Telegram rate limit exceeded"

**Cause:** Too many messages sent too quickly

**Solution:**
1. Check rate limiting in config (default: 1 sec between messages)
2. Reduce signals per day: `BOT_VALIDATION_MIN_RRR=2.0`
3. Messages queue automatically, wait 5 minutes
4. Check Telegram bot's rate limits

### Issue: "No data for stock"

**Cause:** Invalid symbol or no data available

**Solution:**
1. Verify symbol format: `NSE_EQ|INE009A01021`
2. Check Upstox API is working: `python market_analyzer.py`
3. Check date - market might be closed
4. Verify internet connection

### Issue: "Configuration validation failed"

**Cause:** Invalid parameter values

**Solution:**
1. Run: `python config.py`
2. Check error messages
3. Common issues:
   - min_rrr < 1.0 (must be ≥ 1.0)
   - max_daily_loss < 1% (too low)
   - RSI settings invalid (oversold must be < overbought)
4. Fix in config.py or .env
5. Restart bot

### Issue: "Bot crashes after market close"

**Cause:** Scheduler trying to run after hours

**Solution:**
1. Bot runs on NSE hours (09:15-15:30 IST)
2. It should gracefully idle outside hours
3. Check logs: `tail -f logs/bot_*.log`
4. If crashes, check for errors and fix

### Issue: "Memory usage keeps growing"

**Cause:** Memory leak in data structures

**Solution:**
1. Check bot has been running long (>1 week)
2. Normal: grows to ~200-300MB then stable
3. If exceeds 1GB: restart bot
4. Set auto-restart in systemd (already configured)
5. Monitor: `watch free -h`

### Issue: "Telegram not sending notifications"

**Cause:** Bot token invalid or chat ID wrong

**Solution:**
1. Test: `python telegram_notifier.py`
2. Verify token: Open Telegram, `/start` with @BotFather
3. Verify chat ID: Open @userinfobot in Telegram
4. Check bot is added to your chat/group
5. Check signal tier is HIGH or better (MEDIUM+ to send)

### Debug Mode

Enable debug logging:
```bash
BOT_LOG_LEVEL=DEBUG python main.py
# Much more verbose output
# Useful for troubleshooting
```

---

## PRODUCTION CHECKLIST

Before going live:

- [ ] Python 3.8+ installed
- [ ] requirements.txt installed
- [ ] .env file created with all credentials
- [ ] Configuration validates: `python config.py`
- [ ] Backtest runs successfully
- [ ] Paper trading validated (1-2 weeks)
- [ ] Telegram notifications working
- [ ] VPS/Server ready (if not local)
- [ ] Systemd service configured (if on Linux)
- [ ] Log monitoring setup
- [ ] Backup of credentials (.env)
- [ ] Firewall rules configured
- [ ] DNS/IP whitelisted with Upstox
- [ ] Error alerting configured
- [ ] Daily monitoring schedule set

---

## SUPPORT & RESOURCES

**Official Documentation:**
- Upstox API: https://upstox.com/developer
- Python Schedule: https://schedule.readthedocs.io
- Telegram Bot API: https://core.telegram.org/bots/api

**Troubleshooting:**
1. Check logs: `tail -f logs/bot_*.log`
2. Run: `python config.py` to validate
3. Test components individually
4. Search GitHub issues
5. Review code comments in each module

**Performance Tips:**
- Reduce stocks to monitor (5-10 optimal)
- Use BACKTEST mode for strategy testing
- Monitor log file size (auto-rotated)
- Restart daily for fresh connections
- Update API token monthly

---

## SECURITY BEST PRACTICES

1. **Credential Security**
   - Never commit .env to git
   - Use .gitignore to exclude .env
   - Rotate API tokens monthly
   - Use strong Telegram bot token

2. **VPS Security**
   - Use SSH keys (not password)
   - Configure firewall (UFW on Ubuntu)
   - Keep system updated
   - Monitor logs for suspicious activity
   - Run as non-root user (trading)

3. **API Security**
   - Whitelist VPS IP in Upstox
   - Use environment variables only
   - Don't log sensitive data
   - Validate all inputs
   - Use HTTPS for API calls

4. **Backup Strategy**
   - Daily backup of .env
   - Weekly backup of logs
   - Monthly backup of signals history
   - Store credentials securely (encrypted)

---

**Version:** 4.0.0 (Institutional Grade)
**Last Updated:** 2025-11-30
**Author:** rahulreddyallu
