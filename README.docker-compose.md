# Panduan Docker Compose untuk AI Quiz Generator

Panduan ini menjelaskan cara menjalankan dan mengkonfigurasi AI Quiz Generator menggunakan Docker Compose.

## 🐳 Prasyarat

- Docker terinstal
- Docker Compose terinstal
- API Key Google Gemini AI

## 🚀 Konfigurasi Docker Compose

File `docker-compose.yml` yang digunakan:

```yaml
services:
  quiz-api:
    build: .
    ports:
      - "3001:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - API_WORKERS=1
      - LOG_LEVEL=INFO
      - RATE_LIMIT_PER_MINUTE=60
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
```

## 📝 Penjelasan Konfigurasi

### Port Mapping
- `3001:8000`: Aplikasi berjalan di port 8000 dalam container dan diakses melalui port 3001 di host

### Environment Variables
- `API_HOST`: Host untuk menjalankan aplikasi (default: 0.0.0.0)
- `API_PORT`: Port internal aplikasi (default: 8000)
- `API_WORKERS`: Jumlah worker uvicorn (default: 1)
- `LOG_LEVEL`: Level logging (default: INFO)
- `RATE_LIMIT_PER_MINUTE`: Batas rate request per menit (default: 60)

### Volume
- `./logs:/app/logs`: Mount direktori logs untuk menyimpan log aplikasi

### Healthcheck
- Memeriksa kesehatan aplikasi setiap 30 detik
- Timeout 10 detik
- Maksimal 3 kali retry

## 🚀 Cara Menjalankan

1. **Menjalankan dalam mode detached:**
```bash
docker-compose up -d
```

2. **Menjalankan dengan log:**
```bash
docker-compose up
```

3. **Menjalankan dan rebuild image:**
```bash
docker-compose up -d --build
```

## 🛠️ Perintah Docker Compose yang Berguna

```bash
# Melihat status container
docker-compose ps

# Melihat log
docker-compose logs -f

# Menghentikan semua container
docker-compose down

# Restart container
docker-compose restart

# Menghapus container dan volume
docker-compose down -v

# Membangun ulang image
docker-compose build
```

## 🔧 Konfigurasi Tambahan

### Menambahkan Environment Variables

Buat file `.env`:
```env
GEMINI_API_KEY=your-api-key-here
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
LOG_LEVEL=INFO
RATE_LIMIT_PER_MINUTE=60
```

### Mengubah Rate Limit

Untuk mengubah rate limit, edit environment variable `RATE_LIMIT_PER_MINUTE` di `docker-compose.yml` atau file `.env`.

## 🔍 Troubleshooting

1. **Container tidak bisa start**
   - Periksa log: `docker-compose logs`
   - Pastikan port 3001 tidak digunakan
   - Pastikan semua environment variables sudah benar

2. **Rate limit error**
   - Periksa log untuk melihat rate limit yang aktif
   - Sesuaikan `RATE_LIMIT_PER_MINUTE` jika diperlukan

3. **Masalah koneksi**
   - Pastikan port 3001 sudah dibuka
   - Coba akses: `curl http://localhost:3001/health`

## 📚 Akses API

Setelah container berjalan, API dapat diakses di:
- Swagger UI: `http://localhost:3001/docs`
- ReDoc: `http://localhost:3001/redoc`
- Health Check: `http://localhost:3001/health`

## 🔐 Keamanan

- Jangan menyimpan API key di file konfigurasi
- Gunakan file `.env` untuk menyimpan secret
- Batasi akses ke port 3001 jika diperlukan
- Gunakan network Docker yang terisolasi

## 🚀 Production Deployment

Untuk deployment di production:

1. Buat file `.env.prod`:
```env
GEMINI_API_KEY=your-api-key-here
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=WARNING
RATE_LIMIT_PER_MINUTE=100
```

2. Jalankan dengan environment production:
```bash
docker-compose --env-file .env.prod up -d
```

## 🤝 Kontribusi

Silakan buat pull request untuk kontribusi terkait Docker Compose. Untuk perubahan besar, buka issue terlebih dahulu untuk mendiskusikan perubahan yang diinginkan. 