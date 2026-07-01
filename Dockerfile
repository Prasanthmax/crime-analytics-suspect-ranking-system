# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend
COPY app/frontend/package*.json ./
RUN npm ci

COPY app/frontend/ ./
RUN npm run build
# Output: /build/frontend/dist


# ── Stage 2: Python backend ────────────────────────────────────────────────────
FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/           ./src/
COPY data/          ./data/
COPY models/        ./models/
COPY app/backend/main.py ./app/backend/main.py

# Copy built React app — absolute known path, set as env var for main.py
COPY --from=frontend-builder /build/frontend/dist /app/frontend_dist

RUN chown -R appuser:appuser /app
USER appuser

ENV PORT=7860
ENV FRONTEND_DIST=/app/frontend_dist
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
