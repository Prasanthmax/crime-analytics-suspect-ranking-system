# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend
COPY app/frontend/package*.json ./
RUN npm ci

COPY app/frontend/ ./
# Inject the HF Space backend URL at build time (same-origin because
# FastAPI will serve the frontend too)
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build
# Output: /build/frontend/dist


# ── Stage 2: Python backend ───────────────────────────────────────────────────
FROM python:3.11-slim

# HuggingFace Spaces runs as user 1000 (non-root)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "fastapi[standard]" uvicorn[standard]

# Copy project source
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY app/backend/main.py ./app/backend/main.py

# Copy built frontend into a location FastAPI will serve as static files
COPY --from=frontend-builder /build/frontend/dist ./frontend_dist

# Patch main.py so FastAPI also serves the React build
# (mounts /frontend_dist at "/" after all API routes)
COPY app/backend/main.py ./app/backend/main.py

RUN chown -R appuser:appuser /app
USER appuser

# HuggingFace Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
