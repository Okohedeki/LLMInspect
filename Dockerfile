# The public Hub scanner: Unmask and Sweep as a web app (llminspect/public.py). No model is loaded and no GPU is
# needed; Hub repos are read by byte range. Runs as an unprivileged user (uid 1000) on port 7860.
#   docker build -t llminspect-public .
#   docker run -p 7860:7860 -e HF_TOKEN=hf_... llminspect-public
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements-public.txt .
RUN pip install --no-cache-dir -r requirements-public.txt

COPY --chown=user llminspect ./llminspect
COPY --chown=user ui ./ui
RUN mkdir -p /data && chown user /data
USER user
ENV HOME=/home/user LLMINSPECT_DATA=/data PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz')"
CMD ["uvicorn", "llminspect.public:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
