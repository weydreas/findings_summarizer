FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY medical_summary.py ./
COPY evidence_pipeline.py ./
COPY semantic_pipeline.py ./
COPY run_test.py ./
COPY run_suite.py ./
COPY examples/ ./examples/
COPY tests/ ./tests/
USER 65534:65534
CMD ["python", "medical_summary.py"]
