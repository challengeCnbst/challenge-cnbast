FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY challenge/ /app/challenge/
COPY model.pkl /app/

# Expose port 8080 (required by Cloud Run)
EXPOSE 8080

# Run the API with uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]