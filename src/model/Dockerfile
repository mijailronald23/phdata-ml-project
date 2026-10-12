FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app:/app/src

# Copy requirements from project root
COPY requirements.txt .

# Copy the artifact scripts and source modules
COPY create_model.py .
COPY create_imputer.py .
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt

# Symlinks keep artifact paths consistent across local and container execution
RUN ln -s /app/src/data /app/data && \
    ln -s /app/src/model /app/model

CMD ["python", "create_model.py"]