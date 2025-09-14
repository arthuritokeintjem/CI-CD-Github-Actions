FROM python:3.11-slim
WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY app.py ./app.py

# copy model yang disiapkan oleh job CI (folder 'model/' dibuat di workflow)
COPY model/model.pkl /app/model/model.pkl

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]