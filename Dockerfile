FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
CMD ["python", "spectral_solver.py", "--nx", "128", "--ny", "128", "--dt", "0.002", "--tend", "0.5"]
