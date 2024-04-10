# Dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

EXPOSE 5959

COPY . /app

CMD python3 app/app.py
