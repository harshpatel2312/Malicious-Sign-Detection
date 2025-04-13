FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install Flask requests plotly
EXPOSE 5001
CMD ["python3", "app.py"]