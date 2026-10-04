FROM agrigorev/zoomcamp-model:2025
WORKDIR /code
RUN pip install scikit-learn==1.6.1 fastapi uvicorn
COPY serve_docker.py .
EXPOSE 8000
CMD ["uvicorn", "serve_docker:app", "--host", "0.0.0.0", "--port", "8000"]