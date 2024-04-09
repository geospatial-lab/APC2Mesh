ARG IMAGE_NAME
FROM ${IMAGE_NAME}:version0

WORKDIR /app
COPY . /app

RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /a2p
USER appuser

CMD ["python3", "main.py"]