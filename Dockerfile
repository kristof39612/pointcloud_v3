FROM pytorch/pytorch:latest

WORKDIR /app

COPY train_sample.py .

RUN mkdir -p /trained_model

VOLUME ["/trained_model"]

CMD ["python3", "train_sample.py"]

