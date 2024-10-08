FROM pytorch/pytorch:latest

WORKDIR /app

COPY train_sample.py .
COPY pytorch_pointnet/shapenet_partanno_v0.tar.gz .
RUN tar -xzf shapenet_partanno_v0.tar.gz

RUN mkdir -p /trained_model

VOLUME ["/trained_model"]

CMD ["python3", "train_sample.py"]

