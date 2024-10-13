FROM pytorch/pytorch:latest

WORKDIR /app

COPY pytorch_pointnet/ /app/
RUN tar -xzf shapenet_partanno_v0.tar.gz

RUN mkdir -p /trained_model

RUN pip install -r requirements.txt
