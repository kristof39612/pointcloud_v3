FROM pytorch/pytorch:latest

WORKDIR /app

COPY pytorch_pointnet/datasets.py /app/datasets.py
COPY pytorch_pointnet/model /app/model
COPY pytorch_pointnet/train.py /app/train.py
COPY pytorch_pointnet/requirements.txt /app/requirements.txt
COPY pytorch_pointnet/utils.py /app/utils.py
COPY pytorch_pointnet/shapenet_partanno_v0_final.tar.gz /app/shapenet_partanno_v0_final.tar.gz

RUN tar -xzf shapenet_partanno_v0_final.tar.gz
RUN rm shapenet_partanno_v0_final.tar.gz
RUN mkdir -p /trained_model

RUN pip install -r requirements.txt

COPY pytorch_pointnet/run_training.sh /app/run_training.sh
RUN chmod +x run_training.sh

CMD ["/bin/bash", "run_training.sh"]