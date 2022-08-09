FROM python:3.8-slim

WORKDIR /app

RUN apt-get clean \
    && apt-get -y update

RUN apt-get -y install \
    python3-dev \
    build-essential

RUN apt-get install -y python3-opencv

RUN pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

RUN make -C ./adaptis/inference/cython_utils

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT [ "python" ]

CMD [ "train_toy_v2.py", "--batch-size=1", "--workers=0", "--gpus=0", \
       "--dataset-path=app/custom_dataset" ]
