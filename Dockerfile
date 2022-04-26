FROM nikolaik/python-nodejs:python3.10-nodejs17
ARG DEBIAN_FRONTEND=noninteractive

# set timezone
RUN apt-get update && apt-get install -y \
    tzdata \
    &&  ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    &&  apt-get clean \
    &&  rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo


# make python and jupyter enviroment
COPY requirements.txt /code/
COPY setup.py /code/

WORKDIR /code

# external modules
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

RUN jupyter nbextension enable --py widgetsnbextension \
 && jupyter labextension install @jupyter-widgets/jupyterlab-manager