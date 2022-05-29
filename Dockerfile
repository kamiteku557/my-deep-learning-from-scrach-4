FROM nikolaik/python-nodejs:python3.10-nodejs17
ARG DEBIAN_FRONTEND=noninteractive

# set timezone
RUN apt-get update && apt-get install -y \
    tzdata \
    xvfb \
    python-opengl \
    &&  ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    &&  apt-get clean \
    &&  rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Tokyo

# make python and jupyter enviroment
COPY requirements.txt /workspace/
COPY setup.py /workspace/

# install modules
RUN mkdir /workspace/src
RUN python -m pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt --no-cache-dir && \
    pip install -e /workspace/

RUN jupyter nbextension enable --py widgetsnbextension\
 && jupyter labextension install @jupyter-widgets/jupyterlab-manager
