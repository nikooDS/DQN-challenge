FROM tensorflow/tensorflow:2.1.0-gpu

# Install python 3.7
ARG PYTHON=python3.7

ENV LANG C.UTF-8

RUN apt-get update
RUN apt-get install wget -y

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y ${PYTHON}

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN ${PYTHON} get-pip.py
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3
RUN ln -sf /usr/local/bin/pip /usr/local/bin/pip3

RUN pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN apt-get install python3.7-dev python3-dev python-dev -y

RUN ln -f $(which ${PYTHON}) /usr/local/bin/python

ARG TF_PACKAGE=tensorflow==2.1.0
RUN pip install --upgrade ${TF_PACKAGE}

# Create a dir in container to copy req and python files
RUN mkdir -p /DQNchallenge
WORKDIR /DQNchallenge

COPY DQNdemo.py /DQNchallenge
COPY requirements.txt /DQNchallenge/requirements.txt

# install requirements
RUN pip install -r /DQNchallenge/requirements.txt

# Invoke DQNdemo.py.
CMD ["python", "/DQNchallenge/DQNdemo.py"]