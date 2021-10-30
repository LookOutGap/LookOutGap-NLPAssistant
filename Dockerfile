FROM continuumio/miniconda3 as build

RUN mkdir -p /opt/app
WORKDIR /opt/app

ADD dev_environment.yml .
ADD setup.py .
ADD README.md .
ADD conftest.py .

ENV PYTHONUNBUFFERED=1

RUN /bin/bash -c "echo '. ~/anaconda/etc/profile.d/conda.sh' >> ~/.bash_profile && conda env create -f dev_environment.yml"

COPY nlprov ./nlprov

RUN /bin/bash -c 'source acti