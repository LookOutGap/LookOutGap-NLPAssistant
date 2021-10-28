FROM continuumio/miniconda3 as build

RUN mkdir -p /opt/app
WORKDIR /opt/app

ADD dev_environment.yml .
ADD setup.py .
ADD README.md .
ADD conftest.py .

ENV PYTHON