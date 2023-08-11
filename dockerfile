# syntax=docker/dockerfile:1

FROM python:3.8-alpine

WORKDIR /app

RUN pip3 install pytest

COPY . .

CMD [ "python3", "cotu.py"]