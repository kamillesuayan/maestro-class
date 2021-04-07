FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

RUN pip3 install .

ENV PYTHONPATH /app

CMD [ "python3" ]