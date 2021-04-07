FROM python:3.7-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ENV PYTHONPATH /app

COPY . .

CMD [ "python3", "attack.py" ]