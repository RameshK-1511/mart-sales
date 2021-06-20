FROM ubuntu:20.04

COPY . /app
WORKDIR /app

EXPOSE 5000
RUN apt-get update && apt-get install -y python3-pip
# RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt
ENTRYPOINT  ["python3"]

CMD ["app.py"]