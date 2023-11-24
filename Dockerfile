FROM python:3.10.13

WORKDIR /tf

COPY . .
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y \
  wget \
  build-essential \
  python3-opencv \
  libopencv-dev

EXPOSE 6969

ENTRYPOINT [ "python3", "main.py" ]
