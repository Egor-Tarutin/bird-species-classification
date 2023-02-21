FROM python:3.9

RUN apt update && apt install -y python3-pip
RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .


CMD [ "python3", "app/inference.py", "--image_path", "app/inference/0.jpg" ]