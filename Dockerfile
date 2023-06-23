FROM python:3.9

WORKDIR /IR

COPY . .

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

EXPOSE 8000:8000