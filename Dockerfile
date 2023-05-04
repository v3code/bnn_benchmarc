FROM python:3.10
RUN apt-get update
RUN apt-get -y install curl
WORKDIR /code/
RUN curl -sSL https://install.python-poetry.org | python -
COPY . .
ENTRYPOINT ["tail", "-f", "/dev/null"]