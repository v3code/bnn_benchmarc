FROM python:3.10
RUN sudo apt-get update
RUN sudo apt-get -y install curl
RUN !curl -sSL https://install.python-poetry.org | python -
COPY . .
RUN poetry install
ENTRYPOINT ["tail", "-f", "/dev/null"]