# Delphi
# Version: 1.0
FROM python:3

# Install Python and Package Libraries
RUN apt-get update && apt-get upgrade -y && apt-get autoremove && apt-get autoclean
RUN apt-get install -y \
    libffi-dev \
    libssl-dev \
    default-libmysqlclient-dev \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    libfreetype6-dev \
    zlib1g-dev \
    net-tools \
    vim \
    build-essential \
    python3-dev

RUN pip install pystan 
RUN pip install fbprophet

# Project Files and Settings
ARG PROJECT=delphi
ARG PROJECT_DIR=/var/www/${PROJECT}
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR
COPY Pipfile Pipfile.lock ./

RUN pip install -U pipenv
RUN pipenv install --system

# Server
STOPSIGNAL SIGINT
CMD ["./docker-entrypoint.sh"]
