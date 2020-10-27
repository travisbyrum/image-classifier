FROM tensorflow/tensorflow:2.3.1

WORKDIR /opt/bco

ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git python3-pip

COPY Pipfile ./

RUN pip3 install --upgrade pip && \
    pip3 install pipenv && \
    pipenv install --skip-lock --deploy --system && \
    rm -rf /var/lib/apt/lists/*

RUN addgroup --system bco && \
    useradd --system -g bco bco

COPY --chown=bco:bco docker-entrypoint.sh .
COPY --chown=bco:bco training.py .
COPY --chown=bco:bco validation.py .

USER bco

ENTRYPOINT ["/opt/bco/docker-entrypoint.sh"]