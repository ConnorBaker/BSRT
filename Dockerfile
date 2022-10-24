ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY . /BSRT
WORKDIR /BSRT

# Install redis
RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
RUN apt update && apt install -y lsb-release
RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
RUN apt update && apt install -y redis

# Install dependencies
RUN pip --no-cache-dir install pip==22.3 --upgrade
RUN pip --no-cache-dir install -e .
