# using ubuntu LTS version
#FROM ubuntu:22.04 AS builder-image
#FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04 AS builder-image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS builder-image

# avoid stuck build due to user prompt
ARG DEBIAN_FRONTEND=noninteractive

RUN set -xe \
	&& apt-get update \
	&& apt-get install --no-install-recommends -y python3.10 python3.10-dev python3.10-venv python3-pip build-essential \
	&& apt-get clean && rm -rf /var/lib/apt/lists/*


# create and activate virtual environment
# using final folder name to avoid path issues with packages
RUN python3.10 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

# install requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools==65.5.1 wheel==0.38.4
RUN pip install --no-cache-dir -r requirements.txt


#FROM ubuntu:22.04 AS runner-image
#FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04 AS runner-image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runner-image
RUN apt-get update && apt-get install --no-install-recommends -y python3.10 python3-venv && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home myuser
COPY --from=builder-image /home/myuser/venv /home/myuser/venv

RUN mkdir /home/myuser/code
RUN chown myuser /home/myuser/code




WORKDIR /home/myuser/code
COPY . .

#EXPOSE 5000

# make sure all messages always reach console
ENV PYTHONUNBUFFERED=1

# activate virtual environment
ENV VIRTUAL_ENV=/home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

# /dev/shm is mapped to shared memory and should be used for gunicorn heartbeat
#CMD [ "python", './main.py --alg="qmix"']
CMD [ "python", "src/main.py"]
