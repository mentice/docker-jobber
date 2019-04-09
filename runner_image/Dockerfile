#   Jobber Runner
#   Manages launching and checkpointing Docker Jobber containers
#
#   From this directory (use version from src/__init__.py and setup.py)
#       docker build --force-rm -t mentice/jobber-runner -t mentice/jobber-runner:0.3.7 -f Dockerfile ..
#
#	Push to docker.io
#		docker push mentice/jobber-runner

FROM ubuntu as base

# Install docker
RUN apt-get update -qqy
RUN apt-get install -qqy apt-transport-https ca-certificates curl software-properties-common gcc

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository \
	"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
	$(lsb_release -cs) \
	stable"
RUN apt-get update
RUN apt-get install -y docker-ce

# Install nvidia-docker2
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
	 curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list > \
	 /etc/apt/sources.list.d/nvidia-docker.list
RUN apt-get update
RUN apt-get install -y nvidia-docker2

# Install python3.6 and pip3
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev python-pip python3-pip
# Note: can't run get-pip.py because its not compatible with 3.6 (it does work with 3.7)
#       Should be able to revert from pip3 to back to pip after next upgrade to 3.7
# RUN ln /usr/bin/python3.6 /usr/bin/python
# RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
# RUN python /tmp/get-pip.py

# Install gcloud (see https://hub.docker.com/r/google/cloud-sdk/builds/bssdz9zrkxfbcpz4wed4eej/)
ENV PATH /google-cloud-sdk/bin:$PATH
ARG CLOUD_SDK_VERSION=219.0.1
ENV CLOUD_SDK_VERSION=$CLOUD_SDK_VERSION
RUN pip install crcmod
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-${CLOUD_SDK_VERSION}-linux-x86_64.tar.gz && \
	 tar xzf google-cloud-sdk-${CLOUD_SDK_VERSION}-linux-x86_64.tar.gz && \
	 rm google-cloud-sdk-${CLOUD_SDK_VERSION}-linux-x86_64.tar.gz && \
	 ln -s /lib /lib64 && \
	 gcloud config set core/disable_usage_reporting true && \
	 gcloud config set component_manager/disable_update_check true && \
	 gcloud config set metrics/environment github_docker_image && \
	 gcloud --version

WORKDIR /tmp
COPY runner_image/requirements.txt ./

# RUN pip install -r requirements.txt
RUN pip3 install -r requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /jobber
COPY setup.py DESCRIPTION.md ./
COPY src /jobber/src/
# RUN pip install .
RUN pip3 install .

WORKDIR /runner
COPY runner_image/runner.py ./
ENTRYPOINT ["python3", "runner.py"]