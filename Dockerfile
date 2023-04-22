#
# RoboCIn Soccer Simulator 2d
# Author: Walber de Macedo Rodrigues
# Build command: sudo docker build . -t simulatoragent:build
# Run command: sudo docker run -it --rm simulatoragent:build

# Pull base image.
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND noninteractive

RUN \
    apt-get update && \ 
    apt-get -y upgrade && \
    apt-get install -y python3 wget git g++ gcc ca-certificates gnupg software-properties-common

RUN \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && \
    add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/' && \
    apt update && apt install -y r-base && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]

# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

COPY ./env.yml ./
RUN mkdir -p ./preprocessing/deepembeddings
ADD ./preprocessing/deepembeddings/ ./preprocessing/deepembeddings

RUN conda env create -f env.yml

# Define default command.
CMD ["bash"]