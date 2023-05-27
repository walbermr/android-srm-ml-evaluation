# Pull base image.
FROM continuumio/miniconda3

RUN \
    apt-get update && \ 
    apt-get -y upgrade && \
    apt-get install -y python3 wget git g++ gcc ca-certificates gnupg software-properties-common r-base

# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

COPY ./env.yml ./
RUN mkdir -p ./preprocessing/deepembeddings
ADD ./preprocessing/deepembeddings/ ./preprocessing/deepembeddings

RUN conda env create -f env.yml

ADD ./preprocessing/r_packages.py ./r_packages.py
RUN conda run -n srm python ./r_packages.py

CMD [ "bash" ]