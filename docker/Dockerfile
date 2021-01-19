FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV TZ=Europe/Moscow
ENV TERM xterm-256color

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing
RUN apt-get install -y \
	python3-pip \
	python3-tk \
	libboost-all-dev \
	unzip \
	wget

RUN apt-get -y install git
RUN python3 -m pip install --upgrade pip

RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh && \
bash Anaconda3-2019.03-Linux-x86_64.sh -b && \
rm Anaconda3-2019.03-Linux-x86_64.sh

ENV PATH=root/anaconda3/bin/:$PATH

COPY requirements.txt .
RUN conda create -y --name ns-vqa -c conda-forge pytorch --file requirements.txt 
ENV CONDA_ENV_NAME ns-vqa
