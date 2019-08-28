FROM tensorflow/tensorflow:1.7.1-devel-gpu
COPY . /root/go9x9
RUN apt-get update -q && apt-get install cython -yqq && sh build.sh
