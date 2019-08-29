# go9x9

## Installation (Docker & Nvidia-Docker)

Set up the repository and install docker

```sh
apt-get update

apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

sudo apt-key fingerprint 0EBFCD88

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

apt-get update

apt-get install docker-ce
```

Install Nvidia-Docker

```sh
# Add the package repositories
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update

# Install nvidia-docker2 and reload the Docker daemon configuration
apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd
```

## Build from source

### Preparation

Pull Docker image `tensorflow/tensorflow:1.7.1-devel-gpu`

```sh
docker pull tensorflow/tensorflow:1.7.1-devel-gpu
```

Start a Docker container and attach

```sh
nvidia-docker run -it -d --name go-play -v -p 0.0.0.0:6006:6006 tensorflow/tensorflow:1.7.1-devel-gpu bash
docker exec -it go-play bash
```

### Build

Clone this repo

```sh
git clone https://github.com/patwu/go9x9.git
cd go9x9
```

Install dependency `cython` via `apt`

```sh
apt update
apt install cython
```

Build libs

```sh
sh build.sh
```

## Build the Docker image

```sh
docker build -t go9x9 .
```
