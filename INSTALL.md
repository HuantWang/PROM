# Installation

Install the latest Prom release using:

```
pip install -U prom
```

PROM was tested with Python 3.8 and Ubuntu 18.04.

## DOCKER

Install docker engine by following the instructions [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

1. Fetch the docker image from docker hub.

```
$ sudo docker pull ssimage/prom_0.1
```

To check the list of images, run:

```
$ sudo docker images
REPOSITORY                                   TAG                 IMAGE ID            CREATED             SIZE
ssimage/prom_0.1		     				latest              ac6b624d06de        2 hours ago         41.8GB
```

1. Run the docker image.

```
$ docker run -dit -P --name=prom ssimage/prom_0.1 /bin/bash
$ docker start prom 
$ docker exec -it prom /bin/bash
```

## Building from source

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to manage the remaining build dependencies. First create a conda environment with the required dependencies:

```
conda create -y -n prom python=3.8
conda activate prom
```

If you plan to contribute to Prom, install the development environment requirements using:

```
make dev-init
```

## Have a Test

```
# A demo for testing Prom 
$ conda activate prom
$ cd ./prom/
$ python test_cases.py --path_to_data ../data/test --mode test 
```
