.. _docker_gate-label:

GATE using Docker
=================

.. contents:: Table of Contents
   :depth: 15
   :local:

GATE 9.2 on docker
------------------

A docker image for gate version 9.2 is available here: `Click here to download GATE 9.2 on docker <https://hub.docker.com/r/opengatecollaboration/gate>`_

Example to run GATE with Docker on your computer (unix-system):
--------------------------------------------------------------

We consider you already have the macro in your computer, you can do::

  cd folder/containing/macFolder
  docker run -i --rm -v $PWD:/APP opengatecollaboration/gate mac/main.mac


If you want to use alias in your macro, you can set the parameters like this::

  docker run -i --rm -v $PWD:/APP opengatecollaboration/gate "-a [nb,1000] mac/main.mac"

You can find more information about the docker image on `github <https://github.com/OpenGATE/Gate/tree/develop/source/docker>`_

Example to install GATE with Docker on Amazon Web Services (AWS) (Amazon Linux machine):
---------------------------------------------------------------------------------------

Example::

  # First: create and launch a Linux Virtual Machine on AWS
  # Second: register your local machine ssh public key to AWS
  # connect with ssh to your new Amazon Server VM on AWS (replace "IPv4" with the corresponding address)
  ssh ec2-user@ec2-"IPv4".eu-west-3.compute.amazonaws.com
  # install docker
  sudo yum update -y
  sudo yum install docker
  sudo service docker start
  sudo usermod -a -G docker ec2-user
  #logout
  exit
  # log back in
  ssh ec2-user@ec2-"IPv4".eu-west-3.compute.amazonaws.com
  docker info
  docker run -it opengatecollaboration/gate:9.2
  Gate

Example to install GATE with Docker on Amazon Web Services (AWS) (Ubuntu Linux machine):
---------------------------------------------------------------------------------------

Example::

  # First: create and launch a Linux Virtual Machine on AWS
  # Second: register your local machine ssh public key to AWS
  # connect with ssh to your new Ubuntu Server VM on AWS (replace "IPv4" with the corresponding address)
  ssh ubuntu@ec2-"IPv4".eu-west-3.compute.amazonaws.com
  # install docker
  sudo apt update
  sudo apt install -y docker.io
  # to run docker without sudo
  sudo usermod -a -G docker ubuntu # and then log out and back in
  # launch a docker container with GATE
  docker run -it opengatecollaboration/gate:9.2
  Gate

