#!/bin/sh

#Variables to modify
Repository=opengatecollaboration
Centos_Version=centos:7
ROOT_Version=v6-14-02
CLHEP_Version=2.3.3.2
Geant4_Version=10.4.2
Gate_Version=8.1

#Variables to preserve
Geant4_Tag=$Repository/geant4:$Geant4_Version
Gate_Tag=$Repository/gate:$Gate_Version-docker

docker build -t $Geant4_Tag -f DockerFileGeant \
    --build-arg Centos_Version=$Centos_Version \
    --build-arg ROOT_Version=$ROOT_Version \
    --build-arg Geant4_Version=v$Geant4_Version .
docker push $Geant4_Tag

docker build -t $Gate_Tag -f DockerFileGate --build-arg Geant4_Version=$Geant4_Tag --build-arg Gate_Version=v$Gate_Version .
docker push $Gate_Tag