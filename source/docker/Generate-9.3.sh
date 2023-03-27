#!/bin/sh

#Variables to modify
Repository=opengatecollaboration
ROOT_Version=v6-24-06
CLHEP_Version=2.4.6.0
Geant4_Version=11.1.1
Gate_Version=9.3

#Variables to preserve
Geant4_Tag=$Repository/geant4:$Geant4_Version
Gate_Tag=$Repository/gate:$Gate_Version-docker

docker build -t $Geant4_Tag -f DockerFileGeant \
    --build-arg ROOT_Version=$ROOT_Version \
    --build-arg CLHEP_Version=$CLHEP_Version \
    --build-arg Geant4_Version=v$Geant4_Version .
docker push $Geant4_Tag

docker build -t $Gate_Tag -f DockerFileGate \
    --build-arg Geant4_Version=$Geant4_Tag \
    --build-arg Gate_Version=v$Gate_Version .
docker push $Gate_Tag
