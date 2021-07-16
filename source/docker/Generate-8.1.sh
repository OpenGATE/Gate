#!/bin/sh

#Variables to modify
Repository=opengatecollaboration
ROOT_Version=v6-19-02
# ROOT_Version=v6-14-02 -> This version doesn't compile
CLHEP_Version=2.3.3.2
Geant4_Version=10.4.2
Gate_Version=8.1

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
    --build-arg Gate_Version=v$Gate_Version \
    --build-arg USE_RTK=OFF .
docker push $Gate_Tag