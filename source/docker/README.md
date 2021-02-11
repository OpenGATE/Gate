# GateDocker
Gate Docker scripts, Generate the image by running the `Generate.sh` script

# First Image Geant4
## Docker for gate prerequisites

[optional if not done before]
`systemctl start docker`
login: `docker login`

[build and use]
* build: 
    * `docker build -t bishopwolf/geant4:10.6.1 -f DockerFileGeant --build-arg ROOT_Version=v6-19-02 --build-arg Geant4_Version=v10.6.1 .`
* push: 
    * `docker push bishopwolf/geant4:$version`
* interactive: 
    * `docker run -it --rm -v $PWD:/APP bishopwolf/geant4:$version /bin/bash`

Where: 

* `$version` is `10.6.1` for gate `9.0`

# Second image Gate
## Docker for gate

[optional if not done before]
`systemctl start docker`
login: `docker login`

[build and use]
* build: 
    * `docker build -t bishopwolf/gate:9.0 -f DockerFileGate --build-arg Geant4_Version=bishopwolf/geant4:10.6.1 --build-arg Gate_Version=v9.0 .`
* push:  
    * `docker push bishopwolf/gate:$version`
* run command:  
    * `docker run -i --rm -v $PWD:/APP bishopwolf/gate:$version mac/main.mac`
* interactive:  
    * `docker run -it --rm -v $PWD:/APP --entrypoint /bin/bash bishopwolf/gate:$version`

You can just install docker and then create an alias in your configuration
* ```echo "alias Gate='docker run -i --rm -v $PWD:/APP bishopwolf/gate:$version'" >> ~/.bashrc```

Where: 
* `$version=9.0`
