Dosigate examples
=================

Examples of Gate simulations in radiation therapy. 

Data are not in the git repository but (kindly) hosted on [MIDAS](http://midas3.kitware.com/midas/community/28). To download the data use as follow.

 * First, clone the repository : ```git clone https://github.com/dsarrut/dosigate-examples.git``` 
 * In the same folder: ```ccmake .```, then hit `c` and `g` to configure and generate
 * Just type ```make``` to download the data


Please edit the number of particles in all main.mac files, to obtain a good statistical uncertainty. 

To run example, go into one folder: 

 * Brachytherapy
 * External-beam-therapy-photon
 * Molecular-therapy-I131
 * Protontherapy

And type : ```Gate mac/main.mac```. The Gate executable must be in your path. 
