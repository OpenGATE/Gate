.. _installation_guide-label:

Installation Guide V9.2
=======================

.. contents:: Table of Contents
   :depth: 15
   :local:

How to retrieve GATE
--------------------

There are several ways to get a working GATE installation on your computer depending on your system and your needs. 

For macOS users, you can install pre-compiled version of GATE (via MacPorts) or you can compile it (:ref:`compilation_instructions-label`). 

For linux users, you can use docker version (no compilation), singularity (from docker, no compilation) or you can compile it. 

For windows, we provide vGATE, an complete virtual machine which runs ubuntu with GATE already installed. 


Without compilation
-------------------

Installing with MacPorts (macOS users only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GATE can be installed on Mac OS X by following the previous installation instruction on Linux. An alternative way is to install Gate via MacPorts (http://www.macports.org/) with::

    sudo port install gate

Apart from the `Gate` command this also installs a standalone app::

    /Applications/MacPorts/Gate.app

(Thanks Mojca Miklavec for this contribution).


With Docker
~~~~~~~~~~~

See :ref:`docker_gate-label`


With Singularity
~~~~~~~~~~~~~~~~

To be written

Via virtual machine
~~~~~~~~~~~~~~~~~~~

See :ref:`vgate-label`

Compiling GATE (macOS, unix and linux users)
----------------------------------------------


See dedicated page : :ref:`compilation_instructions-label`

Validating Installation
-----------------------

If you are able to run Gate after installation by typing::

   Gate

it is an indication that your installation was successful.

**However, before you do any research, it is highly recommended that you validate your installation.**

See :ref:`validating_installation-label` for benchmarks and further information.

Other Web Sites
---------------
 
* G4 Agostinelli S et al 2003 GEANT4 - a simulation toolkit Institute Nucl. Instr. Meth.  A506  250-303 GEANT4 website: http://geant4.web.cern.ch/geant4/
* CLHEP - A Class Library for High Energy Physics: http://proj-clhep.web.cern.ch
* OGL OpenGL Homepage: http://www.opengl.org
* DAWN release:  http://geant4.kek.jp/
* ROOT  Brun R, Rademakers F 1997 ROOT - An object oriented data analysis framework Institute Nucl. Instr. Meth.  A389  81-86 ROOT website: http://root.cern.ch
* libxml website: http://www.libxml.org
