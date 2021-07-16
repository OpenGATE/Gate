.. _installation_guide-label:

Installation Guide V9.1
=======================

.. contents:: Table of Contents
   :depth: 15
   :local:

General Information about GATE
------------------------------

The GATE mailing list
~~~~~~~~~~~~~~~~~~~~~

You are encouraged to participate in the dialog and post your suggestion or even implementation on the
Gate-users mailing list, the GATE mailing list for users.
You can subscribe to the Gate-users mailing list, by `signing up to the gate-users mailing list <http://lists.opengatecollaboration.org/mailman/listinfo/gate-users>`_.

If you have a question, it is possible that it has been asked and answered before, and stored in the `archives <http://lists.opengatecollaboration.org/pipermail/gate-users/>`_.
These archives are public and are indexed by the usual search engines. By starting your Google search string with *site:lists.opengatecollaboration.org* you'll get list of all matches of your search on the gate-users mailing list, e.g. `site:lists.opengatecollaboration.org pencilbeam <https://www.google.com/search?q=site%3Alists.opengatecollaboration.org+pencilbeam>`_.

The GATE project on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~

GATE project is now publicly available on `GitHub <https://github.com/OpenGATE/Gate>`_. You can use this to:

* Check out the bleeding edge development version
* Report bugs by creating a new `issue <https://github.com/OpenGATE/Gate/issues>`_. (If you are not entirely sure that what you are reporting is indeed a bug in Gate, then please first check the `gate-users mailing list <http://lists.opengatecollaboration.org/mailman/listinfo/gate-users>`_.)
* Contribute to Gate by changing the source code to fix bugs or implement new features:

  * Get a (free) account on GitHub, if you do not have one already.
  * `Install Git <https://git-scm.com/download/linux>`_ on the computer where you do your development, if it has not yet been installed already. And make sure to configure git `with your name <https://help.github.com/articles/setting-your-username-in-git/>`_ and `with your email address <https://help.github.com/articles/setting-your-commit-email-address-in-git/>`_.
  * Start by `making a fork <https://help.github.com/articles/fork-a-repo/>`_ of the GATE public repository (click the "Fork" button in the upper right corner on the `Gate main page <https://github.com/OpenGATE/Gate/>`_ on GitHub.
  * Note that we use the *develop* branch to collect all the bleeding edge developments and the *master* to track the releases. In the future we may merge these two, and use only *master*, like it's done in most other projects on GitHub. Releases are defined using "tags".
  * Then clone your own fork: *git clone https://github.com/YOUR_USERNAME/Gate.git* to get the code on the computer that you will use to develop and compile Gate.
  * Make a new branch, dedicated to the bugfix or new feature that want to implement in Gate. You can either create the branch first on GitHub and then *git pull* it to your clone, or create it directly in your clone and *git push* it later. Make sure that your branch is based on the *develop* branch. Note that after creating your branch you also need to check it out.
  * With *git branch -l* you can check which branches are available in your clone and which one is currently checked out. With *git checkout <branchname>* you can change between branches. Be careful not to do this when you still have uncommitted changes (unless you deliberately want to undo those changes).
  * Now: implement your bugfix or your new feature and *commit* your changes to your new branch. It's usually better to make many small commits than a single big one (though it is of course also desirable that every commit leaves the code in a compilable state). Please provide `concise but informative commit messages <https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53>`_ ! Use *git push* to upload your commits to (your fork on) GitHub. This facilitates developing on multiple machines and also avoids loss of time and effort in the unfortunate event of a hardware failure.
  * If you are working for a longer time on your fix or new feature, like a few days, weeks or even months, then it is important to make sure to `keep your fork in sync with the upstream repository <https://help.github.com/articles/syncing-a-fork/>`_.
  * Once you are convinced that your code is OK, make sure it's all pushed to your fork on GitHub. Then:

    1) Create a `pull-request <https://help.github.com/articles/using-pull-requests/>`_ from the branch on your Gate repository to the official Gate repository
    2) Provide an example that tests your new feature
    3) If you implemented a new feature, have the associated documentation ready
    4) Inform these three people from the collaboration (Sebastien Jan, David Sarrut and David Boersma) who will then get in touch with you to integrate your changes in the official repository.

  * For your next bugfix or new feature you do not need to make a new fork, you can use the existing one. But before doing any new work you should make sure to `synchronize <https://help.github.com/articles/syncing-a-fork/>`_ the *develop* branch in your fork with the "upstream" (main) *develop* branch:

    1) Check your "remote repositories" with *git remote -v*
    2) The "origin" repository should be your own fork on GitHub, *https://github.com/YOUR_USERNAME/Gate*.
    3) The "upstream" repository should be the main Gate one, that is *https://github.com/OpenGATE/Gate*.
    4) If your clone does not yet have an "upstream", then add it with *git remote add upstream https://github.com/OpenGATE/Gate*.
    5) Run *git status* to make sure that you checked out the *develop* branch, and *git pull* to make sure that it is in sync with your fork on GitHub and that there no uncommitted edits.
    6) Then run *git fetch upstream*, followed by *git merge upstream/develop*.
    7) Now you are ready to create new branches for new bugfixes and features.

* For more detailed references, recipes, and tutorials on git: please check the web. When copypasting commands, remember that in Gate the "develop" branch currently plays the role of the "master" branch. Our "master" branch is used to track the releases. You will not find the latest bleeding edge code on it. We may change this policy in the near future, to be more conforming to the predominant conventions.

Installing GATE on Linux
~~~~~~~~~~~~~~~~~~~~~~~~

This section describes the installation procedure of GATE. This includes three steps:

* Install Geant4
* Install ROOT
* Install GATE

This section starts with a brief overview of the recommended configurations, followed by a short introduction to the installation of Geant4, and then explains the installation of GATE itself on Linux.

It should be highlighted that features depending on external components (libraries or packages) may only be activated if the corresponding component is installed. It is the user's responsibility to check that these components are installed before activating a feature. Except for Geant4, which is closely related to GATE, the user should refer to the Installation Guide of the external components.

In addition, you should also install any Geant4 helper you wish to use, especially *OpenGL* if required, before installing Geant4 itself. You can either download the source codes and compile the libraries or download precompiled packages which are available for a number of platform-compiler. If you choose to or have to compile the packages, you will need:

* a C++ compiler (new enough to compile code with the C++11 standard)
* the GNU version of *make*
* `CMAKE <https://www.cmake.org>`_ tool (3.3 or newer)

The ROOT data analysis package may also be needed for post-processing or for using the GATE online plotter (enabling the visualization of various simulation parameters and results in real time). ROOT is available for many platforms and a variety of precompiled packages can be found on the `ROOT homepage <http://root.cern.ch/>`_. If your gcc compiler is version 6 or newer, then you should use a recent ROOT 6 release.

The `LMF <http://opengatecollaboration.org/sites/default/files/lmf_v3_0.tar.gz>`_ and `ecat7 <http://www.opengatecollaboration.org/ECAT>`_ packages are also provided on the `GATE <http://www.opengatecollaboration.org>`_ website. They offer the possibility to have different output formats for your simulations. Note that this code is very old and not supported by the Gate collaboration, only provided "as is". With newer compilers you may have to do some minor hacking (for ECAT you may need to add compiler flags to select the C90 standard, for instance).

Package Requirements
~~~~~~~~~~~~~~~~~~~~

Compiling software usually requires certain system libraries and compilation tools.
Furthermore, GATE and Geant4 have various package requirements which have to be met BEFORE installing or compiling.
Currently lists have been created for Ubuntu 14.04 (and newer) and SuSE Leap 42.3. Visit the :ref:`package_requirements-label` page for detailed package lists.

Installing with MacPorts on OS X
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GATE can be installed on Mac OS X by following the previous installation instruction on Linux. An alternative way is to install Gate via MacPorts (http://www.macports.org/) with::

    sudo port install gate

Apart from the `Gate` command this also installs a standalone app::

    /Applications/MacPorts/Gate.app

(Thanks Mojca Miklavec for this contribution).

GATE compilation and installation
---------------------------------

Recommended configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

For the 9.1 release, the recommended configuration is the following:

* Geant4 10.7 (available in http://geant4.web.cern.ch/geant4/support/download.shtml), but remains backward compatible with 10.6 also. 
* The `GateRTion 1.0 <http://opengatecollaboration.org/GateRTion>`_ release, which is very similar to Gate 8.1, can *only* be built with Geant4 10.03.p03.
* CMake minimal version: 3.3 (with SSL support)

Compilation instructions
~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`compilation_instructions-label`

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
