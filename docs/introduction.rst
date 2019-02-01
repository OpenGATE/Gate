Introduction
============

.. figure:: GoldGate.jpg

Geant4 Application for Emission Tomography: a simulation toolkit for PET
and SPECT

`OpenGATE Collaboration`_

.. _`OpenGATE Collaboration`: http://www.opengatecollaboration.org

Authors
~~~~~~~

OpenGATE spokesperson: I. Buvat (IMIV UMR1023 Inserm-CEA-Université
Paris Sud, ERL 9218 CNRS Orsay)

OpenGATE technical coordinator: S. Jan (IMIV UMR1023
Inserm-CEA-Université Paris Sud, ERL 9218 CNRS Orsay)

Editorial board at the time of the first Users Guide version (2004) S.
Glick (UMASS), S. Kerhoas (CEA Saclay), F. Mayet (CNRS-LPSC Grenoble)

Authors of the first edition (2004) : S. Jan, G. Santin, D. Strul, S.
Staelens, K. Assié, D. Autret, S. Avner, R. Barbier, M. Bardiès, P. M.
Bloomfield, D. Brasse, V. Breton, P. Bruyndonckx, I. Buvat, A. F.
Chatziioannou, Y. Choi, Y. H. Chung, C. Comtat, D. Donnarieix, L.
Ferrer, S. J. Glick, C. J. Groiselle, D. Guez, P.-F. Honore, S.
Kerhoas-Cavata, A. S. Kirov, V. Kohli, M. Koole, M. Krieguer, D. J. van
der Laan, F. Lamare, G. Largeron, C. Lartizien, D. Lazaro, M. C. Maas,
L. Maigne, F. Mayet, F. Melot, C. Merheb, E. Pennacchio, J. Perez, U.
Pietrzyk, F. R. Rannou, M. Rey, D. R. Schaart, C. R. Schmidtlein, L.
Simon, T. Y. Song, J.-M. Vieira, D. Visvikis, R. Van de Walle, E.
Wieers, C. Morel

Special Thanks: Geant4 Collaboration and LOW energy WG

The GATE mailing list
~~~~~~~~~~~~~~~~~~~~~

You are encouraged to participate in the dialog and post your
suggestions, questions and answers to colleagues' questions on the
gate-users mailing list, the GATE mailing list for users. You can
subscribe to the gate-users mailing list, by registering at the GATE web
site: http://www.opengatecollaboration.org

If you have a question, it is very likely that it has already been asked
and answered, and is now stored in the archives. Please use the search
engine to see if your question has already been answered before sending
a mail to the GATE-users .

Forewords
~~~~~~~~~

Monte Carlo simulation is an essential tool in emission tomography to
assist in the design of new medical imaging devices, assess new
implementations of image reconstruction algorithms and/or scatter
correction techniques, and optimise scan protocols. Although dedicated
Monte Carlo codes have been developed for Positron Emission Tomography
(PET) and for Single Photon Emission Computerized Tomography (SPECT),
these tools suffer from a variety of drawbacks and limitations in terms
of validation, accuracy, and/or support (Buvat). On the other hand,
accurate and versatile simulation codes such as GEANT3 (G3), EGS4, MCNP,
and GEANT4 have been written for high energy physics. They all include
well-validated physics models, geometry modeling tools, and efficient
visualization utilities. However these packages are quite complex and
necessitate a steep learning curve.

GATE, the *GEANT4 Application for Emission Tomography* (MIC02, Siena02,
ITBS02, GATE, encapsulates the GEANT4 libraries in order to achieve a
modular, versatile, scripted simulation toolkit adapted to the field of
nuclear medicine. In particular, GATE provides the capability for
modeling time-dependent phenomena such as detector movements or source
decay kinetics, thus allowing the simulation of time curves under
realistic acquisition conditions.

GATE was developed within the OpenGATE Collaboration with the objective
to provide the academic community with a free software, general-purpose,
GEANT4-based simulation platform for emission tomography. The
collaboration currently includes 21 laboratories fully dedicated to the
task of improving, documenting, and testing GATE thoroughly against most
of the imaging systems commercially available in PET and SPECT
(Staelens, Lazaro).

Particular attention was paid to provide meaningful documentation with
the simulation software package, including installation and user's
guides, and a list of FAQs. This will hopefully make possible the long
term support and continuity of GATE, which we intend to propose as a new
standard for Monte Carlo simulation in nuclear medicine.

In name of the OpenGATE Collaboration

Christian MOREL CPPM CNRS/IN2P3, Marseille

Member institutes of the OpenGATE Collaboration (May 2015):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  CNRS Laboratory of Corpuscular Physics @ Clermont-Ferrand (LPC)
-  CNRS Centre de Physique des Particules de Marseille (CPPM), Marseille
-  CNRS Imaging and Modelling in Neurobiology and Cancerology lab @
   Orsay (IMNC), Orsay
-  CNRS IRES, Centre Pluridisciplinaire Hubert Curien (CPHC) @
   Strasbourg
-  CNRS Laboratoire de Physique Subatomique et des technologies
   associées (SUBATECH), Nantes
-  INSERM - CNRS CREATIS lab @ Lyon
-  U892 INSERM @ Nantes
-  LATIM, U1101 INSERM, Brest
-  IMIV, UMR1023 Inserm-CEA-Université Paris Sud, ERL 9218 CNRS, Service
   Hospitalier Frédéric Joliot (SHFJ), Orsay
-  Sungkyunkwan University School of Medicine (DMN), Seoul
-  Forschungszentrum-Juelich (IME)
-  Memorial Sloan-Kettering Cancer Center (Department of Medical
   Physics), New York
-  University of Athens (IASA)
-  Delft University of Technology (IRI)
-  UC Davis, California
-  MedAustron, Wiener Neustadt
-  Medical University Vienna, Wien
-  UMR 1037 INSERM/UPS, Toulouse

Overview
~~~~~~~~

GATE combines the advantages of the GEANT4 simulation toolkit well-validated
physics models, sophisticated geometry description, and powerful visualization
and 3D rendering tools with original features specific to emission tomography.
It consists of several hundred C++ classes. Mechanisms used to manage time,
geometry, and radioactive sources form a core layer of C++ classes close to the
GEANT4 kernel :nameref:`gate_layers`. An application layer allows for the
implementation of user classes derived from the core layer classes, e.g.
building specific geometrical volume shapes and/or specifying operations on
these volumes like rotations or translations. Since the application layer
implements all appropriate features, the use of GATE does not require C++
programming: a dedicated scripting mechanism - hereafter referred to as the
macro language - that extends the native command interpreter of GEANT4 makes it
possible to perform and to control Monte Carlo simulations of realistic setups.

.. figure:: GATE_layers.jpg
   :alt: Figure 1: Structure of GATE
   :name: gate_layers

   Structure of GATE

One of the most innovative features of GATE is its capability to synchronize all
time-dependent components in order to allow a coherent description of the
acquisition process. As for the geometry definition, the elements of the
geometry can be set into movement via scripting. All movements of the
geometrical elements are kept synchronized with the evolution of the source
activities. For this purpose, the acquisition is subdivided into a number of
time-steps during which the elements of the geometry are considered to be at
rest. Decay times are generated within these time-steps so that the number of
events decreases exponentially from time-step to time-step, and decreases also
inside each time-step according to the decay kinetics of each radioisotope. This
allows for the modeling of time-dependent processes such as count rates, random
coincidences, or detector dead-time on an event-by-event basis. Moreover, the
GEANT4 interaction histories can be used to mimic realistic detector output. In
GATE, detector electronic response is modeled as a linear processing chain
designed by the user to reproduce e.g. the detector cross-talk, its energy
resolution, or its trigger efficiency.

The first users guide was organized as follow: chapter 1 of this document guides
you to get started with GATE. The macro language is detailed in Chapter 2.
Visualisation tools are described in Chapter 3. Then, Chapter 4 illustrates how
to define a geometry by using the macro language, Chapter 5 how to define a
system, Chapter 6 how to attach sensitive detectors, and Chapter 7 how to set up
the physics used for the simulation. Chapter 8 discusses the different
radioactive source definitions. Chapter 9 introduces the digitizer which allows
you to tune your simulation to the very experimental parameters of your setup.
Chapter 10 draws the architecture of a simulation. Data output are described in
Chapter 11. Finally, Chapter 12 gives the principal material definitions
available in GATE. Chapter 13 illustrates the interactive, bathc, or cluster
modes of running GATE.

