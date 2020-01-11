.. _materials-label:

Materials
=========

.. contents:: Table of Contents
   :depth: 15
   :local:

The Gate material database
--------------------------

The primary method for defining the properties of the materials used in Gate is by a materials database. This file holds all the information required for Gate to assign the nuclear properties from the Geant4 data sets, and is easily modified by the user. The OpenGate collaboration supplies a fairly extensive listing of materials in this file as part of Gate. This chapter describes the details of how to modify this database.

As alluded to in the previous paragraph, there exists an alternate method for materials definitions. As discussed in previous chapters, Gate scripts are developed from Geant4 C++ data classes in order to simplify and standardize input for Geant4. As a result, materials definitions can be written and compiled in C++ directly using the Geant4 tools. Specifying materials in this manner is beyond the scope of this document. For those interested in direct access to Geant4's materials should refer to the *Geant4 User's Guide: For Application Developers* and the *Geant4 User's Guide: For Toolkit Developers* for more detailed information.

The material database contains two Geant4 structures called elements and materials that are used to define the physical properties of the atoms, molecules, and compounds. In contrast with Geant4, Gate does not use isotopic abundances. This omission has little bearing on Gate applications because isotopic abundances are unimportant in low to mid energy photon and charged particle interactions. In fact, this distinction is only important for enriched or depleted materials interacting with neutrons or, high energy (>5 MeV) photons or charged particles.

It is possible to use several material database. If a material is defined in several database, Gate keeps the material in last database called (with a warning message). To call a database::

  /gate/geometry/setMaterialDatabase MyMaterialDatabase.db

Elements
~~~~~~~~

Elements are the building blocks of all the materials used in Gate simulations. Elements in Gate are defined as in a periodic table. Gate stores the elements name, symbol, atomic number, and molar mass. As stated above, isotopic abundances are not referenced or used. The supplied file *GateMaterials.db* contains the most commonly used elements and their molar masses as they are found in nature.

Some elements, particularly those that have an isotope with a large cross section for neutron absorption, have isotopic abundances and thus molar masses that vary depending upon their source. One element that exhibits this behavior is boron. In practice this behavior is not important for Gate applications.

Materials
~~~~~~~~~

In Gate, materials are defined as combinations of elements, and are an important parameter that Gate uses for all of the particle interactions that take place during a simulation. These combinations of elements require defining four additional parameters. These are the material's name, density, constituent element(s), and their individual abundances.

The composition of elements within a material can be defined in two different ways. If the material is a chemical compound then its relative amounts of elements are specified by the number of atoms in the chemical formula of the compound. For example, methane CH_{4} would be defined as having one carbon atom and four hydrogen atoms. If the material is better described as a mixture, such as 304-stainless steel, then the relative combinations of the elements are given by mass fraction. In the case of 304-stainless steel, the various mass fractions are given as 0.695 Iron, 0.190 Chromium, 0.095 Nickel, and 0.020 Manganese. Note that the mass fractions from the elements must all sum to one.

Densities of materials often vary greatly between different sources and must be carefully selected for the specific application in mind. Units of density must also be defined. These are typically given in g/cm3 but can be given in more convenient units for extreme cases. For example, a vacuum's density may be expressed in units of mg/cm3.

Modifying the Gate material database
------------------------------------

New element
~~~~~~~~~~~

Defining a new element is a simple and straightforward process. Simply open the *GateMaterials.db* file with the text editor of your choice. At the top of the file is the header named **[Elements]** and somewhere in the middle of the file is another header named **[Materials]**. All element definitions required by the application must be included between these two headers. The format for entering an element is given by the elements name, symbol, atomic number, and molar mass. Below is an example::

  Element Example GateMaterials.db:

  [Elements]
  Hydrogen:   S= H   ; Z=  1. ; A=   1.01  g/mole  
  Helium:     S= He  ; Z=  2. ; A=   4.003 g/mole
  Lithium:    S= Li  ; Z=  3. ; A=   6.941 g/mole
  Beryllium:  S= Be  ; Z=  4. ; A=   9.012 g/mole
  Boron:      S= B   ; Z=  5. ; A=  10.811 g/mole
  Carbon:     S= C   ; Z=  6. ; A=  12.01  g/mole

In this example the name of the element is given first and is followed by a colon. Next, the standard symbol for the element is given by S=symbolic name followed by a semi-colon. The atomic number and molar mass follow the symbolic name given by *Z=atomic number* with a semi-colon and by *A=molar mass units* for the molar mass and its units.

New material
~~~~~~~~~~~~

Materials are defined in a similar manner to elements but contain some additional parameters to account for their density and composition. Defining density is straightforward and performed the same way for all materials. However, material compositions require different definitions depending upon their form. These compositional forms are pure substances, chemical compounds, and mixtures of elements.

To add or modify a material in the material database, the *GateMaterials.db* file should be open using a text editor. The new entry should be inserted below the header named *Materials*. All material definitions required by the application must be included below this second header. Material definitions require several lines. The first line specifies their name, density, number of components, and an optional parameter describing the materials state (solid, liquid, or gas). The second and following lines specify the individual components and their relative abundances that make up this material.

The compositional forms of materials that Gate uses are pure substances, chemical compounds, mixtures of elements, and mixtures of materials. Gate defines each of these cases slightly differently and each will be dealt with separately below. In every case, the elements being used in a material definition must be previously defined as elements.

Elements as materials
^^^^^^^^^^^^^^^^^^^^^

Substances made of a pure element are the easiest materials to define. On the first line, enter the name of the material (the name of the material can be the same as that of the element), its density, its number of constituents (which is one in this case), and optionally its state (solid, liquid, or gas). The default state is gaseous. On the second line enter the element that it is composed of and the number of atoms of that element (in the case of an element as a material this number is 1). For example::

  Elements as materials example GateMaterials.db:

  [Materials]
  Vacuum: d=0.000001 mg/cm3 ; n=1 
         +el: name=Hydrogen ; n=1
  Aluminium: d=1.350 g/cm3 ; n=1 ; state=solid
         +el: name=auto ; n=1
  Uranium: d=18.90 g/cm3 ; n=1 ; state=solid
         +el: name=auto ; n=1

On the first line the density (with units) is defined by *d=material density units* and is separated by a semi-colon from the number of constituents in the material defined by *n=number of elements*. If the optional material form parameter is used it is also separated by a semi-colon. The available forms are gas, liquid, and solid. On the second line the individual elements and their abundances are defined by *+el: name=name of the element* ; *n=number of atoms*. If the name of the element and the material are the same, the element name can be defined by *+el: name=auto* command.

Compounds as materials
^^^^^^^^^^^^^^^^^^^^^^

Chemical compounds are defined based upon the elements they are made of and their chemical formula. The first line is identical to the first line of a pure substance except that the number of constituent elements is now greater than one. On the second and subsequent lines, the individual elements and their abundances are defined by *+el: name=name of the element*;*n=number of atoms*.

For example::

  Compounds as materials example GateMaterials.db:

  [Materials]
  NaI: d=3.67 g/cm3; n=2;  state=solid
         +el: name=Sodium ; n=1
         +el: name=Iodine ; n=1

  PWO: d=8.28 g/cm3; n=3 ; state=Solid
         +el: name=Lead; n=1
         +el: name=Tungsten; n=1
         +el: name=Oxygen; n=4

Mixtures as materials
^^^^^^^^^^^^^^^^^^^^^

Mixture of elements are defined by indicating the mass fraction of the elements that make up the mixture. The first line of this definition is identical to the first line of the definition of a chemical compound. On the second and subsequent lines, the individual elements and their mass fractions are defined by *+el: name=name of element*;*f=mass fraction*.

In the case of material mixtures, the sum of the mass fractions should be one. For example::

  Mixtures as materials example GateMaterials.db:

  [Materials]
  Lung:  d=0.26 g/cm3 ; n=9
         +el: name=Hydrogen  ; f=0.103
         +el: name=Carbon    ; f=0.105
         +el: name=Nitrogen  ; f=0.031
         +el: name=Oxygen    ; f=0.749
         +el: name=Sodium    ; f=0.002
         +el: name=Phosphor  ; f=0.002
         +el: name=Sulfur    ; f=0.003
         +el: name=Chlorine  ; f=0.003
         +el: name=Potassium ; f=0.002

  SS304:  d=7.92 g/cm3 ; n=4 ; state=solid
         +el: name=Iron      ; f=0.695
         +el: name=Chromium  ; f=0.190
         +el: name=Nickel    ; f=0.095
         +el: name=Manganese ; f=0.020

Mixtures of materials as materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another way material can be defined is as mixtures of other materials and elements. As an example::

  Mixtures of mixtures as materials example GateMaterials.db:

  [Materials]
  Aerogel:  d=0.200 g/cm3 ; n=3
          +mat: name=SiO2     ; f=0.625
          +mat: name=Water    ; f=0.374
          +el:  name=Carbon   ; f=0.001

In this example, the material, Aerogel, is defined to be made up of two materials, silicon dioxide and water, and one element, carbon. Mass fractions of the silicon dioxide, water, and carbon are given to specify the atom densities of the material when related to the density of the Aerogel. When specifying materials rather than elements the *+mat: name=identifier* must be used.

Ionization potential
--------------------

The ionization potential is the energy required to remove an electron to an atom or a molecule. By default, the ionization potential is calculated thanks to the Bragg’s additivity rule. It is possible to define the ionization potential of each material defined in gate. For example::

  /gate/geometry/setIonisationPotential Water 75 eV

