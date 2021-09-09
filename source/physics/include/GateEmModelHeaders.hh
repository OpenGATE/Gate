/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"

// Compton
#include "G4LivermoreComptonModel.hh"
#include "G4LivermorePolarizedComptonModel.hh"
#include "G4PenelopeComptonModel.hh"

// eIonisation
#include "G4LivermoreIonisationModel.hh"
#include "G4PenelopeIonisationModel.hh"

//Bremsstrahlung
#include "G4LivermoreBremsstrahlungModel.hh"
#include "G4PenelopeBremsstrahlungModel.hh"

//Annihilation
#include "G4PenelopeAnnihilationModel.hh"

//PhotoElectric
//#include "G4LivermorePolarizedPhotoElectricModel.hh"
// The following three lines are necessary to work around a bug in the
// Geant4 10.02 release (December 2015): the "polarized" Livermore
// photo-electric model include file has the exact same #include guard
// as the (not polarized) Livermore photo-electric model.
// FIXME: once Geant4 releases a patched version in which this issue
// is fixed, this ugly work-around can be removed. Maybe we should then
// add a Geant4 version check in CMakeLists.txt to warn in case the
// user still uses the unpatched 4.10.02 version.
#ifdef G4LivermorePhotoElectricModel_h
#undef G4LivermorePhotoElectricModel_h
#endif
#include "G4LivermorePhotoElectricModel.hh"
#include "G4PenelopePhotoElectricModel.hh"

//Rayleigh
#include "G4LivermoreRayleighModel.hh"
#include "G4LivermorePolarizedRayleighModel.hh"
#include "G4PenelopeRayleighModel.hh"

//GammaConversion
#include "G4LivermoreGammaConversionModel.hh"
#include "G4LivermorePolarizedGammaConversionModel.hh"
#include "G4PenelopeGammaConversionModel.hh"



