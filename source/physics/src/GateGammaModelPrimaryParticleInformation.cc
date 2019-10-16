/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateGammaModelPrimaryParticleInformation.hh"

GateGammaModelPrimaryParticleInformation::GateGammaModelPrimaryParticleInformation() { }

GateGammaModelPrimaryParticleInformation::~GateGammaModelPrimaryParticleInformation() { }

void GateGammaModelPrimaryParticleInformation::Print() const { }

void GateGammaModelPrimaryParticleInformation::setGammaSourceModel( GateGammaModelPrimaryParticleInformation::GammaSourceModel model ) { fGammaSourceModel = model; }

GateGammaModelPrimaryParticleInformation::GammaSourceModel GateGammaModelPrimaryParticleInformation::getGammaSourceModel() const { return fGammaSourceModel; }

void GateGammaModelPrimaryParticleInformation::setGammaKind( GateGammaModelPrimaryParticleInformation::GammaKind kind ) { fGammaKind = kind; }

GateGammaModelPrimaryParticleInformation::GammaKind GateGammaModelPrimaryParticleInformation::getGammaKind() const { return fGammaKind; }

void GateGammaModelPrimaryParticleInformation::setInitialPolarization( const G4ThreeVector& polarization ) { fInitialPolarization = polarization; }

G4ThreeVector GateGammaModelPrimaryParticleInformation::getInitialPolarization() const { return fInitialPolarization; }
