/**
 *  @copyright Copyright 2019 The J-PET Gate Authors. All rights reserved.
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  @file GateGammaModelPrimaryParticleInformation.cc
 */
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
