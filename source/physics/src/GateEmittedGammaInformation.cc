/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#include "GateEmittedGammaInformation.hh"

GateEmittedGammaInformation::GateEmittedGammaInformation() {}

GateEmittedGammaInformation::~GateEmittedGammaInformation() {}

void GateEmittedGammaInformation::SetSourceKind( GateEmittedGammaInformation::SourceKind source_kind ) { fSourceKind = source_kind; }

GateEmittedGammaInformation::SourceKind GateEmittedGammaInformation::GetSourceKind() const { return fSourceKind; }

void GateEmittedGammaInformation::SetDecayModel( GateEmittedGammaInformation::DecayModel decay_model ) { fDecayModel = decay_model; }

GateEmittedGammaInformation::DecayModel GateEmittedGammaInformation::GetDecayModel() const { return fDecayModel; }

void GateEmittedGammaInformation::SetGammaKind( GateEmittedGammaInformation::GammaKind gamma_kind ) { fGammaKind = gamma_kind; }

GateEmittedGammaInformation::GammaKind GateEmittedGammaInformation::GetGammaKind() const { return fGammaKind; }

void GateEmittedGammaInformation::SetInitialPolarization( const G4ThreeVector& polarization ) { fInitialPolarization = polarization; }

G4ThreeVector GateEmittedGammaInformation::GetInitialPolarization() const { return fInitialPolarization; }

void GateEmittedGammaInformation::SetTimeShift( const G4double& time_shift ) { fTimeShift = time_shift; }

G4double GateEmittedGammaInformation::GetTimeShift() const { return fTimeShift; }

void GateEmittedGammaInformation::Print() const { G4cout << "GateEmittedGammaInformation" << G4endl; }
