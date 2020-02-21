/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#ifndef GATESOURCEPHASESPACEMESSENGER_H
#define GATESOURCEPHASESPACEMESSENGER_H 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateMessenger.hh"
#include "GateVSourceMessenger.hh"
#include "GateUIcmdWithAVector.hh"

class GateSourcePhaseSpace;
class GateClock;

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithABool;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

//----------------------------------------------------------------------------------------
class GateSourcePhaseSpaceMessenger: public GateVSourceMessenger
{
public:
  GateSourcePhaseSpaceMessenger(GateSourcePhaseSpace* source);
  ~GateSourcePhaseSpaceMessenger();

  void SetNewValue(G4UIcommand*, G4String);

private:
  GateSourcePhaseSpace*      pSource;
  G4UIcmdWithAString*        AddFileCmd;
  G4UIcmdWithAString*        setParticleTypeCmd;
  G4UIcmdWithoutParameter*   RelativeVolumeCmd;
  G4UIcmdWithoutParameter*   RegularSymmetryCmd;
  G4UIcmdWithoutParameter*   RandomSymmetryCmd;
  G4UIcmdWithABool*          setUseNbParticleAsIntensityCmd;
  G4UIcmdWithABool*          ignoreWeightCmd;
  G4UIcmdWithADoubleAndUnit* setRmaxCmd;
  G4UIcmdWithADoubleAndUnit* setSphereRadiusCmd;
  G4UIcmdWithADouble*        setStartIdCmd;
  G4UIcmdWithAnInteger*      setPytorchBatchSizeCmd;
  G4UIcmdWithAString*        setPytorchParamsCmd;
};
//----------------------------------------------------------------------------------------

#endif // GATESOURCEPHASESPACEMESSENGER_H

#endif // G4ANALYSIS_USE_ROOT
