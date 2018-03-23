/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

#ifndef GateARFDataToRootMessenger_h
#define GateARFDataToRootMessenger_h 1

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateOutputModuleMessenger.hh"
#include "GateARFDataToRoot.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;

class GateARFDataToRootMessenger: public GateOutputModuleMessenger
  {
public:
  GateARFDataToRootMessenger(GateARFDataToRoot* GateARFDataToRoot);
  ~GateARFDataToRootMessenger();

  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateARFDataToRoot* mGateArfDataToRoot;

  G4UIcmdWithAString* mSetArfDataFileCmd;
  G4UIcmdWithADoubleAndUnit* mSetDepth;
  G4UIcmdWithAString* mSmoothDrfCmd;
  };

#endif
#endif

