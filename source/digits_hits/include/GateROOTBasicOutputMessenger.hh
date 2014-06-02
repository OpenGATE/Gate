/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#ifndef GateROOTBasicOutputMessenger_h
#define GateROOTBasicOutputMessenger_h 1
#include "GateUserActions.hh"

#include "globals.hh"
#include "G4UImessenger.hh"
#include "G4ios.hh"

#include "GateActions.hh"

class GateUserActions;
class GateRunAction;
class GateEventAction;

//---------------------------------------------------------------------------------
class GateROOTBasicOutput;
class G4UIdirectory;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithADouble;
class G4UIcmdWithAString;
//---------------------------------------------------------------------------------


//---------------------------------------------------------------------------------
class GateROOTBasicOutputMessenger: public G4UImessenger
{
public:

  GateROOTBasicOutputMessenger(GateROOTBasicOutput*);
  ~GateROOTBasicOutputMessenger();

  void SetNewValue(G4UIcommand* ,G4String );

private:

  GateROOTBasicOutput*            xeHisto;
  G4UIdirectory*                  plotDir;
  G4UIcmdWithAString*             setfileNameCmd;
};
//---------------------------------------------------------------------------------
#endif
#endif
