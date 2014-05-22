/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEGENERALHADPROCESSMESSENGER_CC
#define GATEGENERALHADPROCESSMESSENGER_CC

#include "GateHadronicStandardProcessMessenger.hh"


//-----------------------------------------------------------------------------
GateHadronicStandardProcessMessenger::GateHadronicStandardProcessMessenger(GateVProcess *pb):GateVProcessMessenger(pb)
{
  mPrefix = mPrefix+"processes/";
  BuildCommands(pb->GetG4ProcessName() );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronicStandardProcessMessenger::BuildCommands(G4String base)
{
  BuildModelsCommands(base);
  BuildEnergyRangeModelsCommands(base);
  BuildDataSetCommands(base);
  BuildWrapperCommands(base);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHadronicStandardProcessMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  SetEnergyRangeModelsNewValue(command,param);
  SetDataSetNewValue(command,param);
  SetWrapperNewValue(command,param);
 // SetModelsNewValue(command,param);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEGENRALHADPROCESSMESSENGER_CC */
