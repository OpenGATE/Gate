/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#ifdef G4ANALYSIS_USE_ROOT

#ifndef GATEAUGERDETECTORACTORMESSENGER_CC
#define GATEAUGERDETECTORACTORMESSENGER_CC

#include "GateAugerDetectorActorMessenger.hh"
#include "GateAugerDetectorActor.hh"


//-----------------------------------------------------------------------------
GateAugerDetectorActorMessenger::GateAugerDetectorActorMessenger(GateAugerDetectorActor * v)
: GateActorMessenger(v),
  pActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateAugerDetectorActorMessenger::~GateAugerDetectorActorMessenger()
{
  delete pMaxTOFCmd;
  delete pMinEdepCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActorMessenger::BuildCommands(G4String base)
{
  pMaxTOFCmd = new G4UIcmdWithADoubleAndUnit((base+"/setMaxTOF").c_str(),this); 
  pMaxTOFCmd->SetGuidance("Set maximum time of flight window");
  pMaxTOFCmd->SetParameterName("MaxTOF",false);
  pMaxTOFCmd->SetDefaultUnit("ns");

  pMinEdepCmd = new G4UIcmdWithADoubleAndUnit((base+"/setMinEdep").c_str(),this); 
  pMinEdepCmd->SetGuidance("Set minimum energy deposition to trigger detection");
  pMinEdepCmd->SetParameterName("MinEdep",false);
  pMinEdepCmd->SetDefaultUnit("MeV");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateAugerDetectorActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pMaxTOFCmd) pActor->setMaxTOF(  pMaxTOFCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pMinEdepCmd) pActor->setMinEdep(  pMinEdepCmd->GetNewDoubleValue(newValue)  ) ;

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEAUGERDETECTORACTORMESSENGER_CC */
#endif
