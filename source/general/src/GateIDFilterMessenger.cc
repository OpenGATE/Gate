/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEIDFILTERMESSENGER_CC
#define GATEIDFILTERMESSENGER_CC

#include "GateIDFilterMessenger.hh"

#include "GateIDFilter.hh"



//-----------------------------------------------------------------------------
GateIDFilterMessenger::GateIDFilterMessenger(GateIDFilter* idFilter)

  : pIDFilter(idFilter)
{
  BuildCommands(pIDFilter->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateIDFilterMessenger::~GateIDFilterMessenger()
{
  delete pAddIDCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateIDFilterMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;
  
  bb = base+"/selectID";
  pAddIDCmd = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Select ID";
  pAddIDCmd->SetGuidance(guidance);
  pAddIDCmd->SetParameterName("ID",false);

  bb = base+"/selectParentID";
  pAddParentIDCmd = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Select parent ID";
  pAddParentIDCmd->SetGuidance(guidance);
  pAddParentIDCmd->SetParameterName("Parent ID",false);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateIDFilterMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command==pAddIDCmd)
    pIDFilter->addID(pAddIDCmd->GetNewIntValue(param));
  if(command==pAddParentIDCmd)
    pIDFilter->addParentID(pAddParentIDCmd->GetNewIntValue(param));
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEIDFILTERMESSENGER_CC */
