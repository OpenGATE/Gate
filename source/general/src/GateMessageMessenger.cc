/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateMessageMessenger.hh"


#include "GateMessageManager.hh"


//-----------------------------------------------------------------------------
// Ctor
GateMessageMessenger::GateMessageMessenger(G4String base, GateMessageManager* man)
 :pMessageManager(man)
{

  G4String cmd = base+"/verbose";

  pVerboseCmd = new G4UIcmdWithAString(cmd, this);
  pVerboseCmd->SetGuidance("Set level of verbosity for a given type of message");
  pVerboseCmd->SetParameterName("level of verbosity", true);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Dtor
GateMessageMessenger::~GateMessageMessenger()
{
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMessageMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  //GateMessage("Manager",5,"GateMessageMessenger::SetNewValue("<<newValue<<")"<<Gateendl);
  if (cmd == pVerboseCmd) {
    str_size pos = newValue.index (' ');
    if (pos<newValue.length()) {
      G4String code = newValue.substr(0,pos);
      G4String svalue = newValue.substr(pos,newValue.length()-pos);
      int value = atoi(svalue);  

      pMessageManager->SetMessageLevel(code,value);
    }
    else if (newValue==G4String("print")) {
      pMessageManager->PrintInfo();
    }
    else {
      GateWarning("Bad syntax in '/gate/verbose "<<newValue<<"' command"); 
    }
  } 
}
//-----------------------------------------------------------------------------


