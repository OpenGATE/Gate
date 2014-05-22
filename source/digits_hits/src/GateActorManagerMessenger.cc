/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEACTORMANAGERMESSENGER_CC
#define GATEACTORMANAGERMESSENGER_CC

#include "GateActorManagerMessenger.hh"

#include "GateActorManager.hh"

#include "G4UIdirectory.hh"


//-----------------------------------------------------------------------------
GateActorManagerMessenger::GateActorManagerMessenger(GateActorManager* sMan)

  : pActorManager(sMan)
{
  //  G4cout << " Constructeur GateActorManagerMessenger " << G4endl;
  BuildCommands("/gate/actor");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateActorManagerMessenger::~GateActorManagerMessenger()
{
  delete pAddActor;
  delete pActorCommand;
  delete pInitActor;
  delete pResetAfterSaving;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateActorManagerMessenger::BuildCommands(G4String base)
{
  //  G4cout << " GateActorManagerMessenger::BuildCommands " << G4endl;

  pActorCommand = new G4UIdirectory("/gate/actor/");
  pActorCommand->SetGuidance("GATE actors control.");

  G4String guidance;
  G4String bb;
  bb = base+"/addActor";

  //  G4cout << " base addActor " << bb << G4endl;
  pAddActor = new GateUIcmdWith2String(bb,this);
  guidance = "Add a new sensor";
  pAddActor->SetGuidance(guidance);
  pAddActor->SetParameterName("Type","Name",false,false);

  bb = base+"/init";
  pInitActor = new G4UIcmdWithoutParameter(bb,this);
  guidance = "Initialization of sensors";
  pInitActor->SetGuidance(guidance);

  pResetAfterSaving = new G4UIcmdWithABool((base+"/resetAfterSaving").c_str(),this);
  pResetAfterSaving->SetGuidance("reset actor after saving results. usefull for checkpointing.");
  pResetAfterSaving->SetParameterName("reset",false);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateActorManagerMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if (command==pAddActor) {
    std::string par1,par2;
    std::istringstream is(param);
    is >> par1 >> par2 ;
    pActorManager->AddActor(par1,par2);
  }

  if (command==pInitActor)
    GateActorManager::GetInstance() ->CreateListsOfEnabledActors();

  if (command==pResetAfterSaving)
    GateActorManager::GetInstance()->SetResetAfterSaving( G4UIcmdWithABool::GetNewBoolValue(param) );
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEACTORMANAGERMESSENGER_CC */
