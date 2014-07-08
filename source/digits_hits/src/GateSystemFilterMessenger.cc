/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateSystemFilterMessenger.hh"
#include "GateSystemFilter.hh"
#include "GateSystemListManager.hh"

GateSystemFilterMessenger::GateSystemFilterMessenger(GateSystemFilter* itsSystemFilter)
   : GatePulseProcessorMessenger(itsSystemFilter),
   m_insertedSystems("")
{
   ObtainCandidates();
   G4String cmdName;

   cmdName = GetDirectoryName()+"selectSystem";
   m_SetSystemNameCmd = new G4UIcmdWithAString(cmdName,this);
   m_SetSystemNameCmd->SetGuidance("Select the system which the pulses come from");
   m_SetSystemNameCmd->SetParameterName("systemName",false);
   m_SetSystemNameCmd->SetCandidates(m_insertedSystems);

}

GateSystemFilterMessenger::~GateSystemFilterMessenger()
{
delete m_SetSystemNameCmd;
}

//=============================================================================
//=============================================================================
void GateSystemFilterMessenger::SetNewValue(G4UIcommand* aCommand, G4String aString)
{
   if (aCommand==m_SetSystemNameCmd )
   {
      GetSystemFilter()->SetSystemName(aString);
      GetSystemFilter()->SetSystemToItsChain();
   }

   else
      GatePulseProcessorMessenger::SetNewValue(aCommand,aString);

}

//=============================================================================
//=============================================================================
void GateSystemFilterMessenger::ObtainCandidates()
{
   GateSystemListManager* systemListManager = GateSystemListManager::GetInstance();

   size_t NISN = systemListManager->GetInsertedSystemsNames()->size();

   for(size_t i=0; i<NISN; i++)
   {
      m_insertedSystems += systemListManager->GetInsertedSystemsNames()->at(i);
            if(i < (NISN-1))
               m_insertedSystems += " ";
   }

}
