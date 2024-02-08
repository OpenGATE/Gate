/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEENEFILTERMESSENGER_CC
#define GATEENEFILTERMESSENGER_CC

#include "GateEnergyFilterMessenger.hh"

#include "GateEnergyFilter.hh"



//-----------------------------------------------------------------------------
GateEnergyFilterMessenger::GateEnergyFilterMessenger(GateEnergyFilter* partFilter)

  : pEnergyFilter(partFilter)
{
  BuildCommands(pEnergyFilter->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateEnergyFilterMessenger::~GateEnergyFilterMessenger()
{
  delete pSetEminCmd;
  delete pSetEmaxCmd;
  delete pInvertCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergyFilterMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;
  
  bb = base+"/setEmin";
  pSetEminCmd = new G4UIcmdWithADoubleAndUnit(bb,this);  
  guidance = G4String("Set the Emin");
  pSetEminCmd->SetGuidance(guidance);
  pSetEminCmd->SetParameterName("Emin", false);
  pSetEminCmd->SetDefaultUnit("MeV");

  bb = base+"/setEmax";
  pSetEmaxCmd = new G4UIcmdWithADoubleAndUnit(bb,this);  
  guidance = G4String("Set the Emax");
  pSetEmaxCmd->SetGuidance(guidance);
  pSetEmaxCmd->SetParameterName("Emax", false);
  pSetEmaxCmd->SetDefaultUnit("MeV");

  bb = base+"/invert";
  pInvertCmd = new G4UIcmdWithoutParameter(bb,this);
  guidance = "Invert the filter";
  pInvertCmd->SetGuidance(guidance);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergyFilterMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pSetEminCmd) 
    {
      pEnergyFilter->SetEmin(pSetEminCmd->GetNewDoubleValue(param) );
    }

  if(command == pSetEmaxCmd) 
    {
      pEnergyFilter->SetEmax(pSetEmaxCmd->GetNewDoubleValue(param) );
    }
  if(command==pInvertCmd)
    pEnergyFilter->setInvert();
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEENEFILTERMESSENGER_CC */
