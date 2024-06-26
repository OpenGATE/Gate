/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateDistributionMessenger.hh"
#include "GateVDistribution.hh"
#include "GateMessageManager.hh"

#include "G4UIdirectory.hh"
#include "G4UnitsTable.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

// Constructor
GateDistributionMessenger::GateDistributionMessenger(GateVDistribution* itsDistribution,
    			     const G4String& itsDirectoryName)
: GateNamedObjectMessenger( itsDistribution,itsDirectoryName)
{
  G4String cmdName,guidance;

  cmdName = GetDirectoryName()+"getMinX";
  guidance = "Get the definition domain's minimum value";
  getMinX_Cmd = new G4UIcmdWithoutParameter(cmdName,this);
  getMinX_Cmd->SetGuidance(guidance);

  cmdName = GetDirectoryName()+"getMaxX";
  guidance = "Get the definition domain's maximum value";
  getMaxX_Cmd = new G4UIcmdWithoutParameter(cmdName,this);
  getMaxX_Cmd->SetGuidance(guidance);

  cmdName = GetDirectoryName()+"getMinY";
  guidance = "Get the image range's minimum value";
  getMinY_Cmd = new G4UIcmdWithoutParameter(cmdName,this);
  getMinY_Cmd->SetGuidance(guidance);

  cmdName = GetDirectoryName()+"getMaxY";
  guidance = "Get the image range's maximum value";
  getMaxY_Cmd = new G4UIcmdWithoutParameter(cmdName,this);
  getMaxY_Cmd->SetGuidance(guidance);

  cmdName = GetDirectoryName()+"getRandom";
  guidance = "Shoot a random number following the specified distribution";
  getRandom_Cmd = new G4UIcmdWithoutParameter(cmdName,this);
  getRandom_Cmd->SetGuidance(guidance);

  cmdName = GetDirectoryName()+"getValue";
  guidance = "Retrieve a value at specified x";
  getValueCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  getValueCmd->SetGuidance(guidance);
  getValueCmd->SetParameterName("x",false);


}



// Destructor
GateDistributionMessenger::~GateDistributionMessenger()
{
  delete getMinX_Cmd;
  delete getMinY_Cmd;
  delete getMaxX_Cmd;
  delete getMaxY_Cmd;
  delete getRandom_Cmd;
  delete getValueCmd;

}
G4String GateDistributionMessenger::withUnity(G4double value,G4String category) const
{
    std::ostringstream ss;
    if (category.empty() || category == "None")
    	ss<<value<<" (unitless)";
    else
    	ss<<G4BestUnit(value,category);
    return ss.str();
}
// UI command interpreter method
void GateDistributionMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if       ( command==getValueCmd ){
    G4double x = getValueCmd->GetNewDoubleValue(newValue);
    G4double y = GetDistribution()->Value(x);}
else if( command==getMinX_Cmd ) {
    G4double x = GetDistribution()->MinX();
    G4cout<<GetDistribution()->GetObjectName()<<" MinX "<<withUnity(x,UnitCategoryX())<< Gateendl;
  } else if( command==getMinY_Cmd ) {
    G4double x = GetDistribution()->MinY();
    G4cout<<GetDistribution()->GetObjectName()<<" MinY "<<withUnity(x,UnitCategoryY())<< Gateendl;
  } else if( command==getMaxX_Cmd ) {
    G4double x = GetDistribution()->MaxX();
    G4cout<<GetDistribution()->GetObjectName()<<" MaxX "<<withUnity(x,UnitCategoryX())<< Gateendl;
  } else if( command==getMaxY_Cmd ) {
    G4double x = GetDistribution()->MaxY();
    G4cout<<GetDistribution()->GetObjectName()<<" MaxY "<<withUnity(x,UnitCategoryY())<< Gateendl;
  } else if( command==getRandom_Cmd ) {
    G4double x = GetDistribution()->ShootRandom();
    G4cout<<GetDistribution()->GetObjectName()<<" Random "<<withUnity(x,UnitCategoryX())<< Gateendl;
  }
  else
    GateNamedObjectMessenger::SetNewValue(command,newValue);
}
void GateDistributionMessenger::SetUnitX(const G4String& unitX)
{
    m_unitX = unitX;
    getValueCmd->SetUnitCategory(UnitCategoryX());
}
void GateDistributionMessenger::SetUnitY(const G4String& unitY)
{
    m_unitY = unitY;
}
