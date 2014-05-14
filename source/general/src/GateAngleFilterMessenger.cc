/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEANGLEFILTERMESSENGER_CC
#define GATEANGLEFILTERMESSENGER_CC

#include "GateAngleFilterMessenger.hh"

#include "GateAngleFilter.hh"



//-----------------------------------------------------------------------------
GateAngleFilterMessenger::GateAngleFilterMessenger(GateAngleFilter* partFilter)

  : pAngleFilter(partFilter)
{
  BuildCommands(pAngleFilter->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateAngleFilterMessenger::~GateAngleFilterMessenger()
{
  delete pSetDirectionCmd;
  delete pSetAngleCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateAngleFilterMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;
  
  bb = base+"/setAngle";
  pSetAngleCmd = new G4UIcmdWithADoubleAndUnit(bb,this);  
  guidance = G4String("Set the angle");
  pSetAngleCmd->SetGuidance(guidance);
  pSetAngleCmd->SetParameterName("Angle", false);
  pSetAngleCmd->SetDefaultUnit("deg");

  bb = base+"/setDirection";
  pSetDirectionCmd = new G4UIcmdWith3Vector(bb,this);  
  guidance = G4String("Sets direction");
  pSetDirectionCmd->SetGuidance(guidance);
  pSetDirectionCmd->SetParameterName("direction_x","direction_y", "direction_z", false, false);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateAngleFilterMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
   if(command == pSetAngleCmd) 
  {
    pAngleFilter->SetAngle(pSetAngleCmd->GetNewDoubleValue(param) );
  }

  if(command == pSetDirectionCmd) 
  {
    pAngleFilter->SetMomentum( pSetDirectionCmd->GetNew3VectorValue(param) );
  }
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEANGLEFILTERMESSENGER_CC */
