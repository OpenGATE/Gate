/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#include "GateMultiScatteringMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

//-----------------------------------------------------------------------------
GateMultiScatteringMessenger::GateMultiScatteringMessenger(GateVProcess *pb):GateEMStandardProcessMessenger(pb)
{
  
  pSetDistanceToBoundary=0;
  BuildCommands(pb->GetG4ProcessName() );

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateMultiScatteringMessenger::~GateMultiScatteringMessenger()
{
  if(pSetDistanceToBoundary) delete pSetDistanceToBoundary;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMultiScatteringMessenger::BuildCommands(G4String base)
{

  G4String baseModel ="";
  baseModel +=  mPrefix;
  baseModel += base;

  G4String bb;
  G4String guidance;

  bb = baseModel+"/setGeometricalStepLimiterType";
  pSetDistanceToBoundary = new GateUIcmdWith2String(bb,this);
  guidance = "Set geometrical step limiter type";
  pSetDistanceToBoundary->SetGuidance(guidance);
  pSetDistanceToBoundary->SetParameterName("Particle or Group of particles","Limit type", false,false);
  //pSetDistanceToBoundary->SetCandidates(" ","safety distanceToBoundary");


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMultiScatteringMessenger::SetNewValue(G4UIcommand* command, G4String param)
{

  if(command==pSetDistanceToBoundary){
      char par1[30];
      char par4[30];
      std::istringstream is(param);
      is >> par1 >>  par4;

      pProcess->SetMsclimitation(par1,par4);
  }

  GateEMStandardProcessMessenger::SetNewValue( command,  param);

}
//-----------------------------------------------------------------------------

