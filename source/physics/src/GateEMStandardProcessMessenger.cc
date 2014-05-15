/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#ifndef GATEGENERALEMPROCESSMESSENGER_CC
#define GATEGENERALEMPROCESSMESSENGER_CC

#include "GateEMStandardProcessMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

//-----------------------------------------------------------------------------
GateEMStandardProcessMessenger::GateEMStandardProcessMessenger(GateVProcess *pb):GateVProcessMessenger(pb)
{
  mPrefix = mPrefix+"processes/";
  pSetStepFctCmd=0;
  pSetLinearlossLimit = 0;
  BuildCommands(pb->GetG4ProcessName() );

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateEMStandardProcessMessenger::~GateEMStandardProcessMessenger()
{
  if(pSetStepFctCmd) delete pSetStepFctCmd;
  if(pSetLinearlossLimit) delete pSetLinearlossLimit;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEMStandardProcessMessenger::BuildCommands(G4String base)
{

  G4String baseModel ="";
  baseModel +=  mPrefix;
  baseModel += base;

  G4String bb;
  G4String guidance;

  bb = baseModel+"/setStepFunction";
  pSetStepFctCmd = new GateUIcmdWithAStringADoubleAndADoubleWithUnit(bb,this);
  guidance = "Set step function";
  pSetStepFctCmd->SetGuidance(guidance);
  pSetStepFctCmd->SetParameterName("Particle or Group of particles","Ratio (step/range)", "Final range", "Unit", false,false,false,false);

  bb = baseModel+"/setLinearLossLimit";
  pSetLinearlossLimit = new GateUIcmdWithAStringAndADouble(bb,this);
  guidance = "Set linear loss limit";
  pSetLinearlossLimit->SetGuidance(guidance);
  pSetLinearlossLimit->SetParameterName("Particle or Group of particles","Limit", false,false);

  BuildModelsCommands(base);
  BuildWrapperCommands(base);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEMStandardProcessMessenger::SetNewValue(G4UIcommand* command, G4String param)
{

  if(command==pSetStepFctCmd){
      char par1[30];
      char par4[30];
      double par2;
      double par3;
      std::istringstream is(param);
      is >> par1 >> par2 >> par3 >> par4;
      std::ostringstream par;
      par<<par3;
      G4String val(par.str());
      val.append(" ");
      val.append(par4);
      par3 = G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(val);
      pProcess->SetStepFunction(par1,par2,par3);
  }

  if(command==pSetLinearlossLimit){
     char par1[30];
     double par2;
     std::istringstream is(param);
     is >> par1 >> par2 ;

     pProcess->SetLinearlosslimit(par1 , par2  );
  }

  SetModelsNewValue(command,param);
  SetWrapperNewValue(command,param);

}
//-----------------------------------------------------------------------------

#endif /* end #define GATEGENRALEMPROCESSMESSENGER_CC */
