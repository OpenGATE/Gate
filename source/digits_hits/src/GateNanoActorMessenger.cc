/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATENANOACTORMESSENGER_CC
#define GATENANOACTORMESSENGER_CC

#include "GateNanoActorMessenger.hh"
#include "GateNanoActor.hh"

//-----------------------------------------------------------------------------
GateNanoActorMessenger::GateNanoActorMessenger(GateNanoActor* sensor)
  :GateImageActorMessenger(sensor),
  pNanoActor(sensor)
{

  pEnableNanoAbsorptionCmd= 0;
  pSigmaCmd = 0;
  pTimeCmd = 0;
  pDiffusivityCmd = 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateNanoActorMessenger::~GateNanoActorMessenger()
{
  if(pEnableNanoAbsorptionCmd) delete pEnableNanoAbsorptionCmd;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNanoActorMessenger::BuildCommands(G4String base)
{

  G4String  n = base+"/enableNanoAbsorption";
  pEnableNanoAbsorptionCmd = new G4UIcmdWithABool(n, this);
  G4String  guid = G4String("Enable energy absorption by the nano material");
  pEnableNanoAbsorptionCmd->SetGuidance(guid);

  pSigmaCmd = new G4UIcmdWithADouble((base+"/setConvolutionGaussianSigma").c_str(),this);
  pSigmaCmd->SetGuidance("Set sigma of the Gaussian used for the convolution");

  pTimeCmd = new G4UIcmdWithADouble((base+"/setTime").c_str(),this);
  pTimeCmd->SetGuidance("Set the diffusion time");

  pDiffusivityCmd = new G4UIcmdWithADouble((base+"/setDiffusivity").c_str(),this);
  pDiffusivityCmd->SetGuidance("Set the thermal diffusivity");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNanoActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableNanoAbsorptionCmd) pNanoActor->EnableNanoAbsorptionImage(pEnableNanoAbsorptionCmd->GetNewBoolValue(newValue));
  if(cmd == pSigmaCmd) pNanoActor->setGaussianSigma(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pTimeCmd) pNanoActor->setTime(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pDiffusivityCmd) pNanoActor->setDiffusivity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATENANOACTORMESSENGER_CC */
