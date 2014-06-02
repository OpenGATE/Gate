/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#include "GateFragmentationAndProductionActorMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateFragmentationAndProductionActor.hh"

//-----------------------------------------------------------------------------
GateFragmentationAndProductionActorMessenger::GateFragmentationAndProductionActorMessenger(GateFragmentationAndProductionActor * v)
: GateActorMessenger(v),
  pActor(v)
{

  BuildCommands(baseName+pActor->GetObjectName());

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateFragmentationAndProductionActorMessenger::~GateFragmentationAndProductionActorMessenger()
{
  delete pNBinsCmd;
  //delete pEminCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActorMessenger::BuildCommands(G4String base)
{
  pNBinsCmd = new G4UIcmdWithAnInteger((base+"/setNBins").c_str(), this);
  pNBinsCmd->SetGuidance("set number of bins in histograms");
  pNBinsCmd->SetParameterName("nBins",false/*omittable*/);

  //bb = base+"/energyLossHisto/setNumberOfBins";
  //pEdepNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  //guidance = G4String("Set number of bins of the energy loss histogram");
  //pEdepNBinsCmd->SetGuidance(guidance);
  //pEdepNBinsCmd->SetParameterName("Nbins", false);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{

  if(cmd==pNBinsCmd) pActor->SetNBins(pNBinsCmd->GetNewIntValue(newValue));
  //if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif
