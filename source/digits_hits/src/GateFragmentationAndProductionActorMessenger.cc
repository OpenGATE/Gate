/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#ifdef G4ANALYSIS_USE_ROOT

#ifndef GATEFRAGMENTATIONANDPRODUCTIONACTORMESSENGER_CC
#define GATEFRAGMENTATIONANDPRODUCTIONACTORMESSENGER_CC

#include "GateFragmentationAndProductionActorMessenger.hh"
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
  //delete pEmaxCmd;
  //delete pEminCmd;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActorMessenger::BuildCommands(G4String base)
{
  //bb = base+"/energySpectrum/setEmin";
  //pEminCmd = new G4UIcmdWithADoubleAndUnit(bb, this); 
  //guidance = G4String("Set minimum energy of the energy spectrum");
  //pEminCmd->SetGuidance(guidance);
  //pEminCmd->SetParameterName("Emin", false);
  //pEminCmd->SetDefaultUnit("MeV");

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

  //if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEFRAGMENTATIONANDPRODUCTIONACTORMESSENGER_CC */
#endif
