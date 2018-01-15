

#include "GateComptonCameraActorMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateComptonCameraActor.hh"
#include "G4SystemOfUnits.hh"

//-----------------------------------------------------------------------------
GateComptonCameraActorMessenger::GateComptonCameraActorMessenger(GateComptonCameraActor * v)
  : GateActorMessenger(v),
    pActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateComptonCameraActorMessenger::~GateComptonCameraActorMessenger()
{
  delete pEmaxCmd;
  delete pEminCmd;
  delete pNBinsCmd;
  delete pEdepmaxCmd;
  delete pEdepminCmd;
  delete pEdepNBinsCmd;
  delete pSaveAsText;
  delete pNVolCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/comptonCamera/setEmin";
  pEminCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum energy of the energy spectrum");
  pEminCmd->SetGuidance(guidance);
  pEminCmd->SetParameterName("Emin", false);
  pEminCmd->SetDefaultUnit("MeV");

  
  bb = base+"/comptonCamera/setEmax";
  pEmaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum energy of the energy spectrum");
  pEmaxCmd->SetGuidance(guidance);
  pEmaxCmd->SetParameterName("Emax", false);
  pEmaxCmd->SetDefaultUnit("MeV");

  bb = base+"/comptonCamera/setNumberOfBins";
  pNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the energy spectrum");
  pNBinsCmd->SetGuidance(guidance);
  pNBinsCmd->SetParameterName("Nbins", false);

  bb = base+"/energyLossHisto/setEmin";
  pEdepminCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum energy of the energy loss histogram");
  pEdepminCmd->SetGuidance(guidance);
  pEdepminCmd->SetParameterName("Emin", false);
  pEdepminCmd->SetDefaultUnit("MeV");

  bb = base+"/energyLossHisto/setEmax";
  pEdepmaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum energy of the energy loss histogram");
  pEdepmaxCmd->SetGuidance(guidance);
  pEdepmaxCmd->SetParameterName("Emax", false);
  pEdepmaxCmd->SetDefaultUnit("MeV");

  bb = base+"/energyLossHisto/setNumberOfBins";
  pEdepNBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the energy loss histogram");
  pEdepNBinsCmd->SetGuidance(guidance);
  pEdepNBinsCmd->SetParameterName("Nbins", false);

  // More to set the number of subvolumes but I should be avaible to know them
  bb = base+"/volumeNameHisto/setNumberOfBins";
  pNVolCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins for the volume name histogram");
  pNVolCmd->SetGuidance(guidance);
  pNVolCmd->SetParameterName("Nbins", false);

  bb = base+"/saveAsText";
  pSaveAsText = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition to root output files, also write .txt files (that can be open as a source, 'UserSpectrum')");
  pSaveAsText->SetGuidance(guidance);

  

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEmaxCmd) pActor->SetEmax(  pEmaxCmd->GetNewDoubleValue(newValue)  ) ;


  if(cmd == pNVolCmd) pActor->SetVolIDNBins( pNVolCmd->GetNewIntValue(newValue)) ;

  
  if(cmd == pNBinsCmd) pActor->SetENBins(  pNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pEdepminCmd) pActor->SetEdepmin(  pEdepminCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEdepmaxCmd) pActor->SetEdepmax(  pEdepmaxCmd->GetNewDoubleValue(newValue)  ) ;
  if(cmd == pEdepNBinsCmd) pActor->SetEdepNBins(  pEdepNBinsCmd->GetNewIntValue(newValue)  ) ;
  if(cmd == pSaveAsText) pActor->SetSaveAsTextFlag(  pSaveAsText->GetNewBoolValue(newValue)  ) ;

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

#endif
