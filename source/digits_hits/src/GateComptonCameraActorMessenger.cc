//
// Created by Ane Etxebeste 2019
//

/*!
  \class  GateComptonCameraActorMessenger
*/

#include <G4SystemOfUnits.hh>
#include "GateComptonCameraActorMessenger.hh"
#include "GateComptonCameraActor.hh"

//-----------------------------------------------------------------------------
GateComptonCameraActorMessenger::GateComptonCameraActorMessenger(GateComptonCameraActor * v)
  : GateActorMessenger(v),
    pActor(v)
{
    G4cout<<"buildCommand ComptonCamera messenger"<<G4endl;
  BuildCommands(baseName+pActor->GetObjectName());


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateComptonCameraActorMessenger::~GateComptonCameraActorMessenger()
{
    delete pSaveHitsTree;
    delete pSaveSinglesTree;
    delete pSaveCoincidencesTree;
    delete pSaveCoincidenceChainsTree;
    delete pSaveEventInfoTree;


    delete pNameOfAbsorberSDVol;
    delete pNameOfScattererSDVol;
    delete pNumberofDiffScattererLayers;
    delete pNumberofTotScattererLayers;

    delete   pSourceParentIDSpecification;
    delete   pFileName4SourceParentID;

    //singles varaibles included
    delete pEnableEnergyCmd;
    delete pEnableEnergyIniCmd;
    delete pEnableEnergyFinCmd;
    delete pEnableTimeCmd;
    delete pEnablePositionXCmd;
    delete pEnablePositionYCmd;
    delete pEnablePositionZCmd;
    delete pEnableSourcePositionXCmd;
    delete pEnableLocalPositionYCmd;
    delete pEnableLocalPositionZCmd;
    delete pEnableSourcePositionXCmd;
    delete pEnableSourcePositionYCmd;
    delete pEnableSourcePositionZCmd;
    //volume identification
    delete pEnableVolumeIDCmd;
    //delete pEnableLayerNameCmd;
    delete pEnableSourceEnergyCmd;
    delete pEnableSourcePDGCmd;
    delete pEnablenCrystalConvCmd;
    delete pEnablenCrystalComptCmd;
    delete pEnablenCrystalRaylCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;


  bb = base+"/saveHitsTree";
  pSaveHitsTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree wit the hit info inside the attachedVolume");
  pSaveHitsTree->SetGuidance(guidance);

  bb = base+"/saveSinglesTree";
  pSaveSinglesTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree wit the singles info inside the attachedVolume");
  pSaveSinglesTree->SetGuidance(guidance);

  bb = base+"/saveCoincidencesTree";
  pSaveCoincidencesTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree with the coincidences info inside the attachedVolume");
  pSaveCoincidencesTree->SetGuidance(guidance);

  bb = base+"/saveEventInfoTree";
  pSaveEventInfoTree= new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a  file with  some extra info of each event such as electron escape");
  pSaveEventInfoTree->SetGuidance(guidance);



  bb = base+"/saveCoincidenceChainsTree";
  pSaveCoincidenceChainsTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree wit the coincidence chain info inside the attachedVolume");
  pSaveCoincidenceChainsTree->SetGuidance(guidance);






  bb = base+"/absorberSDVolume";
  pNameOfAbsorberSDVol = new G4UIcmdWithAString(bb,this);
  guidance = "Specifies the absorber volume to track particles";
  pNameOfAbsorberSDVol->SetGuidance(guidance);
  //pNameOfAbsorberSDVol->SetParameterName(" absorber SD Volume name",false);




  bb = base+"/scattererSDVolume";
  pNameOfScattererSDVol = new G4UIcmdWithAString(bb,this);
  guidance = "Specifies the scatterer  volume to track particles";
  pNameOfScattererSDVol->SetGuidance(guidance);
  //pNameOfScattererSDVol->SetParameterName(" scatterer SD Volume name",false);


  bb = base+"/numberOfDiffScatterers";
  pNumberofDiffScattererLayers = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Specifies the number of different  scatterer layers non repeaters.";
  pNumberofDiffScattererLayers->SetGuidance(guidance);
  //The name of the layers must me the same but for a number. When the repeaters are used that number is set by the copyNumber,
  //if the user generated name, nameNumb Study!!)

  bb = base+"/numberOfTotScatterers";
  pNumberofTotScattererLayers = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Specifies the number of different  scatterer layers non repeaters.";
  pNumberofTotScattererLayers->SetGuidance(guidance);



  bb = base+"/specifysourceParentID";
  pSourceParentIDSpecification= new G4UIcmdWithABool(bb, this);
  guidance = G4String("By deflaut set to zero and parentID=0 particles are considered to register sourceEkine and sourcePDG information");
  pSourceParentIDSpecification->SetGuidance(guidance);

  bb = base+"/parentIDFileName";
  pFileName4SourceParentID =new G4UIcmdWithAString(bb,this);
  guidance = "File name where the parentID are specified. One integer number per line.";
  pFileName4SourceParentID->SetGuidance(guidance);


  bb = base+"/enableEnergy";
  pEnableEnergyCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save energy deposition in the SD.";
  pEnableEnergyCmd->SetGuidance(guidance);
  pEnableEnergyCmd->SetParameterName("State",false);

  bb = base+"/enableEnergyIni";
  pEnableEnergyIniCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save the energy of the photon before the interaction. Only available when using idealComptPhotadder. Otherwise dummy values (-1)";
  pEnableEnergyIniCmd->SetGuidance(guidance);
  pEnableEnergyIniCmd->SetParameterName("State",false);


  bb = base+"/enableEnergyFin";
  pEnableEnergyFinCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save the energy of the photon after the interaction. Only available when using idealComptPhotadder. Otherwise dummy values (-1)";
  pEnableEnergyFinCmd->SetGuidance(guidance);
  pEnableEnergyFinCmd->SetParameterName("State",false);


  bb = base+"/enableTime";
  pEnableTimeCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save the timne";
  pEnableTimeCmd->SetGuidance(guidance);
  pEnableTimeCmd->SetParameterName("State",false);

  bb = base+"/enablePositionX";
  pEnablePositionXCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save global X position)";
  pEnablePositionXCmd->SetGuidance(guidance);
  pEnablePositionXCmd->SetParameterName("State",false);

  bb = base+"/enablePositionY";
  pEnablePositionYCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save global Y position)";
  pEnablePositionYCmd->SetGuidance(guidance);
  pEnablePositionYCmd->SetParameterName("State",false);


  bb = base+"/enablePositionZ";
  pEnablePositionZCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save global Z position)";
  pEnablePositionZCmd->SetGuidance(guidance);
  pEnablePositionZCmd->SetParameterName("State",false);


  bb = base+"/enableLocalPositionX";
  pEnableLocalPositionXCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save local X position)";
  pEnableLocalPositionXCmd->SetGuidance(guidance);
  pEnableLocalPositionXCmd->SetParameterName("State",false);

  bb = base+"/enableLocalPositionY";
  pEnableLocalPositionYCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save local Y position)";
  pEnableLocalPositionYCmd->SetGuidance(guidance);
  pEnableLocalPositionYCmd->SetParameterName("State",false);


  bb = base+"/enableLocalPositionZ";
  pEnableLocalPositionZCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save local Z position)";
  pEnableLocalPositionZCmd->SetGuidance(guidance);
  pEnableLocalPositionZCmd->SetParameterName("State",false);


  bb = base+"/enableSourcePositionX";
  pEnableSourcePositionXCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save source global X position)";
  pEnableSourcePositionXCmd->SetGuidance(guidance);
  pEnableSourcePositionXCmd->SetParameterName("State",false);

  bb = base+"/enableSourcePositionY";
  pEnableSourcePositionYCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save source global Y position)";
  pEnableSourcePositionYCmd->SetGuidance(guidance);
  pEnableSourcePositionYCmd->SetParameterName("State",false);


  bb = base+"/enableSourcePositionZ";
  pEnableSourcePositionZCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save source global Z position)";
  pEnableSourcePositionZCmd->SetGuidance(guidance);
  pEnableSourcePositionZCmd->SetParameterName("State",false);

  bb = base+"/enableSourceEnergy";
  pEnableSourceEnergyCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save source energy)";
  pEnableSourceEnergyCmd->SetGuidance(guidance);
  pEnableSourceEnergyCmd->SetParameterName("State",false);


  bb = base+"/enableSourcePDG";
  pEnableSourcePDGCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save source energy)";
  pEnableSourcePDGCmd->SetGuidance(guidance);
  pEnableSourcePDGCmd->SetParameterName("State",false);

  bb = base+"/enableVolumeID";
  pEnableVolumeIDCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save volumeID)";
  pEnableVolumeIDCmd->SetGuidance(guidance);
  pEnableVolumeIDCmd->SetParameterName("State",false);

  bb = base+"/enablenCrystalCompt";
  pEnablenCrystalComptCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save number of Compt interactions occurred before)";
  pEnablenCrystalComptCmd->SetGuidance(guidance);
  pEnablenCrystalComptCmd->SetParameterName("State",false);

  bb = base+"/enablenCrystalRayl";
  pEnablenCrystalRaylCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save number of Rayl interactions occurred before)";
  pEnablenCrystalRaylCmd->SetGuidance(guidance);
  pEnablenCrystalRaylCmd->SetParameterName("State",false);

  bb = base+"/enablenCrystalConv";
  pEnablenCrystalConvCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save number of Pair Creation interactions occurred before)";
  pEnablenCrystalConvCmd->SetGuidance(guidance);
  pEnablenCrystalConvCmd->SetParameterName("State",false);


  // pEnableLayerNameCmd;


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pSaveHitsTree) pActor->SetSaveHitsTreeFlag(  pSaveHitsTree->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveSinglesTree) pActor->SetSaveSinglesTreeFlag(  pSaveSinglesTree->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveCoincidencesTree) pActor->SetSaveCoincidencesTreeFlag(  pSaveCoincidencesTree->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveCoincidenceChainsTree) pActor->SetSaveCoincidenceChainsTreeFlag(  pSaveCoincidenceChainsTree->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveEventInfoTree) pActor->SetSaveEventInfoTreeFlag(  pSaveEventInfoTree->GetNewBoolValue(newValue)  ) ;


  if(cmd == pNumberofDiffScattererLayers) pActor->SetNumberOfDiffScattererLayers( pNumberofDiffScattererLayers->GetNewIntValue(newValue)  ) ;
   if(cmd == pNumberofTotScattererLayers) pActor->SetNumberOfTotScattererLayers( pNumberofTotScattererLayers->GetNewIntValue(newValue)  ) ;
  if(cmd == pNameOfScattererSDVol) pActor->SetNameOfScattererSDVol(newValue) ;
  if(cmd == pNameOfAbsorberSDVol) pActor->SetNameOfAbsorberSDVol( newValue ) ;

  if(cmd ==  pSourceParentIDSpecification) pActor->SetParentIDSpecificationFlag( pSourceParentIDSpecification->GetNewBoolValue(newValue)  ) ;
  if(cmd == pFileName4SourceParentID) pActor->SetParentIDFileName(newValue) ;
  //

  //flags for singles output
  if(cmd == pEnableEnergyCmd) pActor->SetIsEnergyEnabled(pEnableEnergyCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableEnergyIniCmd) pActor->SetIsEnergyIniEnabled(pEnableEnergyIniCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableEnergyFinCmd) pActor->SetIsEnergyFinEnabled(pEnableEnergyFinCmd->GetNewBoolValue(newValue));
   if(cmd == pEnableTimeCmd) pActor->SetIsTimeEnabled(pEnableTimeCmd->GetNewBoolValue(newValue));
  if(cmd == pEnablePositionXCmd) pActor->SetIsXPositionEnabled(pEnablePositionXCmd->GetNewBoolValue(newValue));
  if(cmd == pEnablePositionYCmd) pActor->SetIsXPositionEnabled(pEnablePositionYCmd->GetNewBoolValue(newValue));
  if(cmd == pEnablePositionZCmd) pActor->SetIsXPositionEnabled(pEnablePositionZCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableLocalPositionXCmd) pActor->SetIsXLocalPositionEnabled(pEnableLocalPositionXCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableLocalPositionYCmd) pActor->SetIsXLocalPositionEnabled(pEnableLocalPositionYCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableLocalPositionZCmd) pActor->SetIsXLocalPositionEnabled(pEnableLocalPositionZCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableSourcePositionXCmd) pActor->SetIsXLocalPositionEnabled(pEnableSourcePositionXCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableSourcePositionYCmd) pActor->SetIsXLocalPositionEnabled(pEnableSourcePositionYCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableSourcePositionZCmd) pActor->SetIsXLocalPositionEnabled(pEnableSourcePositionZCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableVolumeIDCmd) pActor->SetIsVolumeIDEnabled(pEnableVolumeIDCmd->GetNewBoolValue(newValue));
  if(cmd == pEnableSourceEnergyCmd) pActor->SetIsSourceEnergyEnabled(pEnableSourceEnergyCmd->GetNewBoolValue(newValue));
  if(cmd==pEnableSourcePDGCmd) pActor->SetIsSourcePDGEnabled(pEnableSourcePDGCmd->GetNewBoolValue(newValue));
  if(cmd==pEnablenCrystalConvCmd) pActor->SetIsnCrystalConvEnabled(pEnablenCrystalConvCmd->GetNewBoolValue(newValue));
  if(cmd==pEnablenCrystalComptCmd) pActor->SetIsnCrystalComptEnabled(pEnablenCrystalComptCmd->GetNewBoolValue(newValue));
   if(cmd==pEnablenCrystalRaylCmd) pActor->SetIsnCrystalRaylEnabled(pEnablenCrystalComptCmd->GetNewBoolValue(newValue));
  //delete pEnableLayerNameCmd;

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
