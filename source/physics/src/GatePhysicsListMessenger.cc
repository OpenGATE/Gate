/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEPHYSICSLISTMESSENGER_CC
#define GATEPHYSICSLISTMESSENGER_CC

#include "GatePhysicsList.hh"
#include "GatePhysicsListMessenger.hh"
#include "GateMiscFunctions.hh"

//----------------------------------------------------------------------------------------
GatePhysicsListMessenger::GatePhysicsListMessenger(GatePhysicsList * pl)
  :pPhylist(pl)
{
  nInit = 0;
  nEMStdOpt = 0;
  nMuHandler = GateMaterialMuHandler::GetInstance();
}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
GatePhysicsListMessenger::~GatePhysicsListMessenger()
{
  delete pList;
  delete pRemove;
  delete pAdd;
  delete pInit;
  delete pPrint;
  delete pCutInMaterial;
  delete electronCutCmd;
  delete gammaCutCmd;
  delete positronCutCmd;
  delete protonCutCmd;
  delete pMaxStepSizeCmd;
  delete pMaxTrackLengthCmd;
  delete pMaxToFCmd;
  delete pMinKineticEnergyCmd;
  delete pMinRemainingRangeCmd;
  delete pActivateStepLimiterCmd;
  delete pActivateSpecialCutsCmd;
  delete pSetDEDXBinning;
  delete pSetLambdaBinning;
  delete pSetEMin;
  delete pSetEMax;
  delete pSetSplineFlag;

  delete pMuHandlerUsePrecalculatedElements;
  delete pMuHandlerSetEMin;
  delete pMuHandlerSetEMax;
  delete pMuHandlerSetENumber;
  delete pMuHandlerSetAtomicShellEMin;
  delete pMuHandlerSetPrecision;

  delete pAddAtomDeexcitation;
  delete pAddPhysicsList;
  delete pAddPhysicsListMixed;
  delete pAddProcessMixed;

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
void GatePhysicsListMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/addProcess";
  pAdd = new GateUIcmdWith2String(bb,this);
  G4String guidance = "Enable processes";
  pAdd->SetGuidance(guidance);
  pAdd->SetParameterName("Process","Particle or Group of particles",false,true);
  pAdd->SetDefaultValue("","Default");



  //Add Mixed processes (DNA)
  bb = base+"/addProcessMixed";
  pAddProcessMixed = new GateUIcmdWith2String(bb,this);
  guidance = "Enable mixed processes";
  pAddProcessMixed->SetGuidance(guidance);
  pAddProcessMixed->SetParameterName("Process","mixed processes",false,true);
  pAddProcessMixed->SetDefaultValue("","Default");

  bb = base+"/removeProcess";
  pRemove = new GateUIcmdWith2String(bb,this);
  guidance = "Disable processes";
  pRemove->SetGuidance(guidance);
  pRemove->SetParameterName("Process","Particle or Group of particles",false,true);
  pRemove->SetDefaultValue("","Default");

  bb = base+"/processList";
  pList = new GateUIcmdWith2String(bb,this);
  guidance = "List of processes";
  pList->SetGuidance(guidance);
  pList->SetParameterName("State","Particle",true, true);
  pList->SetDefaultValue("Available","All");
  pList->SetCandidates("Available Enabled Initialized","");

  bb = base+"/print";
  pPrint = new G4UIcmdWithAString(bb,this);
  guidance = "Print the physics list";
  pPrint->SetGuidance(guidance);
  pPrint->SetParameterName("File name",false);
  pPrint->SetCandidates("");

  bb = base+"/init";
  pInit = new G4UIcmdWithoutParameter(bb,this);
  guidance = "Physics list Initialisation";
  pInit->SetGuidance(guidance);

  // Cuts messengers
  bb = base+"/displayCuts";
  pCutInMaterial = new G4UIcmdWithoutParameter(bb,this);
  guidance = "Print cuts in volumes";
  pCutInMaterial->SetGuidance(guidance);

  bb = base+"/Gamma/SetCutInRegion";
  gammaCutCmd = new G4UIcmdWithAString(bb,this);
  gammaCutCmd->SetGuidance("Set gamma production cut for a given region (two parameters 'regionName' and 'cutValue')");

  bb = base+"/Electron/SetCutInRegion";
  electronCutCmd = new G4UIcmdWithAString(bb,this);
  electronCutCmd->SetGuidance("Set electron production cut for a given region (two parameters 'regionName' and 'cutValue')");

  bb = base+"/Positron/SetCutInRegion";
  positronCutCmd = new G4UIcmdWithAString(bb,this);
  positronCutCmd->SetGuidance("Set positron production cut for a given region (two parameters 'regionName' and 'cutValue')");

  bb = base+"/Proton/SetCutInRegion";
  protonCutCmd = new G4UIcmdWithAString(bb,this);
  protonCutCmd->SetGuidance("Set proton production cut for a given region (two parameters 'regionName' and 'cutValue')");

  bb = base+"/SetMaxStepSizeInRegion";
  pMaxStepSizeCmd = new G4UIcmdWithAString(bb,this);
  pMaxStepSizeCmd->SetGuidance("Set the maximum step size for a given region (two parameters 'regionName' and 'cutValue') YOU ALSO NEED TO SET ActivateStepLimiter");

  bb = base+"/SetMaxToFInRegion";
  pMaxToFCmd = new G4UIcmdWithAString(bb,this);
  pMaxToFCmd->SetGuidance("Set the maximum time of flight for a given region (two parameters 'regionName' and 'cutValue') YOU ALSO NEED TO SET ActivateSpecialCuts");

  bb = base+"/SetMinKineticEnergyInRegion";
  pMinKineticEnergyCmd = new G4UIcmdWithAString(bb,this);
  pMinKineticEnergyCmd->SetGuidance("Set the minimum energy of the track for a given region (two parameters 'regionName' and 'cutValue') YOU ALSO NEED TO SET ActivateSpecialCuts");

  bb = base+"/SetMaxTrackLengthInRegion";
  pMaxTrackLengthCmd = new G4UIcmdWithAString(bb,this);
  pMaxTrackLengthCmd->SetGuidance("Set the maximum length of the track for a given region (two parameters 'regionName' and 'cutValue') YOU ALSO NEED TO SET ActivateSpecialCuts");

  bb = base+"/SetMinRemainingRangeInRegion";
  pMinRemainingRangeCmd = new G4UIcmdWithAString(bb,this);
  pMinRemainingRangeCmd->SetGuidance("Set the minimum remaining range of the track for a given region (two parameters 'regionName' and 'cutValue') YOU ALSO NEED TO SET ActivateSpecialCuts");

  bb = base+"/ActivateStepLimiter";
  pActivateStepLimiterCmd = new G4UIcmdWithAString(bb,this);
  pActivateStepLimiterCmd->SetGuidance("Activate step limiter for a given particle");

  bb = base+"/ActivateSpecialCuts";
  pActivateSpecialCutsCmd = new G4UIcmdWithAString(bb,this);
  pActivateSpecialCutsCmd->SetGuidance("Activate special cuts for a given particle");

  // options messengers	G4ReferenceManual5.2
  bb = base+"/setDEDXBinning";
  pSetDEDXBinning = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Set DEDXBinning for Standard EM Processes";
  pSetDEDXBinning->SetGuidance(guidance);

  bb = base+"/setLambdaBinning";
  pSetLambdaBinning = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Set LambdaBinning for Standard EM Processes";
  pSetLambdaBinning->SetGuidance(guidance);

  bb = base+"/setEMin";
  pSetEMin = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set EMin for Standard EM Processes";
  pSetEMin->SetGuidance(guidance);

  bb = base+"/setEMax";
  pSetEMax = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set EMax for Standard EM Processes";
  pSetEMax->SetGuidance(guidance);

  bb = base+"/setSplineFlag";
  pSetSplineFlag = new G4UIcmdWithABool(bb,this);
  guidance = "Set SplineFlag for Standard EM Processes";
  pSetSplineFlag->SetGuidance(guidance);

  // Mu Handler commands
  bb = base+"/MuHandler/setElementFolderName";
  pMuHandlerUsePrecalculatedElements = new G4UIcmdWithAString(bb,this);
  guidance = "Point the folder where the Mu and Muen files per elements are stored";
  pMuHandlerUsePrecalculatedElements->SetGuidance(guidance);

  bb = base+"/MuHandler/setEMin";
  pMuHandlerSetEMin = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set minimal energy for attenuation and energy-absorption coefficients simulation";
  pMuHandlerSetEMin->SetGuidance(guidance);

  bb = base+"/MuHandler/setEMax";
  pMuHandlerSetEMax = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set maximum energy for attenuation and energy-absorption coefficients simulation";
  pMuHandlerSetEMax->SetGuidance(guidance);

  bb = base+"/MuHandler/setENumber";
  pMuHandlerSetENumber = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Set number of energies for attenuation and energy-absorption coefficients simulation";
  pMuHandlerSetENumber->SetGuidance(guidance);

  bb = base+"/MuHandler/setAtomicShellEMin";
  pMuHandlerSetAtomicShellEMin = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set atomic shell minimal energy";
  pMuHandlerSetAtomicShellEMin->SetGuidance(guidance);

  bb = base+"/MuHandler/setPrecision";
  pMuHandlerSetPrecision = new G4UIcmdWithADouble(bb,this);
  guidance = "Set precision to be reached in %";
  pMuHandlerSetPrecision->SetGuidance(guidance);

  bb = base+"/addAtomDeexcitation";
  pAddAtomDeexcitation = new G4UIcommand(bb,this);
  guidance = "Add atom deexcitation into the energy loss table manager";
  pAddAtomDeexcitation->SetGuidance(guidance);

  // Command to call G4 Physics List builders
  bb = base+"/addPhysicsList";
  pAddPhysicsList = new G4UIcmdWithAString(bb,this);
  guidance = "Select a Geant4 Physic List builder";
  pAddPhysicsList->SetGuidance(guidance);
  pAddPhysicsList->SetParameterName("Builder name",false);

   //Command to call Mixed G4 Physics List
  bb = base+"/addPhysicsListMixed";
  pAddPhysicsListMixed = new G4UIcmdWithAString(bb,this);
  guidance = "Select a Geant4 Physic List builder to mix: emstandard_opt3_mixed_emdna, emlivermore_mixed_emdna";
  pAddPhysicsListMixed->SetGuidance(guidance);
  pAddPhysicsListMixed->SetParameterName("Mixed name",false);

  bb = base+"/ConstructProcessMixed";
  pConstructProcessMixed = new G4UIcmdWithABool(bb,this);
  guidance = "Construct mixed processes for DNA";
  pConstructProcessMixed->SetGuidance(guidance);
  pConstructProcessMixed->SetParameterName("Mixed name",true);
  pConstructProcessMixed->SetDefaultValue(true);

  // To set the low edge energy
  bb = base+"/SetEnergyRangeMinLimit";
  pEnergyRangeMinLimitCmd = new G4UIcmdWithADoubleAndUnit(bb,this);
  double low = G4ProductionCutsTable::GetProductionCutsTable()->GetLowEdgeEnergy();
  double high= G4ProductionCutsTable::GetProductionCutsTable()->GetHighEdgeEnergy();
  G4String guid = "Set the minimum limit of the Energy range. Default are [";
  guid += G4BestUnit(low,"Energy");
  guid += " ";
  guid += G4BestUnit(high, "Energy");
  guid += "]";
  pEnergyRangeMinLimitCmd->SetGuidance(guid);

}
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
void GatePhysicsListMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  // Cut for regions
  if (command == gammaCutCmd || command == electronCutCmd || command == positronCutCmd || command == protonCutCmd ||
      command == pMaxStepSizeCmd || command == pMaxToFCmd || command == pMinKineticEnergyCmd || command == pMaxTrackLengthCmd || command == pMinRemainingRangeCmd)	{
    G4String regionName;
    double cutValue;
    GetStringAndValueFromCommand(command, param, regionName, cutValue);
    if (command == gammaCutCmd) { pPhylist->SetCutInRegion("gamma", regionName, cutValue); }
    if (command == electronCutCmd) { pPhylist->SetCutInRegion("e-", regionName, cutValue); }
    if (command == positronCutCmd) { pPhylist->SetCutInRegion("e+", regionName, cutValue); }
    if (command == protonCutCmd) { pPhylist->SetCutInRegion("proton", regionName, cutValue);}

    if (command == pMaxStepSizeCmd) pPhylist->SetSpecialCutInRegion("MaxStepSize", regionName, cutValue);
    if (command == pMaxToFCmd) pPhylist->SetSpecialCutInRegion("MaxToF", regionName, cutValue);
    if (command == pMinKineticEnergyCmd) pPhylist->SetSpecialCutInRegion("MinKineticEnergy", regionName, cutValue);
    if (command == pMaxTrackLengthCmd) pPhylist->SetSpecialCutInRegion("MaxTrackLength", regionName, cutValue);
    if (command == pMinRemainingRangeCmd) pPhylist->SetSpecialCutInRegion("MinRemainingRange", regionName, cutValue);  }

  if (command == pActivateStepLimiterCmd) pPhylist->mListOfStepLimiter.push_back(param);
  if (command == pActivateSpecialCutsCmd) pPhylist->mListOfG4UserSpecialCut.push_back(param);

  // processes
  if (command == pInit)
    {
      if (nInit!=0)
        {
          GateWarning("Physic List already initialized\n");
          return;
        }
      nInit++;
      pPhylist->ConstructProcess();
      return;
    }

  if (command == pPrint)
    {
      char par1[30];
      std::istringstream is(param);
      is >> par1;
      pPhylist->Write(par1);
    }

  if (command == pCutInMaterial)
    {
      pPhylist->GetCuts();
    }

  char par1[30];
  char par2[30];
  std::istringstream is(param);
  is >> par1 >> par2;

  if (command == pList)
    {
      pPhylist->Print(par1,par2);
    }
  else
    {
      if (command == pRemove)
        {
          if (nInit!=0)
            {
              GateWarning("Physic List already initialized: you can't remove process\n");
              return;
            }
          pPhylist->RemoveProcesses(par1,par2);
        }
      if (command == pAdd)
        {
          if (nInit!=0)
            {
              GateWarning("Physic List already initialized: you can't add process\n");
              return;
            }
          pPhylist->AddProcesses(par1,par2);
        }
         // for mixed processes (DNA)
         if (command == pAddProcessMixed)
        {
          if (nInit!=0)
            {
              GateWarning("Physic List already initialized: you can't add process\n");
              return;
            }
          pPhylist->ConstructProcessMixed();
        }
    }

  // options for EM standard
  if (command == pSetDEDXBinning) {
    int nbBins = pSetDEDXBinning->GetNewIntValue(param);
    pPhylist->SetOptDEDXBinning(nbBins);
    GateMessage("Physic", 1, "(EM Options) DEDXBinning set to "<<nbBins<<" bins. DEDXBinning defaut Value 84."<<G4endl);
  }
  if (command == pSetLambdaBinning) {
    int nbBins = pSetLambdaBinning->GetNewIntValue(param);
    pPhylist->SetOptLambdaBinning(nbBins);
    GateMessage("Physic", 1, "(EM Options) LambdaBinning set to "<<nbBins<<" bins. LambdaBinning defaut Value 84."<<G4endl);
  }
  if (command == pSetEMin) {
    double val = pSetEMin->GetNewDoubleValue(param);
    pPhylist->SetOptEMin(val);
    GateMessage("Physic", 1, "(EM Options) Min Energy set to "<<G4BestUnit(val,"Energy")<<". MinEnergy defaut Value 0.1keV."<<G4endl);
  }
  if (command == pSetEMax) {
    double val = pSetEMax->GetNewDoubleValue(param);
    pPhylist->SetOptEMax(val);
    GateMessage("Physic", 1, "(EM Options) Max Energy set to "<<G4BestUnit(val,"Energy")<<". MaxEnergy defaut Value 100TeV."<<G4endl);
  }
  if (command == pSetSplineFlag) {
    G4bool flag = pSetSplineFlag->GetNewBoolValue(param);
    pPhylist->SetOptSplineFlag(flag);
    GateMessage("Physic", 1, "(EM Options) Spline Falg set to "<<flag<<". Spline Flag defaut 1."<<G4endl);
  }

  // Mu Handler commands
  if (command == pMuHandlerUsePrecalculatedElements){
    nMuHandler->SetElementsFolderName(param);
  }
  if(command == pMuHandlerSetEMin){
    double val = pMuHandlerSetEMin->GetNewDoubleValue(param);
    nMuHandler->SetEMin(val);
    GateMessage("Physic", 1, "(MuHandler Options) Min Energy set to "<<G4BestUnit(val,"Energy")<<". MinEnergy defaut Value: 250 eV."<<G4endl);
  }
  if(command == pMuHandlerSetEMax){
    double val = pMuHandlerSetEMax->GetNewDoubleValue(param);
    nMuHandler->SetEMax(val);
    GateMessage("Physic", 1, "(MuHandler Options) Max Energy set to "<<G4BestUnit(val,"Energy")<<". MaxEnergy defaut Value: 1 MeV."<<G4endl);
  }
  if(command == pMuHandlerSetENumber){
    int nbVal = pMuHandlerSetENumber->GetNewIntValue(param);
    nMuHandler->SetENumber(nbVal);
    GateMessage("Physic", 1, "(MuHandler Options) ENumber set to "<<nbVal<<" values. ENumber defaut Value: 25 between 250 eV and 1 MeV (logscale)."<<G4endl);
  }
  if(command == pMuHandlerSetAtomicShellEMin){
    double val = pMuHandlerSetAtomicShellEMin->GetNewDoubleValue(param);
    nMuHandler->SetAtomicShellEMin(val);
    GateMessage("Physic", 1, "(MuHandler Options) Min Energy for atomic shell set to "<<G4BestUnit(val,"Energy")<<". MinEnergy defaut Value: 250 eV."<<G4endl);
  }
  if(command == pMuHandlerSetPrecision){
    double val = pMuHandlerSetPrecision->GetNewDoubleValue(param);
    nMuHandler->SetPrecision(val);
    GateMessage("Physic", 1, "(MuHandler Options) Precision set to "<<val<<". Precision defaut Value: 0.01"<<G4endl);
  }

  if (command == pAddAtomDeexcitation) {
    pPhylist->AddAtomDeexcitation();
    GateMessage("Physic", 1, "Atom Deexcitation process has been added into the energy loss table manager"<<G4endl);
  }

  // Command to call G4 Physics List builders
  if (command == pAddPhysicsList) {
    pPhylist->ConstructPhysicsList(param);
  }

  //Command to call mixed physics list (DNA)
  if (command == pAddPhysicsListMixed){
    pPhylist->ConstructPhysicsListDNAMixed(param);
  }
  if (command == pConstructProcessMixed){
    pPhylist->ConstructProcessMixed();
  }

  if (command == pEnergyRangeMinLimitCmd) {
    double val = pEnergyRangeMinLimitCmd->GetNewDoubleValue(param);
    pPhylist->SetEnergyRangeMinLimit(val);
    GateMessage("Physic", 1, "Min Energy range set to "<<G4BestUnit(val,"Energy") << G4endl);
  }

}
//----------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------------
G4double GatePhysicsListMessenger::ScaleValue(G4double value,G4String unit)
{
  double res = 0.;
  if (unit=="eV")  res = value *  eV;
  if (unit=="keV") res = value * keV;
  if (unit=="MeV") res = value * MeV;
  if (unit=="GeV") res = value * GeV;

  return res;
}
//----------------------------------------------------------------------------------------

#endif /* end #define GATEPHYSICSLISTMESSENGER_CC */
