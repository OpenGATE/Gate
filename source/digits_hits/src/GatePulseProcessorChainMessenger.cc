/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
//GND 2022 Class to Remove
#include "GateConfiguration.h"
#include "GatePulseProcessorChainMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"

#include "GateVPulseProcessor.hh"
#include "GatePulseProcessorChain.hh"
/*
#include "GateReadout.hh"
#include "GatePileup.hh"
#include "GateDeadTime.hh"
#include "GateBlurring.hh"
#include "GateLocalTimeDelay.hh"
#include "GateLocalBlurring.hh"
#include "GateLocalEfficiency.hh"
#include "GateEnergyEfficiency.hh"
#include "GateNoise.hh"
#include "GateBuffer.hh"
#include "GateDiscretizer.hh"
#include "GateBlurringWithIntrinsicResolution.hh"
#include "GateLightYield.hh"
#include "GateTransferEfficiency.hh"
#include "GateCrosstalk.hh"
#include "GateQuantumEfficiency.hh"
#include "GateSigmoidalThresholder.hh"
#include "GateCalibration.hh"
#include "GateSpblurring.hh"
//#include "GatePulseAdder.hh"
//#include "GatePulseAdderLocal.hh"
#include "GatePulseAdderCompton.hh"
#include "GatePulseAdderComptPhotIdeal.hh"
#include "GatePulseAdderComptPhotIdealLocal.hh"
//#include "GateCrystalBlurring.hh"
#include "GateTemporalResolution.hh"
#include "GateLocalClustering.hh"
#include "GateClustering.hh"
#include "GateEnergyThresholder.hh"
#include "GateLocalEnergyThresholder.hh"
#include "GateCC3DlocalSpblurring.hh"
#include "GateDoIModels.hh"
#include "GateGridDiscretization.hh"
#include "GateLocalMultipleRejection.hh"
#include "GateLocalTimeResolution.hh"

#ifdef GATE_USE_OPTICAL
#include "GateOpticalAdder.hh"
#endif
#include "GateSystemFilter.hh"
*/
GatePulseProcessorChainMessenger::GatePulseProcessorChainMessenger(GatePulseProcessorChain* itsProcessorChain)
:GateListMessenger(itsProcessorChain)
{ 
  pInsertCmd->SetCandidates(DumpMap());

  G4String cmdName;

  cmdName = GetDirectoryName()+"setInputName";
  SetInputNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetInputNameCmd->SetGuidance("Set the name of the input pulse channel");
  SetInputNameCmd->SetParameterName("Name",false);
}




GatePulseProcessorChainMessenger::~GatePulseProcessorChainMessenger()
{
  delete SetInputNameCmd;
}




void GatePulseProcessorChainMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if (command == SetInputNameCmd) 
    { GetProcessorChain()->SetInputName(newValue); }
  else
    GateListMessenger::SetNewValue(command,newValue);
}




const G4String& GatePulseProcessorChainMessenger::DumpMap()
{
   static G4String theList = "readout pileup thresholder energyThresholder localEnergyThresholder DoImodel upholder blurring localBlurring localTimeDelay localEfficiency energyEfficiency noise discretizer buffer transferEfficiency crosstalk lightYield quantumEfficiency intrinsicResolutionBlurring sigmoidalThresholder calibration spblurring sp3Dlocalblurring adder adderLocal adderCompton adderComptPhotIdeal adderComptPhotIdealLocal localClustering  clustering deadtime crystalblurring timeResolution localTimeResolution opticaladder systemFilter gridDiscretization  localMultipleRejection";
  return theList;
}



void GatePulseProcessorChainMessenger::DoInsertion(const G4String& childTypeName)
{
  
  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(childTypeName);
    
  AvoidNameConflicts();

  GateVPulseProcessor* newProcessor=0;

  G4String newInsertionName = GetProcessorChain()->MakeElementName(GetNewInsertionBaseName());

 /* if (childTypeName=="readout")
    newProcessor = new GateReadout(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="pileup")
    newProcessor = new GatePileup(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="discretizer")
    newProcessor = new GateDiscretizer(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="thresholder")
    newProcessor = new GateThresholder(GetProcessorChain(),newInsertionName,50.*keV);
  else if (childTypeName=="energyThresholder")
    newProcessor = new GateEnergyThresholder(GetProcessorChain(),newInsertionName,50.*keV);
  else if (childTypeName=="localEnergyThresholder")
    newProcessor = new GateLocalEnergyThresholder(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="DoImodel")
    //newProcessor = new GateDoIModels(GetProcessorChain(),newInsertionName,G4ThreeVector(0.,0.,1.));
    newProcessor = new GateDoIModels(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="upholder")
    newProcessor = new GateUpholder(GetProcessorChain(),newInsertionName,150.*keV);
  else if (childTypeName=="deadtime")
    newProcessor = new GateDeadTime(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="blurring")
    newProcessor = new GateBlurring(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="localBlurring")
    newProcessor = new GateLocalBlurring(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="localTimeDelay")
    newProcessor = new GateLocalTimeDelay(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="transferEfficiency")
    newProcessor = GateTransferEfficiency::GetInstance(GetProcessorChain(),newInsertionName); 
  else if (childTypeName=="lightYield")
    newProcessor = GateLightYield::GetInstance(GetProcessorChain(),newInsertionName); 
  else if (childTypeName=="crosstalk")
    newProcessor = GateCrosstalk::GetInstance(GetProcessorChain(),newInsertionName,0.,0.);
  else if (childTypeName=="quantumEfficiency")
    newProcessor = GateQuantumEfficiency::GetInstance(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="intrinsicResolutionBlurring")
    newProcessor = new GateBlurringWithIntrinsicResolution(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="sigmoidalThresholder")
    newProcessor = new GateSigmoidalThresholder(GetProcessorChain(),newInsertionName,0.,1.,0.5);
  else if (childTypeName=="calibration")
    newProcessor = new GateCalibration(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="spblurring")
    newProcessor = new GateSpblurring(GetProcessorChain(),newInsertionName,0.1);
  else if (childTypeName=="sp3Dlocalblurring")
    newProcessor = new GateCC3DlocalSpblurring(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="adder")
    newProcessor = new GatePulseAdder(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="adderLocal")
    newProcessor = new GatePulseAdderLocal(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="adderCompton")
    newProcessor = new GatePulseAdderCompton(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="adderComptPhotIdeal")
    newProcessor = new GatePulseAdderComptPhotIdeal(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="adderComptPhotIdealLocal")
    newProcessor = new GatePulseAdderComptPhotIdealLocal(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="localClustering")
    newProcessor = new GateLocalClustering(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="clustering")
    newProcessor = new GateClustering(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="crystalblurring")
    newProcessor = new GateCrystalBlurring(GetProcessorChain(),newInsertionName,-1.,-1.,1.,-1.*keV);
  else if (childTypeName=="localEfficiency")
    newProcessor = new GateLocalEfficiency(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="energyEfficiency")
    newProcessor = new GateEnergyEfficiency(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="noise")
    newProcessor = new GateNoise(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="buffer")
    newProcessor = new GateBuffer(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="timeResolution")
    newProcessor = new GateTemporalResolution(GetProcessorChain(),newInsertionName,0. * ns);
  else if (childTypeName=="localTimeResolution")
    newProcessor = new GateLocalTimeResolution(GetProcessorChain(),newInsertionName);
  else if (childTypeName=="systemFilter")
     newProcessor = new GateSystemFilter(GetProcessorChain(),newInsertionName);
 // else if (childTypeName=="stripSpDiscretization")
  //   newProcessor = new GateStripSpatialDiscretization(GetProcessorChain(),newInsertionName);
else if (childTypeName=="gridDiscretization")
     newProcessor = new GateGridDiscretization(GetProcessorChain(),newInsertionName);
else if (childTypeName=="localMultipleRejection")
     newProcessor = new GateLocalMultipleRejection(GetProcessorChain(),newInsertionName);
#ifdef GATE_USE_OPTICAL
  else if (childTypeName=="opticaladder")
    newProcessor = new GateOpticalAdder(GetProcessorChain(), newInsertionName);
#endif
  else {
    G4cout << "Pulse-processor type name '" << childTypeName << "' was not recognised --> insertion request must be ignored!\n";
    return;
  }
  */
  GetProcessorChain()->InsertProcessor(newProcessor);
  SetNewInsertionBaseName("");
}


G4bool GatePulseProcessorChainMessenger::CheckNameConflict(const G4String& name)
{
  // Check whether an object with the same name already exists in the list
  return ( GetListManager()->FindElement( GetListManager()->GetObjectName() + "/" + name ) != 0 ) ;
}





