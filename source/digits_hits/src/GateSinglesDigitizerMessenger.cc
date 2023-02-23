/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateSinglesDigitizer.hh"
#include "GateSinglesDigitizerMessenger.hh"

#include "GateConfiguration.h"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"

#include "GateVDigitizerModule.hh"
#include "GateDigitizerMgr.hh"
#include "GateAdder.hh"
#include "GateReadout.hh"
#include "GateEnergyFraming.hh"
#include "GateTimeResolution.hh"
#include "GateEnergyResolution.hh"
/*UNCOMM#include "GateSpatialResolution.hh"
#include "GateEfficiency.hh"
#include "GateDeadTime.hh"
#include "GatePileup.hh"
#include "GateAdderCompton.hh"
*/
/*

#include "GatePileup.hh"
#include "GateThresholder.hh"
#include "GateUpholder.hh"

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
#include "GatePulseAdder.hh"
#include "GatePulseAdderLocal.hh"
#include "GatePulseAdderCompton.hh"
#include "GatePulseAdderComptPhotIdeal.hh"
#include "GatePulseAdderComptPhotIdealLocal.hh"
#include "GateCrystalBlurring.hh"
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
*/
#ifdef GATE_USE_OPTICAL
#include "GateOpticalAdder.hh"
#endif
#include "GateSystemFilter.hh"

GateSinglesDigitizerMessenger::GateSinglesDigitizerMessenger(GateSinglesDigitizer* itsDigitizer)
:GateListMessenger(itsDigitizer),m_digitizer(itsDigitizer)
{
  pInsertCmd->SetCandidates(DumpMap());

  G4String cmdName;

  cmdName = GetDirectoryName()+"setInputCollection";
  SetInputNameCmd = new G4UIcmdWithAString(cmdName,this);
  SetInputNameCmd->SetGuidance("Set the name of the input pulse channel");
  SetInputNameCmd->SetParameterName("Name",false);




}




GateSinglesDigitizerMessenger::~GateSinglesDigitizerMessenger()
{
  delete SetInputNameCmd;

}




void GateSinglesDigitizerMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{

  if (command == SetInputNameCmd)
    { m_digitizer->SetInputName(newValue); }
  else
    GateListMessenger::SetNewValue(command,newValue);
}




const G4String& GateSinglesDigitizerMessenger::DumpMap()
{
   static G4String theList = "readout adder energyFraming timeResolution energyResolution spatialResolution efficiency deadtime pileup adderCompton";

   return theList;
}



void GateSinglesDigitizerMessenger::DoInsertion(const G4String& childTypeName)
{

  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(childTypeName);

  AvoidNameConflicts();

  GateVDigitizerModule* newDM=0;

  G4String newInsertionName = m_digitizer->MakeElementName(GetNewInsertionBaseName());
  G4String DMname = m_digitizer->GetDMNameFromInsertionName(newInsertionName);


  if (childTypeName=="adder")
    {
  	  newDM = new GateAdder(m_digitizer, DMname);
  	  m_digitizer->AddNewModule(newDM);
    }
  else if (childTypeName=="readout")
  {
	  newDM = new GateReadout(m_digitizer, DMname);
	  m_digitizer->AddNewModule(newDM);
  }
  else if (childTypeName=="energyFraming")
  {
	  newDM = new GateEnergyFraming(m_digitizer, DMname);
	  m_digitizer->AddNewModule(newDM);
  }
 else if (childTypeName=="timeResolution")
    {
  	  newDM = new GateTimeResolution(m_digitizer, DMname);
  	  m_digitizer->AddNewModule(newDM);
    }
   else if (childTypeName=="energyResolution")
    {
  	  newDM = new GateEnergyResolution(m_digitizer, DMname);
  	  m_digitizer->AddNewModule(newDM);
    }
  /*UNCOMM  else if (childTypeName=="spatialResolution")
    {
  	  newDM = new GateSpatialResolution(m_digitizer);
  	  m_digitizer->AddNewModule(newDM);
    }
  else if (childTypeName=="efficiency")
     {
   	  newDM = new GateEfficiency(m_digitizer);
   	  m_digitizer->AddNewModule(newDM);
     }
  else if (childTypeName=="deadtime")
     {
   	  newDM = new GateDeadTime(m_digitizer);
   	  m_digitizer->AddNewModule(newDM);
     }
  else if (childTypeName=="pileup")
       {
     	  newDM = new GatePileup(m_digitizer);
     	  m_digitizer->AddNewModule(newDM);
       }
  else if (childTypeName=="adderCompton")
       {
     	  newDM = new GateAdderCompton(m_digitizer);
     	  m_digitizer->AddNewModule(newDM);
       }
   // newDM = new GateReadout(m_digitizer,newInsertionName);
/*  else if (childTypeName=="pileup")
    newDM = new GatePileup(m_digitizer,newInsertionName);
  else if (childTypeName=="discretizer")
    newDM = new GateDiscretizer(m_digitizer,newInsertionName);
  else if (childTypeName=="thresholder")
    newDM = new GateThresholder(m_digitizer,newInsertionName,50.*keV);
  else if (childTypeName=="energyThresholder")
    newDM = new GateEnergyThresholder(m_digitizer,newInsertionName,50.*keV);
  else if (childTypeName=="localEnergyThresholder")
    newDM = new GateLocalEnergyThresholder(m_digitizer,newInsertionName);
  else if (childTypeName=="DoImodel")
    //newDM = new GateDoIModels(m_digitizer,newInsertionName,G4ThreeVector(0.,0.,1.));
    newDM = new GateDoIModels(m_digitizer,newInsertionName);
  else if (childTypeName=="upholder")
    newDM = new GateUpholder(m_digitizer,newInsertionName,150.*keV);
  else if (childTypeName=="deadtime")
    newDM = new GateDeadTime(m_digitizer,newInsertionName);
  else if (childTypeName=="blurring")
    newDM = new GateBlurring(m_digitizer,newInsertionName);
  else if (childTypeName=="localBlurring")
    newDM = new GateLocalBlurring(m_digitizer,newInsertionName);
  else if (childTypeName=="localTimeDelay")
    newDM = new GateLocalTimeDelay(m_digitizer,newInsertionName);
  else if (childTypeName=="transferEfficiency")
    newDM = GateTransferEfficiency::GetInstance(m_digitizer,newInsertionName);
  else if (childTypeName=="lightYield")
    newDM = GateLightYield::GetInstance(m_digitizer,newInsertionName);
  else if (childTypeName=="crosstalk")
    newDM = GateCrosstalk::GetInstance(m_digitizer,newInsertionName,0.,0.);
  else if (childTypeName=="quantumEfficiency")
    newDM = GateQuantumEfficiency::GetInstance(m_digitizer,newInsertionName);
  else if (childTypeName=="intrinsicResolutionBlurring")
    newDM = new GateBlurringWithIntrinsicResolution(m_digitizer,newInsertionName);
  else if (childTypeName=="sigmoidalThresholder")
    newDM = new GateSigmoidalThresholder(m_digitizer,newInsertionName,0.,1.,0.5);
  else if (childTypeName=="calibration")
    newDM = new GateCalibration(m_digitizer,newInsertionName);
  else if (childTypeName=="spblurring")
    newDM = new GateSpblurring(m_digitizer,newInsertionName,0.1);
  else if (childTypeName=="sp3Dlocalblurring")
    newDM = new GateCC3DlocalSpblurring(m_digitizer,newInsertionName);
  else if (childTypeName=="adder")
  {
	  newDM = new GateAdder(m_digitizer);
	 // m_digitizer->AddNewModule(newDM);
	 // G4DigiManager::GetDMpointer()->AddNewModule(newDM);
  }
    else if (childTypeName=="adderLocal")
    newDM = new GatePulseAdderLocal(m_digitizer,newInsertionName);
  else if (childTypeName=="adderCompton")
    newDM = new GatePulseAdderCompton(m_digitizer,newInsertionName);
  else if (childTypeName=="adderComptPhotIdeal")
    newDM = new GatePulseAdderComptPhotIdeal(m_digitizer,newInsertionName);
  else if (childTypeName=="adderComptPhotIdealLocal")
    newDM = new GatePulseAdderComptPhotIdealLocal(m_digitizer,newInsertionName);
  else if (childTypeName=="localClustering")
    newDM = new GateLocalClustering(m_digitizer,newInsertionName);
  else if (childTypeName=="clustering")
    newDM = new GateClustering(m_digitizer,newInsertionName);
  else if (childTypeName=="crystalblurring")
    newDM = new GateCrystalBlurring(m_digitizer,newInsertionName,-1.,-1.,1.,-1.*keV);
  else if (childTypeName=="localEfficiency")
    newDM = new GateLocalEfficiency(m_digitizer,newInsertionName);
  else if (childTypeName=="energyEfficiency")
    newDM = new GateEnergyEfficiency(m_digitizer,newInsertionName);
  else if (childTypeName=="noise")
    newDM = new GateNoise(m_digitizer,newInsertionName);
  else if (childTypeName=="buffer")
    newDM = new GateBuffer(m_digitizer,newInsertionName);
  else if (childTypeName=="timeResolution")
    newDM = new GateTemporalResolution(m_digitizer,newInsertionName,0. * ns);
  else if (childTypeName=="localTimeResolution")
    newDM = new GateLocalTimeResolution(m_digitizer,newInsertionName);
  else if (childTypeName=="systemFilter")
     newDM = new GateSystemFilter(m_digitizer,newInsertionName);
 // else if (childTypeName=="stripSpDiscretization")
  //   newDM = new GateStripSpatialDiscretization(m_digitizer,newInsertionName);
else if (childTypeName=="gridDiscretization")
     newDM = new GateGridDiscretization(m_digitizer,newInsertionName);
else if (childTypeName=="localMultipleRejection")
     newDM = new GateLocalMultipleRejection(m_digitizer,newInsertionName);
#ifdef GATE_USE_OPTICAL
  else if (childTypeName=="opticaladder")
    newDM = new GateOpticalAdder(m_digitizer, newInsertionName);
#endif
*/
 else {
    G4cout << "Pulse-processor type name '" << childTypeName << "' was not recognised --> insertion request must be ignored!\n";
    return;
  }


  SetNewInsertionBaseName("");
}


G4bool GateSinglesDigitizerMessenger::CheckNameConflict(const G4String& name)
{
  // Check whether an object with the same name already exists in the list
  return ( GetListManager()->FindElement( GetListManager()->GetObjectName() + "/" + name ) != 0 ) ;
}



