/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateSinglesDigitizerMessenger

  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
  */

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
#include "GateSpatialResolution.hh"
#include "GateEfficiency.hh"
#include "GateDeadTime.hh"
#include "GatePileup.hh"
#include "GateAdderCompton.hh"
#include "GateOpticalAdder.hh"
#include "GateNoise.hh"
#include "GateDigitizerMerger.hh"

#include "GateDoIModels.hh"
/*
#include "GateLocalTimeDelay.hh"
#include "GateBuffer.hh"
#include "GateBlurringWithIntrinsicResolution.hh"
#include "GateLightYield.hh"
#include "GateTransferEfficiency.hh"
#include "GateCrosstalk.hh"
#include "GateQuantumEfficiency.hh"
#include "GateCalibration.hh"
#include "GatePulseAdderComptPhotIdeal.hh"
#include "GatePulseAdderComptPhotIdealLocal.hh"
#include "GateLocalClustering.hh"
#include "GateClustering.hh"
#include "GateEnergyThresholder.hh"
#include "GateLocalEnergyThresholder.hh"
#include "GateDoIModels.hh"
#include "GateGridDiscretization.hh"
#include "GateLocalMultipleRejection.hh"

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
  SetInputNameCmd->SetGuidance("Set the name of the input collection name");
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
   static G4String theList = "readout adder energyFraming timeResolution energyResolution spatialResolution efficiency deadtime pileup adderCompton opticaladder noise merger doIModels";

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
   else if (childTypeName=="spatialResolution")
    {
  	  newDM = new GateSpatialResolution(m_digitizer, DMname);
  	  m_digitizer->AddNewModule(newDM);
    }
  else if (childTypeName=="efficiency")
     {
   	  newDM = new GateEfficiency(m_digitizer, DMname);
   	  m_digitizer->AddNewModule(newDM);
     }
   else if (childTypeName=="deadtime")
     {
   	  newDM = new GateDeadTime(m_digitizer, DMname);
   	  m_digitizer->AddNewModule(newDM);
     }
  else if (childTypeName=="pileup")
       {
     	  newDM = new GatePileup(m_digitizer, DMname);
     	  m_digitizer->AddNewModule(newDM);
       }
  else if (childTypeName=="adderCompton")
       {
     	  newDM = new GateAdderCompton(m_digitizer, DMname);
     	  m_digitizer->AddNewModule(newDM);
       }
#ifdef GATE_USE_OPTICAL
  else if (childTypeName=="opticaladder")
       {
     	  newDM = new GateOpticalAdder(m_digitizer, DMname);
     	  m_digitizer->AddNewModule(newDM);
       }
#endif
  else if (childTypeName=="noise")
       {
     	  newDM = new GateNoise(m_digitizer, DMname);
     	  m_digitizer->AddNewModule(newDM);
       }
  else if (childTypeName=="merger")
         {
       	  newDM = new GateDigitizerMerger(m_digitizer, DMname);
       	  m_digitizer->AddNewModule(newDM);
         }
  else if (childTypeName=="doIModels")
        {
          newDM = new GateDoIModels(m_digitizer, DMname);
       	  m_digitizer->AddNewModule(newDM);
        }
/*
  else if (childTypeName=="energyThresholder")
    newDM = new GateEnergyThresholder(m_digitizer,newInsertionName,50.*keV);
  else if (childTypeName=="localEnergyThresholder")
    newDM = new GateLocalEnergyThresholder(m_digitizer,newInsertionName);
  else if (childTypeName=="DoImodel")
    //newDM = new GateDoIModels(m_digitizer,newInsertionName,G4ThreeVector(0.,0.,1.));
    newDM = new GateDoIModels(m_digitizer,newInsertionName);

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

  else if (childTypeName=="calibration")
    newDM = new GateCalibration(m_digitizer,newInsertionName);

  else if (childTypeName=="adderComptPhotIdeal")
    newDM = new GatePulseAdderComptPhotIdeal(m_digitizer,newInsertionName);
  else if (childTypeName=="adderComptPhotIdealLocal")
    newDM = new GatePulseAdderComptPhotIdealLocal(m_digitizer,newInsertionName);
  else if (childTypeName=="localClustering")
    newDM = new GateLocalClustering(m_digitizer,newInsertionName);
  else if (childTypeName=="clustering")
    newDM = new GateClustering(m_digitizer,newInsertionName);

  else if (childTypeName=="noise")
    newDM = new GateNoise(m_digitizer,newInsertionName);
  else if (childTypeName=="buffer")
    newDM = new GateBuffer(m_digitizer,newInsertionName);


  else if (childTypeName=="systemFilter")
     newDM = new GateSystemFilter(m_digitizer,newInsertionName);
 // else if (childTypeName=="stripSpDiscretization")
  //   newDM = new GateStripSpatialDiscretization(m_digitizer,newInsertionName);
else if (childTypeName=="gridDiscretization")
     newDM = new GateGridDiscretization(m_digitizer,newInsertionName);
else if (childTypeName=="localMultipleRejection")
     newDM = new GateLocalMultipleRejection(m_digitizer,newInsertionName);

*/
 else {
    G4cout << "Singles Digitizer type name '" << childTypeName << "' was not recognized --> insertion request must be ignored!\n";
    return;
  }


  SetNewInsertionBaseName("");
}


G4bool GateSinglesDigitizerMessenger::CheckNameConflict(const G4String& name)
{
  // Check whether an object with the same name already exists in the list
  return ( GetListManager()->FindElement( GetListManager()->GetObjectName() + "/" + name ) != 0 ) ;
}



