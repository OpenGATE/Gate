/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateAnalysisMessenger.hh"
#include "GateAnalysis.hh"
#include "GateOutputMgr.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

#include "GateObjectStore.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateAnalysisMessenger::GateAnalysisMessenger(GateAnalysis* gateAnalysis)
  : GateOutputModuleMessenger(gateAnalysis)
  , m_gateAnalysis(gateAnalysis)
{
	// HDS : septal penetration
	G4String cmdName;

	cmdName = GetDirectoryName()+"setSeptalVolumeName";
	SetSeptalVolumeNameCmd = new G4UIcmdWithAString(cmdName,this);
	SetSeptalVolumeNameCmd->SetGuidance("Set the name of the volume in which you want to record septal penetration");
	SetSeptalVolumeNameCmd->SetParameterName("SeptalVolumeName",false);

	cmdName = GetDirectoryName()+"recordSeptalPenetration";
  	RecordSeptalCmd = new G4UIcmdWithABool(cmdName,this);
  	RecordSeptalCmd->SetGuidance("Set the flag for recording septal penetration into hits and singles trees");
  	RecordSeptalCmd->SetGuidance("1. true/false");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateAnalysisMessenger::~GateAnalysisMessenger()
{
	delete SetSeptalVolumeNameCmd;
	delete RecordSeptalCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateAnalysisMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
	// HDS : septal
	if ( command == SetSeptalVolumeNameCmd ) {
		// The user must input the logical name but we have to retrieve the physical name
		// We get the instance of the object Creator store to retrieve the object Creator
		// and the physical volume name
		GateObjectStore* theStore = GateObjectStore::GetInstance();
		GateVVolume* volume = theStore->FindVolumeCreator(newValue) ;
		if (volume) m_gateAnalysis->SetSeptalPhysVolumeName(volume->GetPhysicalVolumeName());

	} else if ( command == RecordSeptalCmd ) {
		m_gateAnalysis->SetRecordSeptalFlag(RecordSeptalCmd->GetNewBoolValue(newValue));
	//
	} else {
		GateOutputModuleMessenger::SetNewValue(command,newValue);
	}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....
