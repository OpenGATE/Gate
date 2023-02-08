
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateVDigitizerModule
	- This class is virtual class to construct DigitizerModules from

	- Use GateDummyDigitizerModule and GateDummyDigitizerModuleMessenger class
	to create your DigitizerModule and its messenger

*/
#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include "GateOutputMgr.hh"

GateVDigitizerModule::GateVDigitizerModule(G4String name, G4String path, GateSinglesDigitizer *digitizer,  GateCrystalSD* SD)
  :G4VDigitizerModule(name),
   GateClockDependent(path),
   m_digitizer(digitizer),
   m_SD(SD)
{

	GateOutputMgr::GetInstance()->RegisterNewSingleDigiCollection(digitizer->GetName()+"_"+ SD->GetName()+"_"+name, false);

}

GateVDigitizerModule::GateVDigitizerModule(G4String name)
  :G4VDigitizerModule(name),
   GateClockDependent(name)
 {
 }





GateVDigitizerModule::~GateVDigitizerModule()
{
}



void GateVDigitizerModule::Describe(size_t indent)
{
  GateClockDependent::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Attached to:        '" << m_digitizer->GetObjectName() << "'\n";
  G4cout << GateTools::Indent(indent) << "Output:             '" << GetObjectName() << "'\n";
  DescribeMyself(indent);
}

//////////////////
void GateVDigitizerModule::InputCollectionID()
{

	GateDigitizerMgr* DigiMan = GateDigitizerMgr::GetInstance();
	G4DigiManager* fDM = G4DigiManager::GetDMpointer();

	G4String DigitizerName=m_digitizer->GetName();

//	DigiMan->ShowSummary();

	G4String outputCollNameTMP = GetName() +"/"+DigitizerName+"_"+m_SD->GetName();
	G4int DCID = -1;

	if(DCID<0)
	{
		DCID = fDM->GetDigiCollectionID(outputCollNameTMP);
	}

	G4String InitDMname="DigiInit/"+DigitizerName+"_"+m_SD->GetName();
	G4int InitDMID = fDM->GetDigiCollectionID(InitDMname);

	//check if this module is the first in this digitizer
	if ( m_digitizer->m_DMlist[0] == this )
	{
		//check if the input collection is from InitDM
		if (m_digitizer->GetInputName() == m_digitizer->GetOutputName() )
		{
			DCID=InitDMID;
		}
		else
		{
			G4String inputCollectionName = m_digitizer->GetInputName()+"_"+m_digitizer->m_SD->GetName();
			GateSinglesDigitizer* inputDigitizer = DigiMan->FindDigitizer(inputCollectionName);
			DCID=inputDigitizer->m_outputDigiCollectionID;
		}
	}
	else
	{
		//sequential
		DCID=DCID-1;
	}




	if(DCID<0)
	{
      G4Exception( "GateVDigitizerModule::InputCollectionID", "InputCollectionID", FatalException, "Something wrong with collection ID. Please, contact olga[dot]kochebina[at]cea.fr. Abort.\n");
	}
// G4cout<<DCID<<G4endl;

 m_DCID = DCID;

}















