/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCoincidenceDigitizer
  GND (Gate New Digitizer) class
  2022 olga.kochebina@cea.fr

  The concept is slightly different for the old digitizer:
  Digitizer Modules (adder, readout, etc.) now are G4VDigitizerModule

*/

#include "GateCoincidenceDigitizer.hh"
#include "GateCoincidenceDigitizerMessenger.hh"
#include "G4UnitsTable.hh"

#include "GateDigitizerMgr.hh"
#include "G4VDigitizerModule.hh"
#include "GateTools.hh"

#include "GateOutputMgr.hh"
#include "GateDigitizerInitializationModule.hh"

GateCoincidenceDigitizer::GateCoincidenceDigitizer( GateDigitizerMgr* itsDigitizerMgr,
										  const G4String& digitizerUsersName)
  : GateModuleListManager(itsDigitizerMgr,itsDigitizerMgr->GetObjectName() + "/CoincidenceDigitizer/" + digitizerUsersName ,"CoincidenceDigitizer"),
	m_outputName(digitizerUsersName),
    m_inputNames(),
	m_recordFlag(false),
	m_digitizerName(digitizerUsersName)
{
	m_messenger = new GateCoincidenceDigitizerMessenger(this);
	/* Example of naming
 	 * m_digitizerName = finalCoin
	 * m_inputName = Coincidences, Delay
	 * m_outputName = finalCoin
	 */


  //Prepare OutputMng for this digitizer
  	GateOutputMgr::GetInstance()->RegisterNewCoincidenceDigiCollection(m_outputName,true);
}




GateCoincidenceDigitizer::~GateCoincidenceDigitizer()
{
  delete m_messenger;
}


void GateCoincidenceDigitizer::AddNewModule(GateVDigitizerModule* DM)
{
	if (nVerboseLevel>1)
		G4cout << "[GateCoincidenceDigitizer::AddNewModule]: "<< DM->GetName() <<"\n";

	m_CDMlist.push_back(DM);
	G4DigiManager::GetDMpointer()->AddNewModule(DM);

	theListOfNamedObject.push_back(DM);

}

GateVDigitizerModule* GateCoincidenceDigitizer::FindDigitizerModule(const G4String& mName)
{

	for(G4int i=0;i<int(m_CDMlist.size());i++)
			{
			G4String DMname = m_CDMlist[i]->GetObjectName();
			if(DMname == mName)
				return m_CDMlist [i];
			}
		return NULL;
}


void GateCoincidenceDigitizer::Describe(size_t indent)
{
	GateModuleListManager::Describe(indent);
	DescribeMyself(indent);
}



void GateCoincidenceDigitizer::DescribeMyself(size_t indent)
{


	G4cout<<"Coincidence Digitizer Describe"<<G4endl;
	G4cout<<"Coincidence Digitizer Name: "<< m_digitizerName<<G4endl;
	G4cout<<"Input Names: "<<G4endl;
		for(G4int i=0; i<(G4int)m_inputNames.size();i++)
			G4cout<<" "<< m_inputNames[i]<<G4endl;
	G4cout<<"Output Name: "<< m_outputName<<G4endl;
	G4cout<<"Coincidence Digitizer Modules: "<<G4endl;
	for (size_t j = 0; j<m_CDMlist.size(); j++)
			{
				G4cout<<"    " <<m_CDMlist[j]->GetName()<<" "<<G4endl;
			}

}


G4String GateCoincidenceDigitizer::GetDMNameFromInsertionName(G4String name)
{
	 size_t pos = 0;
	 std::string token;

	std::string delimiter = "/";
	while ((pos = name.find(delimiter)) != std::string::npos) {
	    token = name.substr(0, pos);
	   name.erase(0, pos + delimiter.length());
	}

	return name;

}



void GateCoincidenceDigitizer::SetCDMCollectionIDs()
{
	//Calculate input IDs for all DMs
	G4DigiManager *fDM = G4DigiManager::GetDMpointer();
	//fDM->List();

	G4String name4fDM;

	for (size_t i_DM = 0; i_DM<m_CDMlist.size(); i_DM++)
	{
		if (i_DM == 0) // first DM: could be the Initialization module or some other set but user
		{
			name4fDM = "CoinDigiInit/"+this->GetName();
		}
		else
		{
			name4fDM = m_CDMlist[i_DM-1]->GetName()+"/"+GetOutputName();
		}
		m_CDMlist[i_DM]->SetCollectionID( fDM->GetDigiCollectionID(name4fDM ) );
	}



}
void GateCoincidenceDigitizer::SetOutputCollectionID()
{

	//Save the ID of the last digitizer module for current digitizer
	//G4cout<<"GateSinglesDigitizer::SetOuptputCollectionID"<<G4endl;

	G4DigiManager *fDM = G4DigiManager::GetDMpointer();

	G4String name;
	if(m_CDMlist.size()>0)
	{
		GateVDigitizerModule *DM = (GateVDigitizerModule*)m_CDMlist[m_CDMlist.size()-1];
		name=DM->GetName()+"/"+m_digitizerName;
	}
	else
		name="CoinDigiInit/"+m_digitizerName;

	m_outputDigiCollectionID  = fDM->GetDigiCollectionID(name);
	//G4cout<<"output collecionID "<<m_digitizerName<<" "<<m_outputDigiCollectionID<<G4endl;


}
