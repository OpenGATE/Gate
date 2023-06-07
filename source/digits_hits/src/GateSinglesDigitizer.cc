/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateSinglesDigitizer
  GND (Gate New Digitizer) class
  2022 olga.kochebina@cea.fr

  The concept is slightly different for the old digitizer:
  Digitizer Modules (adder, readout, etc.) now are G4VDigitizerModule

*/

#include "GateSinglesDigitizer.hh"
#include "GateSinglesDigitizerMessenger.hh"
#include "G4UnitsTable.hh"

#include "GateDigitizerMgr.hh"
#include "G4VDigitizerModule.hh"
#include "GateTools.hh"

#include "GateOutputMgr.hh"
#include "GateDigitizerInitializationModule.hh"

GateSinglesDigitizer::GateSinglesDigitizer( GateDigitizerMgr* itsDigitizerMgr,
										  const G4String& digitizerUsersName,
    			                          GateCrystalSD* SD)
  : GateModuleListManager(itsDigitizerMgr,itsDigitizerMgr->GetObjectName() + "/"+ SD->GetName() +"/SinglesDigitizer/" + digitizerUsersName ,"SinglesDigitizer"),
	m_outputName(digitizerUsersName+"_"+SD->GetName()),
    m_inputName(digitizerUsersName+"_"+SD->GetName()),
	m_recordFlag(false),
	m_SD(SD),
	m_digitizerName(digitizerUsersName)
{
	m_messenger = new GateSinglesDigitizerMessenger(this);

	/* Example of naming
 	 * m_digitizerName = Singles, HESingles
	 * m_inputName = Singles_crystal
	 * m_outputName = Singles_crystal, HESingles_crystal
	 */

  //Prepare OutputMng for this digitizer
  	GateOutputMgr::GetInstance()->RegisterNewSingleDigiCollection(m_digitizerName+"_"+SD->GetName(),false);
  	if(!itsDigitizerMgr->m_isInitialized)
  	{

  		itsDigitizerMgr->AddNewSinglesDigitizer(this);

  	}

}




GateSinglesDigitizer::~GateSinglesDigitizer()
{
  for (auto processor = theListOfNamedObject.begin(); processor != theListOfNamedObject.end(); ++processor)
  {
    GateMessage("Core", 5, "~GateSinglesDigitizer -- delete module: " << (*processor)->GetObjectName() << Gateendl );
    delete (*processor);
  }
  delete m_messenger;
}


void GateSinglesDigitizer::AddNewModule(GateVDigitizerModule* DM)
{
	if (nVerboseLevel>1)
		G4cout << "[GateSinglesDigitizer::AddNewModule]: "<< DM->GetName() <<"\n";

	m_DMlist.push_back(DM);
	G4DigiManager::GetDMpointer()->AddNewModule(DM);

	theListOfNamedObject.push_back(DM);

}

GateVDigitizerModule* GateSinglesDigitizer::FindDigitizerModule(const G4String& mName)
{

	for(G4int i=0;i<int(m_DMlist.size());i++)
			{
			G4String DMname = m_DMlist[i]->GetObjectName();//m_digitizerName+"_" +m_SingleDigitizersList[i]->m_SD->GetName();
			//G4cout << DigitizerName << " "<< mName<< G4endl;
			if(DMname == mName)
				return m_DMlist [i];
			}
		return NULL;
}

void GateSinglesDigitizer::Describe(size_t indent)
{
	GateModuleListManager::Describe(indent);
	DescribeMyself(indent);
}



void GateSinglesDigitizer::DescribeMyself(size_t indent)
{
	G4cout<<"Digitizer Describe"<<G4endl;
	G4cout<<"Digitizer Name: "<< m_digitizerName<<G4endl;
	G4cout<<"Input Name: "<< m_inputName<<G4endl;
	G4cout<<"Output Name: "<< m_outputName<<G4endl;
	G4cout<<"Digitizer Modules: "<<G4endl;
	for (size_t j = 0; j<m_DMlist.size(); j++)
			{
				G4cout<<"    " <<m_DMlist[j]->GetName()<<" "<<G4endl;
			}

}

G4String GateSinglesDigitizer::GetDMNameFromInsertionName(G4String name)
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

void GateSinglesDigitizer::SetDMCollectionIDs()
{
	//G4cout<<"GateSinglesDigitizer::SetInputDMCollectionID"<<G4endl;
	//Calculate input IDs for all DMs
	G4DigiManager *fDM = G4DigiManager::GetDMpointer();
	//fDM->List();

	G4String name4fDM;

	for (size_t i_DM = 0; i_DM<m_DMlist.size(); i_DM++)
	{
		if (i_DM == 0) // first DM: could be the Initialization module or some other set but user
		{
			name4fDM = "DigiInit/"+this->GetName()+"_"+this->GetSD()->GetName();
		}
		else
		{
			name4fDM = m_DMlist[i_DM-1]->GetName()+"/"+GetOutputName();
		}
		m_DMlist[i_DM]->SetCollectionID( fDM->GetDigiCollectionID(name4fDM ) );
	}



}
void GateSinglesDigitizer::SetOutputCollectionID()
{

	//Save the ID of the last digitizer module for current digitizer
	//G4cout<<"GateSinglesDigitizer::SetOuptputCollectionID"<<G4endl;

	G4DigiManager *fDM = G4DigiManager::GetDMpointer();

	G4String name;
	if(m_DMlist.size()>0)
	{
		GateVDigitizerModule *DM = (GateVDigitizerModule*)m_DMlist[m_DMlist.size()-1];
		name=DM->GetName()+"/"+m_digitizerName+"_"+m_SD->GetName();
	}
	else
		name="DigiInit/"+m_digitizerName+"_"+m_SD->GetName();

	m_outputDigiCollectionID  = fDM->GetDigiCollectionID(name);

	//G4cout<<"output collecionID "<<m_digitizerName<<" "<<m_outputDigiCollectionID<<G4endl;


}

