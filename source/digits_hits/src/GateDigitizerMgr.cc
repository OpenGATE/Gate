/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateDigitizerMgr
  GND (Gate New Digitizer) class
  2022 olga.kochebina@cea.fr

  The concept is slightly different for the old digitizer:
  Digitizer Modules (adder, readout, etc.) now are G4VDigitizerModule

*/

#include "GateDigitizerMgr.hh"
#include "GateDigitizerMgrMessenger.hh"
#include "GateDigitizerInitializationModule.hh"
#include "GateHit.hh"
#include "GateOutputMgr.hh"
#include "GateToRoot.hh"



#include "G4SystemOfUnits.hh"
#include "G4DigiManager.hh"
#include "G4RunManager.hh"

#include "GateVDigitizerModule.hh"


GateDigitizerMgr* GateDigitizerMgr::theDigitizerMgr=0;

//-----------------------------------------------------------------
GateDigitizerMgr* GateDigitizerMgr::GetInstance()
{
  if (!theDigitizerMgr)
    theDigitizerMgr = new GateDigitizerMgr;
  return theDigitizerMgr;
}


GateDigitizerMgr::GateDigitizerMgr()
	: GateClockDependent("digitizerMgr"),
	  m_elementTypeName("DigitizerMgr module"),
	  //m_system(0),
	  m_systemList(0),
	  m_collectionID(0),
	  m_isInitialized(1),
	  m_isTheFirstEvent(1),
	  m_recordSingles(0),
	  m_recordCoincidences(0),
	  m_alreadyRun(false)

{
	//	G4cout<<"GateDigitizerMgr:: constructor "<<  nVerboseLevel<<G4endl;
	fMessenger = new GateDigitizerMgrMessenger(this);
}


GateDigitizerMgr::~GateDigitizerMgr()
{

 delete fMessenger;
}

void GateDigitizerMgr::Initialize()
{
	//This function is introduced for speeding up: heavy operations that should not be done at each event

	//G4cout<<"GateDigitizerMgr::Initialize() "<< G4endl;
	//ShowSummary();

	G4DigiManager *fDM = G4DigiManager::GetDMpointer();


	//For SinglesDigitizers:
	//  - Setting collectionID for all DMs
	//  - Setting output collection ID to written down by output module
	for (size_t i_D = 0; i_D<m_SingleDigitizersList.size(); i_D++)
			{
		//G4cout<<m_SingleDigitizersList[i_D]->GetName() << G4endl;

		m_SingleDigitizersList[i_D]->SetDMCollectionIDs();
		m_SingleDigitizersList[i_D]->SetOutputCollectionID();

			if (m_recordSingles)
				{
					if(!m_SingleDigitizersList[i_D]->m_recordFlag)
					{
						G4cout << " <!> *** WARNING *** <!> SinglesDigitizer "<< m_SingleDigitizersList[i_D]->GetName() <<" is set. "
						"However the output flag is not set to 1 for it, so it will not be written down" <<G4endl ;
						G4cout<<"Please, check if your output options are correct"<<G4endl;
					}
				}
			//G4cout<<"coll ID "<< m_collectionID<< " for "<<m_SingleDigitizersList[i_D]->GetName()<< " "<< m_SingleDigitizersList[i_D]->m_outputDigiCollectionID<<G4endl;

			}

	//TODO: check if we have coincidences, i.e. that it is PET and not SPECT or the loop will not enter and it is ok

	//For CoincidenceSorters:
	// - Setting default input collections
	if (m_recordCoincidences)
	{
		for (size_t i = 0; i<m_CoincidenceSortersList.size(); i++)
			{
			//G4cout<<m_CoincidenceSortersList[i]->GetName() << G4endl;

			if ( m_CoincidenceSortersList[i]->GetInputName().empty() )
				{
					if (m_SingleDigitizersList.size()>1)
						GateError("***ERROR*** CoincidenceSorter *** The input collection name is ambiguous as you have several Singles Collections/SinglesDigitizers! \n Please, use /setInputCollection for your CoincidenceSorter to choose the correct one.\n");

					if (m_SDlist.size()==1)
					{
						//G4cout<<"Setting default Input name"<<  m_SDlist[0]->GetName()<<G4endl;
						//m_CoincidenceSortersList[i]->SetInputName("Singles_"+m_SDlist[0]->GetName());
						m_CoincidenceSortersList[i]->SetInputName("Singles_"+m_SDlist[0]->GetName());
					}
					else
						GateError("***ERROR*** The input collection name is ambiguous as you attached several Sensitive Detectors! \n Please, use /setInputCollection for your CoincidenceSorter to choose the correct one.\n");
				}
			}
	}

	//For CoincidenceDigitizers:
	//  - Setting collectionID for all Coincidence DMs
	//  - Setting output collection ID to written down by output module
	for (size_t i_D = 0; i_D<m_CoincidenceDigitizersList.size(); i_D++)
	{
		m_CoincidenceDigitizersList[i_D]->SetCDMCollectionIDs();
		m_CoincidenceDigitizersList[i_D]->SetOutputCollectionID();

		//G4cout<<m_CoincidenceDigitizersList[i_D]->GetName() << G4endl;
		if (m_recordCoincidences)
		{
			if(!m_CoincidenceDigitizersList[i_D]->m_recordFlag)
			{
				G4cout << " <!> *** WARNING *** <!> CoincidenceDigitizer "<< m_CoincidenceDigitizersList[i_D]->GetName() <<" is set. "
						"However the output flag is not set to 1 for it, so it will not be written down" <<G4endl ;
				G4cout<<"Please, check if your output options are correct"<<G4endl;
			}
		}
		//G4cout<<"coll ID "<< m_collectionID<< " for "<<m_SingleDigitizersList[i_D]->GetName()<< " "<< m_SingleDigitizersList[i_D]->m_outputDigiCollectionID<<G4endl;

	}

}

/*
//-----------------------------------------------------------------
void GateDigitizerMgr::SetSystem(GateVSystem* aSystem)
{
  //m_system = aSystem;
  size_t i;
  for (i=0; i<m_SingleDigitizersList.size() ; ++i)
	  m_SingleDigitizersList[i]->SetSystem(aSystem);
  for (i=0; i<m_CoincidenceSortersList.size() ; ++i)
	  m_CoincidenceSortersList[i]->SetSystem(aSystem);
}
//-----------------------------------------------------------------
*/


//-----------------------------------------------------------------
// The next three methods were added for the multi-system approach
void GateDigitizerMgr::AddSystem(GateVSystem* aSystem)
{

  if(!m_systemList)
    m_systemList = new GateSystemList;

  m_systemList->push_back(aSystem);

  // mhadi_Note: We have here only one digitizer, this is the default digitizer created at the detector construction stage and
  //             which has the "Singles" outputName. So this loop here is to set a system to this digitizer.
  size_t i;
  for (i=0; i<m_SingleDigitizersList.size() ; ++i)
    {
   	  m_SingleDigitizersList[i]->SetSystem((*m_systemList)[0]);
    }

  for (i=0; i<m_CoincidenceSortersList.size() ; ++i)
    {
        m_CoincidenceSortersList[i]->SetSystem((*m_systemList)[0]);
    }

}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
GateVSystem* GateDigitizerMgr::FindSystem(GateSinglesDigitizer* digitizer)
{
  GateVSystem* system = 0;
  if(digitizer->size() != 0)
    {
      G4String sysName = digitizer->GetDigitizerModule(0)->GetObjectName();
      system = FindSystem(sysName);
    }
  return system;
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
GateVSystem* GateDigitizerMgr::FindSystem(G4String& systemName)
{
  G4int index = -1;
  for(size_t i=0; i<m_systemList->size(); i++)
    {
      if(systemName.compare(m_systemList->at(i)->GetOwnName()) == 0)
        index = i;
    }

  if(index != -1)
    return m_systemList->at(index);
  else return 0;

}
//-----------------------------------------------------------------



//-----------------------------------------------------------------
// Integrates a new pulse-processor chain
void GateDigitizerMgr::AddNewSD(GateCrystalSD* newSD)
{

  //Add digitizer to the list
	m_SDlist.push_back(newSD);

  //TODO add here multisystem??
   //! Next lines are for the multi-system approach
 // if(m_systemList && m_systemList->size() == 1)
//	  digitizer->SetSystem((*m_systemList)[0]);


}
//-----------------------------------------------------------------



//-----------------------------------------------------------------
// Integrates a new Singles Digitizer
void GateDigitizerMgr::AddNewSinglesDigitizer(GateSinglesDigitizer* digitizer)
{
  GateDigitizerInitializationModule * myDM = new GateDigitizerInitializationModule(digitizer);
  m_digitizerIMList.push_back(myDM);
  G4DigiManager::GetDMpointer()->AddNewModule(myDM);

  
  G4String outputName = digitizer->GetOutputName() ;
  if (nVerboseLevel>1)
    G4cout << "[GateDigitizerMgr::AddNewSinglesDigitizer]: Storing new digitizer '" << digitizer->GetObjectName() << "'"
           << " with output pulse-list name '" << outputName << "'\n";

  //Add digitizer to the list
  m_SingleDigitizersList.push_back(digitizer);

  //! Next lines are for the multi-system approach
  if(m_systemList && m_systemList->size() == m_SDlist.size())
	  digitizer->SetSystem((*m_systemList)[0]);
}
//-----------------------------------------------------------------


//-----------------------------------------------------------------
// Integrates a new Coincidence Sorter
void GateDigitizerMgr::AddNewCoincidenceSorter(GateCoincidenceSorter* coincidenceSorter)
{
	G4String outputName = coincidenceSorter->GetOutputName() ;
	  if (nVerboseLevel>1)
	    G4cout << "[GateDigitizerMgr::AddNewCoincidenceSorter]: Storing new coincidence sorter '" << coincidenceSorter->GetObjectName() << "'"
	           << " with output coincidence-pulse name '" << outputName << "'\n";

	 m_CoincidenceSortersList.push_back ( coincidenceSorter );

	/*
	 * TODO GND: adapt for multiple SD
	 *  //mhadi_add[
	  //! Next lines are for the multi-system approach
	  for (size_t i=0; i<m_SingleDigitizersList.size() ; ++i)
	    {
	      G4String pPCOutputName = m_SingleDigitizersList[i]->GetOutputName();

	      if(pPCOutputName.compare("Singles") == 0)
	        {
	          coincidenceSorter->SetSystem(m_SingleDigitizersList[i]->GetSystem());
	          break;
	        }
	    }
	  //mhadi_add]

*/
}
//-----------------------------------------------------------------


//-----------------------------------------------------------------
// Integrates a new Coincidence Digitizer
void GateDigitizerMgr::AddNewCoincidenceDigitizer(GateCoincidenceDigitizer* CoinDigitizer)
{
	  G4String outputName = CoinDigitizer->GetOutputName() ;
	  if (nVerboseLevel)
	    G4cout << "[GateDigitizer::StoreNewPulseProcessorChain]: Storing new processor chain '" << CoinDigitizer->GetObjectName() << "'"
	           << " with output pulse-list name '" << outputName << "'\n";

	  m_CoincidenceDigitizersList.push_back( CoinDigitizer );

	  //OK GND: TODO GateCoincidenceDigitizerInitializationModule
	  GateCoinDigitizerInitializationModule * myDM = new GateCoinDigitizerInitializationModule(CoinDigitizer);

	  m_coinDigitizerIMList.push_back(myDM);
	  G4DigiManager::GetDMpointer()->AddNewModule(myDM);





	  //OK GND TODO: systems!
	 /* //! Next lines are for the multi-system approach
	  if(CoinDigitizer->GetInputNames().size() == 0)
	    {
	      G4int index = -1;
	      for(size_t i=0; i<GetCoinSorterList().size(); i++)
	        {
	          G4String coincSorterChainName = GetCoinSorterList()[i]->GetOutputName();
	          if(coincSorterChainName.compare("Coincidences") == 0)
	            {
	              index = i;
	              break;
	            }
	        }
	      CoinDigitizer->SetSystem(m_coincidenceSorterList[index]->GetSystem());
	    }
	}
	  */


}
//-----------------------------------------------------------------


GateClockDependent* GateDigitizerMgr::FindElement(G4String mName)
{


	GateClockDependent* element;
	element = (GateClockDependent*)FindSinglesDigitizer(mName);
	if (!element) element = (GateClockDependent*)FindCoincidenceSorter(mName);
	if (!element) element = (GateClockDependent*)FindCoincidenceDigitizer(mName);

	return element;
}


GateSinglesDigitizer* GateDigitizerMgr::FindSinglesDigitizer(G4String mName)
{

	for(G4int i=0;i<int(m_SingleDigitizersList.size());i++)
		{
		G4String DigitizerName = m_SingleDigitizersList[i]->m_digitizerName+"_" +m_SingleDigitizersList[i]->m_SD->GetName();
		if(DigitizerName == mName)
			return m_SingleDigitizersList [i];
		}
	return NULL;
}

GateCoincidenceSorter* GateDigitizerMgr::FindCoincidenceSorter(G4String mName)
{
	for(G4int i=0;i<int(m_CoincidenceSortersList.size());i++)
	{
		if(m_CoincidenceSortersList[i]->GetOutputName() == mName)
			return m_CoincidenceSortersList[i];
	}
return NULL;
}

GateCoincidenceDigitizer* GateDigitizerMgr::FindCoincidenceDigitizer(G4String mName)
{

	for(G4int i=0;i<int(m_CoincidenceDigitizersList.size());i++)
		{
		G4String CoinDigitizerName = m_CoincidenceDigitizersList[i]->GetName();
		if(CoinDigitizerName == mName)
			return m_CoincidenceDigitizersList [i];
		}
	return NULL;
}





/////////////////
void GateDigitizerMgr::RunDigitizers()
{
	//G4cout<<"GateDigitizerMgr::RunDigitizers"<<G4endl;
	//ShowSummary();

	if ( !IsEnabled() )
		return;

	if (m_isTheFirstEvent==true)
	{
		Initialize();
		m_isTheFirstEvent=false;
	}



	if (nVerboseLevel>1)
	    G4cout << "[GateDigitizerMgr::RunDigitizers]: starting\n";





	//Run Initialization Module to convert Hits to Digis
	if (nVerboseLevel>1)
			    G4cout << "[GateDigitizerMgr::RunDigitizers]: launching GateDigitizerInitializationModule\n";

	for (size_t i = 0; i<m_digitizerIMList.size(); i++)
		{
		if (nVerboseLevel>1)
				G4cout << "[GateDigitizerMgr::RunDigitizers]: Running GateDigitizerInitializationModule " << m_SingleDigitizersList[i]->m_digitizerName <<" with "<< m_SingleDigitizersList[i]->m_DMlist.size() << " Digitizer Modules\n";

			m_digitizerIMList[i]->Digitize();

		}

	if (nVerboseLevel>1)
	   G4cout << "[GateDigitizerMgr::RunDigitizers]: launching SingleDigitizers. N = " << m_SingleDigitizersList.size() << "\n";
	   //loops over all digitizers/collections
	   	//collID get from G4DigiManager
		for (size_t i_D = 0; i_D<m_SingleDigitizersList.size(); i_D++)
		{
			if (nVerboseLevel>1)
				G4cout << "[GateDigitizerMgr::RunDigitizers]: Running SingleDigitizer " << m_SingleDigitizersList[i_D]->m_digitizerName <<" with "<< m_SingleDigitizersList[i_D]->m_DMlist.size() << " Digitizer Modules\n";
			//loop over all DMs of the current digitizer
			for (size_t i_DM = 0; i_DM<m_SingleDigitizersList[i_D]->m_DMlist.size(); i_DM++)
			{
				if (nVerboseLevel>2)
				G4cout << "[GateDigitizerMgr::RunDigitizers]: Running DigitizerModule " << m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->GetName()<<" "<<	m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->GetNumberOfCollections ()<<" "<<m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->GetCollectionName (0)<< "\n";

				m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->Digitize();
			}

		}

		m_alreadyRun=true;
}

void GateDigitizerMgr::RunCoincidenceSorters()
{

	//G4cout<<"GateDigitizerMgr::RunCoincidenceSorters"<<G4endl;
	if ( !IsEnabled() )
		return;

	if (!m_recordCoincidences)
		return;

	if (nVerboseLevel>1)
	    G4cout << "[GateDigitizerMgr::RunCoincidenceSorters]: starting\n";


	if (nVerboseLevel>1)
		G4cout << "[GateDigitizerMgr::RunCoincidenceSorters]: launching CoincidenceSorters. N = " << m_CoincidenceSortersList.size() << "\n";


	for (size_t i = 0; i<m_CoincidenceSortersList.size(); i++) //DigitizerList
		{
			if (nVerboseLevel>1)
				G4cout << "[GateDigitizerMgr::RunCoincidenceSorters]: Running CoincidenceSorter "<< m_CoincidenceSortersList[i]->m_coincidenceSorterName << "\n";

			m_CoincidenceSortersList[i]->Digitize();
		}

	m_alreadyRun=true;
}


void GateDigitizerMgr::RunCoincidenceDigitizers()
{
	//G4cout<<"GateDigitizerMgr::RunCoincidenceDigitizers()"<<G4endl;

	for (size_t i = 0; i<m_coinDigitizerIMList.size(); i++)
		{
		if (nVerboseLevel>1)
				G4cout << "[GateDigitizerMgr::RunCoincidenceDigitizers]: Running GateCoinDigitizerInitializationModule " << m_CoincidenceDigitizersList[i]->m_digitizerName <<" with "<< m_CoincidenceDigitizersList[i]->m_CDMlist.size() << " Digitizer Modules\n";

		m_coinDigitizerIMList[i]->Digitize();

		}

	if (nVerboseLevel>1)
		   G4cout << "[GateDigitizerMgr::RunCoincidenceDigitizers]: launching CoincidenceDigitizers. N = " << m_CoincidenceDigitizersList.size() << "\n";
		   //loops over all digitizers/collections
		   	//collID get from G4DigiManager
			for (size_t i_D = 0; i_D<m_CoincidenceDigitizersList.size(); i_D++)
			{
				if (nVerboseLevel>1)
					G4cout << "[GateDigitizerMgr::RunCoincidenceDigitizers]: Running CoincidenceDigitizer " << m_CoincidenceDigitizersList[i_D]->m_digitizerName <<" with "<< m_CoincidenceDigitizersList[i_D]->m_CDMlist.size() << " Digitizer Modules\n";
				//loop over all DMs of the current digitizer
				for (size_t i_DM = 0; i_DM<m_CoincidenceDigitizersList[i_D]->m_CDMlist.size(); i_DM++)
				{
					if (nVerboseLevel>2)
					G4cout << "[GateDigitizerMgr::RunCoincidenceDigitizers]: Running DigitizerModule " << m_CoincidenceDigitizersList[i_D]->m_CDMlist[i_DM]->GetName()<<" "<<	m_CoincidenceDigitizersList[i_D]->m_CDMlist[i_DM]->GetNumberOfCollections ()<<" "<<m_CoincidenceDigitizersList[i_D]->m_CDMlist[i_DM]->GetCollectionName (0)<< "\n";

					m_CoincidenceDigitizersList[i_D]->m_CDMlist[i_DM]->Digitize();
				}

			}

	//m_alreadyRun=true;
}

void GateDigitizerMgr::ShowSummary()
{
	G4cout<<"-----------------------"<<G4endl;
	G4cout<<"DigitizerMgr summary"<<G4endl;
	G4DigiManager *fDM = G4DigiManager::GetDMpointer();
	G4DCtable* DCTable=fDM->GetDCtable ();

	G4cout<<"Table size " <<DCTable->entries ()<<G4endl;
	G4cout<< std::left << std::setw(7)<<"collID" <<"| "<< std::left << std::setw(35) <<"DigitizerModule"<<"| "<< std::left << std::setw(20) <<"DigiCollection"<< "| "<< "size "<<G4endl;
	G4cout<<"---------------------------------------------------------------------------------"<<G4endl;
	for (int i=0; i<DCTable->entries ();i++)
	{
		G4String name=DCTable->GetDMname(i)+"/"+DCTable->GetDCname(i);

		if(fDM->GetDigiCollection(DCTable->GetCollectionID(name))==0)
		{
		G4cout<<std::left << std::setw(7) <<DCTable->GetCollectionID(name) <<"| "<< std::left << std::setw(35)<< DCTable->GetDMname(i) <<"| "<< std::left << std::setw(20) <<DCTable->GetDCname(i)<< "| "<< 0 <<G4endl;//" "<<GetDigiCollection(i)->GetName() <<G4endl;
		}
		else
		{
			G4cout<<std::left << std::setw(7) <<DCTable->GetCollectionID(name) <<"| "<< std::left << std::setw(35)<< DCTable->GetDMname(i) <<"| "<< std::left << std::setw(20) <<DCTable->GetDCname(i)<< "| "<< fDM->GetDigiCollection(DCTable->GetCollectionID(name))->GetSize () <<G4endl;//
		}
	}




}
