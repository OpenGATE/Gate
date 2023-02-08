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
	  m_systemList(0),
	  m_collectionID(0),
	  m_isInitialized(1),
	  m_isTheFirstEvent(1),
	  m_recordSingles(0),
	  m_recordCoincidences(0)
{
	//	G4cout<<"GateDigitizerMgr:: constructor "<<  nVerboseLevel<<G4endl;
	fMessenger = new GateDigitizerMgrMessenger(this);
}


GateDigitizerMgr::~GateDigitizerMgr()
{
 G4cout<<"GateDigitizerMgr::~GateDigitizerMgr "<<G4endl;

/*
 	 if ( !m_SDlist.empty() )
 	    { for ( size_t i = 0; i < m_SDlist.size();i++)
 	        delete m_SDlist[i];
 	    m_SDlist.clear();
 	    }
	 if ( !m_digitizerIMList.empty() )
 	    { for ( size_t i = 0; i < m_digitizerIMList.size();i++)
 	        delete m_digitizerIMList[i];
 	   m_digitizerIMList.clear();
 	    }
	 if ( !m_SingleDigitizersList.empty() )
 	    { for ( size_t i = 0; i < m_SingleDigitizersList.size();i++)
 	        delete m_SingleDigitizersList[i];
 	   m_SingleDigitizersList.clear();
 	    }
	 if ( !m_CoincidenceSortersList.empty() )
 	    { for ( size_t i = 0; i < m_CoincidenceSortersList.size();i++)
 	        delete m_CoincidenceSortersList[i];
 	   m_CoincidenceSortersList.clear();
 	    }
*/

 delete fMessenger;
}

void GateDigitizerMgr::Initialize()
{
	//This function is introduced for speeding up: heavy operations that should not be done at each event

	//G4cout<<"GateDigitizerMgr::Initialize() "<< G4endl;
	//ShowSummary();

	G4DigiManager *fDM = G4DigiManager::GetDMpointer();
	for (long unsigned int i_D = 0; i_D<m_SingleDigitizersList.size(); i_D++)
			{

			for (long unsigned int i_DM = 0; i_DM<m_SingleDigitizersList[i_D]->m_DMlist.size(); i_DM++)
				{
			       m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->InputCollectionID();
				}

			//Save the ID of the last digitizer module for current digitizer
			GateSinglesDigitizer *digitizer=m_SingleDigitizersList[i_D];//
			G4String DigitizerName=digitizer->GetName();

			if(m_SingleDigitizersList[i_D]->m_DMlist.size()>0)
			{
				GateVDigitizerModule * DM = (GateVDigitizerModule*)m_SingleDigitizersList[i_D]->m_DMlist[m_SingleDigitizersList[i_D]->m_DMlist.size()-1];
				G4String name=DM->GetName()+"/"+DigitizerName+"_"+digitizer->m_SD->GetName();
				G4int collectionID  = fDM->GetDigiCollectionID(name);

				m_SingleDigitizersList[i_D]->m_outputDigiCollectionID=collectionID;
				m_SingleDigitizersList[i_D]->m_lastDMname=name;
			}

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

	//TODO: check if we have coincidecnes, i.e. that it is PET and not SPECT or the loop will not enter and it is ok
	//set default input collections for coincidence sorters

	/*if (m_recordCoincidences)
	{
		for (long unsigned int i = 0; i<m_CoincidenceSortersList.size(); i++)
			{
			if ( m_CoincidenceSortersList[i]->GetInputName().empty() )
				{
					if (m_SingleDigitizersList.size()>1)
						GateError("***ERROR*** The input collection name is ambiguous as you have several SinglesDigitizers! \n Please, use /setInputName for your CoincidenceSorter to choose the correct one.\n");

					if (m_SDlist.size()==1)
					{
						//G4cout<<"Setting default Input name"<<  m_SDlist[0]->GetName()<<G4endl;
						m_CoincidenceSortersList[i]->SetInputName("Singles_"+m_SDlist[0]->GetName());
					}
					else
						GateError("***ERROR*** The input collection name is ambiguous as you attached several Sensitive Detectors! \n Please, use /setInputName for your CoincidenceSorter to choose the correct one.\n");
				}
			}
	}
	 */


}


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
      if(!(m_SingleDigitizersList[i]->GetSystem()))
    	  m_SingleDigitizersList[i]->SetSystem((*m_systemList)[0]);
    }

  for (i=0; i<m_CoincidenceSortersList.size() ; ++i)
    {
      if(!((m_CoincidenceSortersList[i])->GetSystem()))
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
 /* if(m_systemList && m_systemList->size() == 1)
	  digitizer->SetSystem((*m_systemList)[0]);
   */

}
//-----------------------------------------------------------------



//-----------------------------------------------------------------
// Integrates a new pulse-processor chain
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
  if(m_systemList && m_systemList->size() == 1)
	  digitizer->SetSystem((*m_systemList)[0]);
}
//-----------------------------------------------------------------

/*
//-----------------------------------------------------------------
// Integrates a new pulse-processor chain
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


}
//-----------------------------------------------------------------

*/

GateClockDependent* GateDigitizerMgr::FindElement(G4String mName)
{

	GateClockDependent* element;
	element = (GateClockDependent*)FindDigitizer(mName);
	// UNCOMM FOR COIN if (!element) element = (GateClockDependent*)FindCoincidenceSorter(mName);

	return element;
}


GateSinglesDigitizer* GateDigitizerMgr::FindDigitizer(G4String mName)
{

	//G4cout<<"GateDigitizerMgr::FindDigitizer "<< m_SingleDigitizersList.size() <<G4endl;

	for(G4int i=0;i<int(m_SingleDigitizersList.size());i++)
		{
		G4String DigitizerName = m_SingleDigitizersList[i]->m_digitizerName+"_" +m_SingleDigitizersList[i]->m_SD->GetName();
		//G4cout << DigitizerName << " "<< mName<< G4endl;
		if(DigitizerName == mName)
			return m_SingleDigitizersList [i];
		}
	return NULL;
}
/*
GateCoincidenceSorter* GateDigitizerMgr::FindCoincidenceSorter(G4String mName)
{
	for(G4int i=0;i<int(m_CoincidenceSortersList.size());i++)
	{
		if(m_CoincidenceSortersList[i]->GetOutputName() == mName)
			return m_CoincidenceSortersList[i];
	}
return NULL;
}
*/





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

	for (long unsigned int i = 0; i<m_digitizerIMList.size(); i++)
		{
		if (nVerboseLevel>1)
				G4cout << "[GateDigitizerMgr::RunDigitizers]: Running GateDigitizerInitializationModule " << m_SingleDigitizersList[i]->m_digitizerName <<" with "<< m_SingleDigitizersList[i]->m_DMlist.size() << " Digitizer Modules\n";

			m_digitizerIMList[i]->Digitize();

			}

	G4DigiManager *fDM = G4DigiManager::GetDMpointer();
	//G4cout<< "m_collectionID = "<< m_collectionID<<G4endl;
	if (nVerboseLevel>1)
	   G4cout << "[GateDigitizerMgr::RunDigitizers]: launching SingleDigitizers. N = " << m_SingleDigitizersList.size() << "\n";
	   //loops over all digitizers/collections
	   	//collID get from G4DigiManager
		for (long unsigned int i_D = 0; i_D<m_SingleDigitizersList.size(); i_D++)
		{
			if (nVerboseLevel>1)
				G4cout << "[GateDigitizerMgr::RunDigitizers]: Running SingleDigitizer " << m_SingleDigitizersList[i_D]->m_digitizerName <<" with "<< m_SingleDigitizersList[i_D]->m_DMlist.size() << " Digitizer Modules\n";
			//loop over all DMs of the current digitizer
			for (long unsigned int i_DM = 0; i_DM<m_SingleDigitizersList[i_D]->m_DMlist.size(); i_DM++)
			{
				if (nVerboseLevel>2)
				G4cout << "[GateDigitizerMgr::RunDigitizers]: Running DigitizerModule " << m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->GetName()<<" "<<	m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->GetNumberOfCollections ()<<" "<<m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->GetCollectionName (0)<< "\n";

				m_SingleDigitizersList[i_D]->m_DMlist[i_DM]->Digitize();
			}

		}

}
/*
void GateDigitizerMgr::RunCoincidenceSorters()
{
	if ( !IsEnabled() )
		return;

	if (!m_recordCoincidences)
		return;

	if (nVerboseLevel>1)
	    G4cout << "[GateDigitizerMgr::RunCoincidenceSorters]: starting\n";


	if (nVerboseLevel>1)
		G4cout << "[GateDigitizerMgr::RunCoincidenceSorters]: launching CoincidenceSorters. N = " << m_CoincidenceSortersList.size() << "\n";


	for (long unsigned int i = 0; i<m_CoincidenceSortersList.size(); i++) //DigitizerList
		{
			if (nVerboseLevel>1)
				G4cout << "[GateDigitizerMgr::RunCoincidenceSorters]: Running CoincidenceSorter "<< m_CoincidenceSortersList[i]->m_coincidenceSorterName << "\n";

			m_CoincidenceSortersList[i]->Digitize();

			//m_collectionID++;
			//m_CoincidenceSortersList[i]->m_outputDigiCollectionID=m_collectionID;
		}
			//Save the name of the last digitizer module for current digitizer

			//G4cout<<"coll ID"<< m_collectionID<< "for "<<  <<G4endl;
			//m_SingleDigitizersList[i]->SetLastDMname();
}


void GateDigitizerMgr::RunCoincidenceDigitizers()
{


	//m_coincidenceSorter->ProcessSingles(); //Digitizer() ??


		//RunCoinDigitizers
}
*/
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
