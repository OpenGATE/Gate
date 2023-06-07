
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GatePileup
    \brief  Digitizer module modeling a Pileup (maximum energy wins) of a crystal-block

    - The PileupOld is parameterized by its 'depth': digis will be summed up if their volume IDs
      are identical up to this depth. For instance, the default depth is 1: this means that
      digis will be considered as taking place in a same block if the first two figures
      of their volume IDs are identical
    - A second parameter is added : the width of the pilup window

    - The class is largely inspired from the GateReadout class,
      but is aimed to work by time and not by event.

    OK: added to GND in Jan2023

*/

#include "GatePileup.hh"
#include "GatePileupMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include "GateVSystem.hh"

GatePileup::GatePileup(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_depth(0),
   m_volumeName(""),
   m_Pileup(0),
   m_firstEvent(true),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GatePileupMessenger(this);

	GateDigiCollection* tempDigiCollection = new GateDigiCollection(GetName(), m_digitizer-> GetOutputName()); // to create the Digi Collection
	m_waiting = tempDigiCollection->GetVector ();
}


GatePileup::~GatePileup()
{
  delete m_Messenger;

}


void GatePileup::Digitize()
{
 if(m_firstEvent==true)
 {
		if(!m_volumeName.empty() && m_depth!=0)
			GateError("***ERROR*** You can choose Pileup parameter either with /setDepth OR /setPileupVolume!");

		 //////////////DEPTH SETTING/////////
		//set the previously default value for compatibility of users macros
		if(m_volumeName.empty()  && m_depth==0)
			m_depth=1; //previously default value


		//set m_depth according user defined volume name
		if(!m_volumeName.empty()) //only for EnergyWinner
		{
			GateVSystem* m_system =  ((GateSinglesDigitizer*)this->GetDigitizer())->GetSystem();
			if (m_system==NULL) G4Exception( "GatePileup::Digitize", "Digitize", FatalException,
													   "Failed to get the system corresponding to that digitizer. Abort.\n");

			G4int systemDepth = m_system->GetTreeDepth();

			GateObjectStore* anInserterStore = GateObjectStore::GetInstance();
			for (G4int i=1;i<systemDepth;i++)
			{
				GateSystemComponent* comp0= (m_system->MakeComponentListAtLevel(i))[0][0];
				GateVVolume *creator = comp0->GetCreator();
				GateVVolume* anInserter = anInserterStore->FindCreator(m_volumeName);

				if(creator==anInserter)
					m_depth=i;

			}
		}
		m_firstEvent=false;
 	 }

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;


	std::vector<GateDigi*>::iterator iter;


	std::vector<std::vector<GateDigi*>::iterator> toDel;


  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  G4double minTime = ComputeStartTime(IDC);


	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  if(i==0) //first pass of an event
		  {
			  for (iter = m_waiting->begin() ; iter != m_waiting->end() ; ++iter )
			  {
				  if ( (*iter)->GetTime()+m_Pileup<minTime)
				  {
					  m_OutputDigiCollection->insert((*iter));
					  toDel.push_back(iter);
				  }
			  }

			  for (int i= (int)toDel.size()-1;i>=0;i--)
			  {
				  m_waiting->erase( toDel[i] );
			  }
		  }

		  //from ProcessOnePulse
		    const GateOutputVolumeID& blockID  = inputDigi->GetOutputVolumeID().Top(m_depth);

		    if (blockID.IsInvalid()) {
		       if (nVerboseLevel>1)
		         	G4cout << "[GatePileup::Digitizer]: out-of-block hit for \n"
		   	      <<  *inputDigi << Gateendl
		   	      << " -> digi ignored\n\n";
		       return;
		     }


		     for (iter = m_waiting->begin() ; iter != m_waiting->end() ; ++iter )
		       if ( ((*iter)->GetOutputVolumeID().Top(m_depth) == blockID )
		            &&  (std::abs((*iter)->GetTime()-inputDigi->GetTime())<m_Pileup) )
		         break;

		     if ( iter != m_waiting->end() )
		     {
		        G4double energySum = (*iter)->GetEnergy() + inputDigi->GetEnergy();

		        if ( inputDigi->GetEnergy() > (*iter)->GetEnergy() )
		        {
		        	G4double time = std::max( (*iter)->GetTime() ,inputDigi->GetTime());
		         	**iter = *inputDigi;
		         	(*iter)->SetTime(time);
		        }
		        (*iter)->SetEnergy(energySum);

		        if (nVerboseLevel>1)
		         	  G4cout  << "Overwritten previous digi for block " << blockID << " with new digi with higher energy.\n"
		         	          << "Resulting digi is: \n"
							  << **iter << Gateendl << Gateendl ;
		     }
		      else
		      {
		    	  m_outputDigi = new GateDigi(*inputDigi);

		    	  if (nVerboseLevel>1)
		    		  G4cout << "Created new digi for block " << blockID << ".\n"
					  << "Resulting digi is: \n"
					  << *m_outputDigi << Gateendl << Gateendl ;

		       	   m_waiting->push_back( m_outputDigi );
		      }

	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GatePileup::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}


///////////////////////////////////////////
////////////// Methods of DM //////////////
///////////////////////////////////////////


// Return the min-time of all digis
G4double GatePileup::ComputeStartTime(GateDigiCollection* inputDigiCollection) const
{

	std::vector< GateDigi* >* inputDigiCollectionVector = inputDigiCollection->GetVector();
	std::vector<GateDigi*>::iterator iter;

	G4double startTime = DBL_MAX;
	GateDigi* digi=0;

	for (iter = inputDigiCollectionVector->begin() ; iter != inputDigiCollectionVector->end() ; ++iter )
	 {
		 if ( (*iter)->GetTime() < startTime )
		 {
			 startTime  = (*iter)->GetTime();
			 digi = *iter;
		 }
	 }

    return digi? digi->GetTime() : DBL_MAX;
}



void GatePileup::DescribeMyself(size_t indent )
{
	  G4cout << GateTools::Indent(indent) << "Pileup at depth:      " << m_depth << Gateendl;

}
