
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*! \class  GateDeadTime
  \brief  Digitizer Module for a simple dead time discriminator.

  - GateDeadTime - by Luc.Simon@iphe.unil.ch

  - The method Digitize() of this class models a simple
  DeadTime discriminator. User chooses value of dead time, mode
  (paralysable or not) and geometric level of application (crystal, module,...)

  \sa GateVDigitizerModule
  \sa GateVolumeID


  - Added to New Digitizer GND by OK: January 2023
*/


#include "GateDeadTime.hh"
#include "GateDeadTimeMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"
#include "GateSinglesDigitizer.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateDeadTime::GateDeadTime(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_bufferSize(0),
   m_bufferMode(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {

	m_isParalysable = false;
	m_DeadTime = 0;
	m_init_done_run_id = -1;

	G4String colName = digitizer->GetOutputName() ;

	collectionName.push_back(colName);
	m_Messenger = new GateDeadTimeMessenger(this);
}


GateDeadTime::~GateDeadTime()
{
  delete m_Messenger;

}


void GateDeadTime::Digitize()
{

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;


/*
	 if (nVerboseLevel==1)
			    {
			    	G4cout << "[ GateDeadTime::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n";
			    	for (long unsigned int k=0; k<m_OutputDigiCollection->entries();k++)
			    		G4cout << *(*IDC)[k] << Gateendl;
			    		G4cout << Gateendl;
			    }
	*/

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  if (inputDigi->GetRunID() != m_init_done_run_id)
		  {
			  // initialise the DeadTime buffer and table
			  //G4cout<<"GateDeadTime::Digitize "<<m_volumeName<<G4endl;
			  CheckVolumeName(m_volumeName);

			  if (nVerboseLevel>1) {
		       G4cout << "first digi in dead time Digitize () \n" ;
		       G4cout << "DeadTime set at  " << m_DeadTime << " ps"<< Gateendl ;
		       G4cout << "mode = " << (m_isParalysable ? "paralysable":"non-paralysable") << Gateendl ;
			  }
			  m_init_done_run_id = inputDigi->GetRunID();
		   }

		  if (inputDigi->GetEnergy()==0) {
		      if (nVerboseLevel>1)
		        G4cout << "[GateDeadTime::ProcessOneHit]: energy is null for " << inputDigi << " -> digi ignored\n\n";
		      return;
		    }

		  // FIND THE ELEMENT ID OF DIGI


		  const GateVolumeID* aVolumeID = &inputDigi->GetVolumeID();
		  G4int m_generalDetId = 0; // a unique number for each detector part
		                              // that depends of the depth of application
		                              // of the dead time
		  size_t m_depth = (size_t)(aVolumeID->GetCreatorDepth(m_volumeName));

		  m_generalDetId = aVolumeID->GetCopyNo(m_depth);

		    /////// Bug Report - 8/6/2006 - Spencer Bowen - S.Jan ////////
		    /*
		      for (G4int i = 1 ; i < numberOfHigherLevels + 1; i++)
		      {
		      m_generalDetId += aVolumeID->GetCopyNo(m_depth-i) * numberOfComponentForLevel[i-1];
		      }
		    */

		    G4int multFactor = 1;
		    for (G4int i = 1 ; i < numberOfHigherLevels + 1; i++)
		    {
		      multFactor *= numberOfComponentForLevel[i-1];
		      m_generalDetId += aVolumeID->GetCopyNo(m_depth-i)*multFactor;
		    }
		  //////////////////////////////////////////////////////////////
		 // FIND TIME OF DIGI
		     unsigned long long int currentTime = (unsigned long long int)((inputDigi->GetTime())/picosecond);
		     if (nVerboseLevel>5) {
		       G4cout << "A new digi is processed by dead time : " << (inputDigi->GetTime())/picosecond
		              << " =  "<< currentTime  << Gateendl  ;
		       G4cout << "ID elt = " <<  m_generalDetId << Gateendl ;
		       G4cout << "Rebirth time for elt " << m_generalDetId << " = " << m_DeadTimeTable[m_generalDetId]<< Gateendl ;
		     }

		     // IS DETECTOR DEAD ?
		     if (currentTime >= m_DeadTimeTable[m_generalDetId])
		     {
		       // NO, DETECTOR IS NOT DEAD : COPY THIS DIGI TO OUTPUT DIGI COLLECTION
		       m_outputDigi = new GateDigi(*inputDigi);
		       m_OutputDigiCollection->insert(m_outputDigi);

		       //  m_DeadTimeTable[m_generalDetId] = currentTime + m_DeadTime;
		       if (m_bufferSize>1){
		         m_bufferCurrentSize[m_generalDetId]++;
		         if (m_bufferCurrentSize[m_generalDetId]==m_bufferSize){
		           m_DeadTimeTable[m_generalDetId] = currentTime + m_DeadTime;
		           m_bufferCurrentSize[m_generalDetId]=0;
		         }
		       } else {
		         m_DeadTimeTable[m_generalDetId] = currentTime + m_DeadTime;
		       }
		       if (nVerboseLevel>5){
		         G4cout << "We have accept " << currentTime << " a digi in element " << m_generalDetId <<
		           "\trebirth time\t" << m_DeadTimeTable[m_generalDetId] << Gateendl;
		         G4cout << "Copied digi to output:\n"
		                << *m_outputDigi << Gateendl << Gateendl ;
		       }
		     }
		     else
		       {
		         // YES DETECTOR IS DEAD : REMOVE DIGI
		         if (nVerboseLevel>5)
		           G4cout << "Removed digi, due to dead time:\n";
		         // AND IF "PARALYSABLE" DEAD TIME, MAKE THE DEATH OF DETECTOR LONGER
		         if ((m_bufferSize>1) && (m_bufferMode==1)){
		           if (m_bufferCurrentSize[m_generalDetId]<m_bufferSize-1) {
		             m_bufferCurrentSize[m_generalDetId]++;
		             m_OutputDigiCollection->insert(m_outputDigi);

		           }
		         } else {
		         	if (m_isParalysable && (m_bufferSize<2)){
		             m_DeadTimeTable[m_generalDetId]  = currentTime + m_DeadTime;
		           }
		         }
		       }
	  } //loop  over input digits
  } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateDeadTime::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}


void GateDeadTime::SetDeadTimeMode(G4String val)
{
  if ((val!="paralysable")&&(val!="nonparalysable"))
    GateError (" Wrong dead time mode : candidates are : paralysable nonparalysable");
  else
	  if( val =="paralysable")
	  {
		  m_isParalysable = (val=="paralysable");
	  }
	  else if ( val =="nonparalysable")
	  {
		  m_isParalysable = (val=="nonparalysable");
	  }
}


///////////////////////////////////////////
////////////// Methods of DM //////////////
///////////////////////////////////////////

void GateDeadTime::CheckVolumeName(G4String val)
{
  GateObjectStore* anInserterStore = GateObjectStore::GetInstance();

  if (anInserterStore->FindCreator(val)) {
    m_volumeName = val;

    FindLevelsParams(anInserterStore);
    m_testVolume = 1;
  }
  else {
	  GateError("***ERROR*** Wrong Volume Name. Abort.");

  }
}



void GateDeadTime::FindLevelsParams(GateObjectStore*  anInserterStore)
{
  G4int numberTotalOfComponentInSystem = 0;
  GateVVolume* anInserter = anInserterStore->FindCreator(m_volumeName);
  GateVVolume* anotherInserter = anInserter; // just to buffer anInserter

  if (nVerboseLevel>1)
    G4cout << "DEAD TIME IS APPLIED ON " <<  m_volumeName << Gateendl;

  // How many levels higher than volumeName level ?
  numberOfHigherLevels = 0;
  while(anotherInserter->GetMotherList()) {
    anotherInserter =  anotherInserter->GetMotherList()->GetCreator();
    numberOfHigherLevels ++;
  }
  //  numberOfHigherLevels--;
  anotherInserter = anInserter;

  // How many components for each levels ?
  numberOfComponentForLevel.resize(numberOfHigherLevels);
  if (numberOfHigherLevels < 1) {
    G4cout << "[GateDeadTime::FindLevelsParams]: ERROR numberOfHigherLevels is zero.\n\n";
    return;
  }

  numberOfComponentForLevel[0] = anotherInserter->GetVolumeNumber();

  for (G4int i = 1 ; i < numberOfHigherLevels ; i++) {
    anotherInserter = anotherInserter->GetMotherList()->GetCreator();
    numberOfComponentForLevel[i] = anotherInserter->GetVolumeNumber();
  }

  numberTotalOfComponentInSystem = 1;
  for (G4int i2 = 0 ; i2 < numberOfHigherLevels ; i2++) {
    numberTotalOfComponentInSystem = numberTotalOfComponentInSystem * numberOfComponentForLevel[i2];
    if (nVerboseLevel>5)
      G4cout << "Level : " << i2 << " has "
             << numberOfComponentForLevel[i2] << " elements\n";
  }

  if (nVerboseLevel>5)
    G4cout << "total number of elements = " <<numberTotalOfComponentInSystem << Gateendl;

  // create the table of "rebirth time" (detector is dead than it rebirth)
  m_DeadTimeTable.resize(numberTotalOfComponentInSystem);
  m_bufferCurrentSize.resize(numberTotalOfComponentInSystem);

  for (G4int i=0;i<numberTotalOfComponentInSystem;i++) {
    m_DeadTimeTable[i] = 0;
    m_bufferCurrentSize[i] = 0.;
  }
}

void GateDeadTime::DescribeMyself(size_t indent )
{
	  G4cout << GateTools::Indent(indent) << "DeadTime: " << G4BestUnit(m_DeadTime,"Time") << Gateendl;
}
