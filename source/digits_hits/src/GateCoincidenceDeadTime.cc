
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCoincidenceDeadTime

  This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).

   - Create your DM by coping this class and GateDummyDigitizerMessenger class for your DM messenger
   - Places to change are marked with // ****** comment and called with "dummy" names
   - Include your module to GateCoincidenceDigitizerMessenger in the method DoInsertion(..)

	If you adapting some already exiting class from Old Gate Digitizer here is some of the tips
	- Digitize () is a fusion of GateVdigiProcessor::ProcessdigiList and GateXXX::ProcessOnedigi
	- digi --> Digi
	- m_outputDigiList --> OutputDigiCollectionVector
	- inputDigi-->inputDigi
	- m_outputDigi --> m_outputDigi
	- how to adapt iterators check GateAdder class

  To create new Digitizer Module (DM), please, follow the steps:
  1) Copy .cc and .hh of GateCoincidenceDeadTime, GateCoincidenceDeadTimeMessenger to GateYourNewDigitizerModule and GateYourNewDigitizerModuleMessenger
  2) Replace in these new files : CoincidenceDeadTime -> YourNewDigitizerModule
  3) Compile 1st time (so not forget to redo ccmake to)
  4) Adapt GateYourNewDigitizerModuleMessenger.cc
  5) Adapt GateYourNewDigitizerModuleMessenger.hh (!!!! DO NOT FORGET TO WRITE A SHORT EXPLANATION ON WHAT DOES YOUR DM !!!!)
  6) Adapt GateYourNewDigitizerModule.hh
  7) Adapt GateYourNewDigitizerModule.cc
  	  - Change the names (x2) of YourDigitizerModule in the constructor of GateYourNewDigitizerModule.cc:
   	   	   	  line:
   	   	   	  :GateVDigitizerModule("Dummy","digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/dummy",digitizer,digitizer->GetSD()),

	  - !!!! DO NOT FORGET TO WRITE A SHORT EXPLANATION ON WHAT DOES YOUR DM !!!!
	  - Comment everything inside Digitize() method
   8) Compile 2ed time
   9) In case of adaptation of old existing class:
   	  - Copy everything inside ProcessOnedigi() and ProcessdigiList() from old module to Digitize() (places where to copy are indicated in this class)
   	  - Replace:
   	  	  inputDigi -> inputDigi
	      m_outputDigi -> m_outputDigi + correct the first declaration (as in this Dummy module)
	      m_outputDigiList.push_back(m_outputDigi) ->  m_OutputDigiCollection->insert(m_outputDigi);
	10) Add YourDigitizerModule to GateCoincidenceDigitizer.cc
			- #include "YourDigitizerModule.hh"
			- in DumpMap() method in
				static G4String theList = " ...."
			- in DoInsertion() :
				  else if (childTypeName=="yourDM")
     	 	 	 	 {
   	  	  	  	  	  newDM = new GateYourDigitizerModule(m_digitizer);
   	  	  	  	  	  m_digitizer->AddNewModule(newDM);
     	 	 	 	 }
	11) Compile 3ed time and execute

	*/

#include "GateCoincidenceDeadTime.hh"
#include "GateCoincidenceDeadTimeMessenger.hh"
#include "GateCoincidenceDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateCoincidenceDeadTime::GateCoincidenceDeadTime(GateCoincidenceDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/CoincidenceDigitizer/"+digitizer->m_digitizerName+"/"+name, digitizer),
   m_isParalysable (false),
   m_deadTime (0),
   m_rebirthTime(0),
   m_bufferCurrentSize(0),
   m_bufferSize(0),
   m_oldEv1(-1),
   m_oldEv2(-1),
   m_oldName(""),
   m_conserveAllEvent(true),
   m_wasTaken(false),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateCoincidenceDeadTimeMessenger(this);
}


GateCoincidenceDeadTime::~GateCoincidenceDeadTime()
{
  delete m_Messenger;

}


void GateCoincidenceDeadTime::Digitize()
{

	//G4cout<<" GateCoincidenceDeadTime::Digitize "<< m_conserveAllEvent<<G4endl;

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateCoincidenceDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();

	//DigiMan->List();
	//G4cout<<m_DCID <<G4endl;


	GateCoincidenceDigiCollection* IDC = 0;
	IDC = (GateCoincidenceDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateCoincidenceDigi* inputDigi = new GateCoincidenceDigi();

	std::vector< GateCoincidenceDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateCoincidenceDigi*>::iterator iter;

/*
	 if (nVerboseLevel==1)
			    {
			    	G4cout << "[ GateCoincidenceDeadTime::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n";
			    	for (size_t k=0; k<m_OutputDigiCollection->entries();k++)
			    		G4cout << *(*IDC)[k] << Gateendl;
			    		G4cout << Gateendl;
			    }
	*/


	//m_bufferSize=0;
	//m_bufferCurrentSize=0;

	//G4cout<<m_bufferSize<<" "<<m_rebirthTime<<" "<<m_bufferCurrentSize<<G4endl;

		//G4cout<<IDC->entries()<<G4endl;


  if (IDC)
     {
	  G4int n_digi = IDC->entries();
	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  if (!inputDigi || inputDigi->size()!=2) {
		        if (nVerboseLevel>1)
		        	G4cout << "[GateCoincidenceDeadTime::Digitize]: input digi was null -> nothing to do\n\n";
		        continue;
		    }

		   unsigned long long int  currentTime = (unsigned long long int)(inputDigi->GetEndTime()/picosecond);

		   if (m_conserveAllEvent){
		        if (i==0) {
		        	m_oldEv1 = (*inputDigi)[0]->GetEventID();
		        	m_oldEv2 = (*inputDigi)[1]->GetEventID();
		        } else {

		            if (m_wasTaken && m_bufferSize>0) m_bufferCurrentSize++;

		            if (m_wasTaken)
		               {

		            	m_outputDigi = new GateCoincidenceDigi(*inputDigi);
		            	m_OutputDigiCollection->insert(m_outputDigi);
		            	continue;

		               }
		               else

		            	continue;
		       }
		      }


		    // FIND TIME OF digi
		      if (nVerboseLevel>5){
		          G4cout << "A new digi is processed by dead time time : " << (inputDigi->GetEndTime())/picosecond
		    	     << " =  "<< currentTime  << Gateendl  ;
		          G4cout << "Rebirth time is " << m_rebirthTime << Gateendl ;

		      }

		      // IS DETECTOR DEAD ?
		      G4bool outputPulseFlag=0;
		   if (currentTime >=  m_rebirthTime) {

		          // NO DETECTOR IS NOT DEAD : COPY THIS digi TO OUTPUT digi

		    	  m_outputDigi = new GateCoincidenceDigi(*inputDigi);
			      m_OutputDigiCollection->insert(m_outputDigi);
			      outputPulseFlag=1;

			      if (m_bufferSize>0){
		        	  m_bufferCurrentSize++;
		        	  if (m_bufferCurrentSize>=m_bufferSize)
		        	  {
		        		  m_rebirthTime = currentTime + m_deadTime;
		        		  m_bufferCurrentSize=0;
		        	  }
		          } else {
		        	  m_rebirthTime = currentTime + m_deadTime;
		          	  }

		          if (nVerboseLevel>5){
		        	  G4cout << "We have accept " << currentTime << " a digi "
		        			  <<"\trebirth time\t" << m_rebirthTime << Gateendl;
		        	  G4cout << "Copied digi to output:\n"
		        			  << &m_outputDigi << Gateendl << Gateendl ;
		          	  }
		      } else {

		          // YES DETECTOR IS DEAD : MAY BE REMOVE digi
		          	  if ((m_bufferSize>0) && (m_bufferMode==1))
		          	  {

		          		  if (m_bufferCurrentSize<m_bufferSize)
		          		  {

		          			  m_bufferCurrentSize++;

		          			  m_outputDigi= new GateCoincidenceDigi(*inputDigi);
		        		      m_OutputDigiCollection->insert(m_outputDigi);
		        		      outputPulseFlag=1;


		          			  if (nVerboseLevel>5){
		          				  G4cout << "We have accept " << currentTime << " a digi "
		          						  <<"\trebirth time\t" << m_rebirthTime << Gateendl;
		          				  //G4cout << "Copied digi to output:\n"
		          				//		  << *m_outputDigi << Gateendl << Gateendl ;
		          			  }

		          			  if (m_isParalysable && (m_bufferCurrentSize==m_bufferSize)){

		          				  m_rebirthTime  = currentTime + m_deadTime;
		          			  	  }
		          		  }
		          		  else {

		          	    	if (nVerboseLevel>5)
		          	    		G4cout << "Removed digi, due to dead time.\n";
		          	    	outputPulseFlag=0;

		            		}
		          }
		          else
		          {

		        	  // AND IF "PARALYSABLE" DEAD TIME, MAKE THE DEATH OF DETECTOR LONGER
		        	  if (m_isParalysable && (m_bufferSize<1)){
		        		  m_rebirthTime  = currentTime + m_deadTime;
		        	  }
		        	  if (nVerboseLevel>5)
		        		  G4cout << "Removed digi, due to dead time.\n";
		        	  outputPulseFlag=0;
		          }
		      }


		      if (m_conserveAllEvent && (i==0))
		      {
		    	  m_wasTaken = (outputPulseFlag!=0);
		      }

		      if (nVerboseLevel>99)
		        getchar();



	if (nVerboseLevel==1) {
			G4cout << "[GateCoincidenceDeadTime::Digitize]: returning output digi-list with " << OutputDigiCollectionVector->size() << " entries\n";
			for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
				G4cout << *iter << Gateendl;
			G4cout << Gateendl;

		}
	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateCoincidenceDeadTime::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}


void GateCoincidenceDeadTime::SetDeadTimeMode(G4String val)
{
  if((val!="paralysable")&&(val!="nonparalysable"))
    G4cout << "*** GateCoincidenceDeadTimeOld.cc : Wrong dead time mode : candidates are : paralysable nonparalysable\n";
  else
   m_isParalysable = (val=="paralysable");

}

void GateCoincidenceDeadTime::DescribeMyself(size_t indent )
{
	  G4cout << GateTools::Indent(indent) << "DeadTime: " << G4BestUnit(m_deadTime,"Time") << Gateendl; ;
}
