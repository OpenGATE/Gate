
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateDummyDigitizerModule

  This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).

   - Create your DM by coping this class and GateDummyDigitizerMessenger class for your DM messenger
   - Places to change are marked with // ****** comment and called with "dummy" names
   - Include your module to GateSinglesDigitizerMessenger.cc in the method DoInsertion(..)

	If you adapting some already exiting class from Old Gate Digitizer here is some of the tips
	- Digitize () is a fusion of GateVPulseProcessor::ProcessPulseList and GateXXX::ProcessOnePulse
	- pulse --> Digi
	- outputPulseList --> OutputDigiCollectionVector
	- inputPulse-->inputDigi
	- outputPulse --> m_outputDigi
	- how to adapt iterators check GateAdder class

  To create new Digitizer Module (DM), please, follow the steps:
  1) Copy .cc and .hh of GateDummyDigitizerModule, GateDummyDigitizerModuleMessenger to GateYourNewDigitizerModule and GateYourNewDigitizerModuleMessenger
  2) Replace in these new files : DummyDigitizerModule -> YourNewDigitizerModule
  3) Compile 1st time (so not forget to redo ccmake too)
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
   	  - Copy everything inside ProcessOnePulse() and ProcessPulseList() from old module to Digitize() (places where to copy are indicated in this class)
   	  - Replace:
   	  	  inputPulse -> inputDigi
	      outputPulse -> m_outputDigi + correct the first declaration (as in this Dummy module)
	      outputPulseList.push_back(outputPulse) ->  m_OutputDigiCollection->insert(m_outputDigi);
	10) Add YourDigitizerModule to GateSinglesDigitizerMessenger
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

#include "GateDummyDigitizerModule.hh"
#include "GateDummyDigitizerModuleMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateDummyDigitizerModule::GateDummyDigitizerModule(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_parameter("dummy"),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateDummyDigitizerModuleMessenger(this);
}


GateDummyDigitizerModule::~GateDummyDigitizerModule()
{
  delete m_Messenger;

}


void GateDummyDigitizerModule::Digitize()
{

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;

/*
	 if (nVerboseLevel==1)
			    {
			    	G4cout << "[ GateDummyDigitizerModule::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n";
			    	for (size_t k=0; k<m_OutputDigiCollection->entries();k++)
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
		  // ***** the following part of the code to adapt
		  /// *** This part is from ProcessPulseList

		  ////// ** This part is from ProcessOnePulse
		     for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
		     {
		    	 if ( (*iter)->GetVolumeID()   == inputDigi->GetVolumeID() )
		    	 {
		    		 if(m_parameter=="dummy"){
		                 DummyMethod1(inputDigi);
		    		 }
		    		 else{
		    			 DummyMethod2( inputDigi );
		    		 }

		      if (nVerboseLevel>1)
		      {
		    	 G4cout << "Merged previous pulse for volume " << inputDigi->GetVolumeID()
		 		 << " with new pulse of energy " << G4BestUnit(inputDigi->GetEnergy(),"Energy") <<".\n"
		 		 << "Resulting pulse is: \n"
		 		 << **iter << Gateendl << Gateendl ;
		      }
		 	  break;
		       }
		     }

		     if ( iter == OutputDigiCollectionVector->end() )
		     {
		       m_outputDigi = new GateDigi(*inputDigi);
		       m_outputDigi->SetEnergyIniTrack(-1);
		       m_outputDigi->SetEnergyFin(-1);
		       if (nVerboseLevel>1)
		 	  G4cout << "Created new pulse for volume " << inputDigi->GetVolumeID() << ".\n"
		 		 << "Resulting pulse is: \n"
		 		 << *m_outputDigi << Gateendl << Gateendl ;
		      /// !!!!!! The following line should be kept !!!! -> inserts the outputdigi to collection
		       m_OutputDigiCollection->insert(m_outputDigi);

		     }
		 ////// ** End of the part from ProcessOnePulse


		if (nVerboseLevel==1) {
			G4cout << "[GateDummyDigitizerModule::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
			for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
				G4cout << **iter << Gateendl;
			G4cout << Gateendl;
		}
	/// *** End of the part from ProcessPulseList
	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateDummyDigitizerModule::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}


void GateDummyDigitizerModule::SetDummyParameter(const G4String &param)
{
	m_parameter=param;
}


///////////////////////////////////////////
////////////// Methods of DM //////////////
///////////////////////////////////////////

void GateDummyDigitizerModule::DummyMethod1(GateDigi *right)
{
	//to copy all variables that are not changed
	m_outputDigi=right;


}

void GateDummyDigitizerModule::DummyMethod2(GateDigi *right)
{
	//to copy all variables that are not changed
	m_outputDigi=right;


}


void GateDummyDigitizerModule::DescribeMyself(size_t indent )
{
  ;
}
