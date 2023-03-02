
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateNoise

  This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).

   - Create your DM by coping this class and GateDummyDigitizerMessenger class for your DM messenger
   - Places to change are marked with // ****** comment and called with "dummy" names
   - Include your module to GateSinglesDigitizerMessenger in the method DoInsertion(..)

	If you adapting some already exiting class from Old Gate Digitizer here is some of the tips
	- Digitize () is a fusion of GateVPulseProcessor::ProcessPulseList and GateXXX::ProcessOnePulse
	- pulse --> Digi
	- outputPulseList --> OutputDigiCollectionVector
	- inputPulse-->inputDigi
	- outputPulse --> m_outputDigi
	- how to adapt iterators check GateAdder class

  To create new Digitizer Module (DM), please, follow the steps:
  1) Copy .cc and .hh of GateNoise, GateNoiseMessenger to GateYourNewDigitizerModule and GateYourNewDigitizerModuleMessenger
  2) Replace in these new files : Noise -> YourNewDigitizerModule
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
   	  - Copy everything inside ProcessOnePulse() and ProcessPulseList() from old module to Digitize() (places where to copy are indicated in this class)
   	  - Replace:
   	  	  inputPulse -> inputDigi
	      outputPulse -> m_outputDigi + correct the first declaration (as in this Dummy module)
	      outputPulseList.push_back(outputPulse) ->  m_OutputDigiCollection->insert(m_outputDigi);
	10) Add YourDigitizerModule to GateSinglesDigitizer.cc
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

#include "GateNoise.hh"
#include "GateNoiseMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include "GateSystemListManager.hh"
#include "GateVDistribution.hh"

GateNoise::GateNoise(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_deltaTDistrib(),
   m_energyDistrib(),
   m_createdDigis(),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateNoiseMessenger(this);
	m_oldTime = -1;
}


GateNoise::~GateNoise()
{
  delete m_Messenger;

}


void GateNoise::Digitize()
{
	G4double shootRND_time = m_deltaTDistrib->ShootRandom();

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi = new GateDigi();

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

		    if (!m_energyDistrib){
		      G4cerr<<"GateNoise::ProcessPulseList : no energy distribution given. Nothing's done\n";
		      //return GateVPulseProcessor::ProcessPulseList(inputPulseList);
		    }

		    if (!m_deltaTDistrib){
		      G4cerr<<"GateNoise::ProcessPulseList : no deltaT distribution given. Nothing's done\n";
		      //return GateVPulseProcessor::ProcessPulseList(inputPulseList);
		    }
		    if (n_digi==0) return;


		    G4double t0;
		    if (m_oldTime<0) t0 = ComputeStartTime(IDC);
		    else t0=m_oldTime;

		    m_oldTime = ComputeFinishTime(IDC);


		   while(t0+=shootRND_time,t0<m_oldTime)
		    {
		    	GateDigi* digi = new GateDigi();
		    	digi->SetTime(t0);
		    	digi->SetEnergy(m_energyDistrib->ShootRandom());

		    	// now define a random outputVolumeID...
		    	GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);
		    	size_t depth = system->GetTreeDepth();
		    	GateOutputVolumeID outputVol(depth);
		    	//      G4cout<<"Choosing ";
		    	for (size_t i=0;i<depth;i++)
		    	{
		    		size_t max = system->ComputeNofElementsAtLevel(i);
		    		long n = CLHEP::RandFlat::shootInt((long int)0,(long int)(max));
		    		outputVol[i]=n;
		    		//          G4cout<<n<<' ';
		    	}
		    	//      G4cout<< Gateendl;
		    	digi->SetOutputVolumeID(outputVol);

		    	GateVolumeID* volID = system->MakeVolumeID(outputVol);
		    	digi->SetVolumeID(*volID);
		    	digi->SetGlobalPos(system->ComputeObjectCenter(volID));
		    	delete volID;

		    	digi->SetEventID(-2);

		    	m_createdDigis.push_back(digi);
		    }

			  //loop over input digits
		    for (G4int i=0;i<n_digi;i++)
		   	  {
				  inputDigi=(*IDC)[i];
				  if (!m_createdDigis.empty())
				  {

					  std::vector<GateDigi*>::iterator it = m_createdDigis.begin();
					  while (it != m_createdDigis.end() && (*it)->GetTime()<=inputDigi->GetTime())
					  {
						m_OutputDigiCollection->insert(*it);
						++it;
					  }
					  m_createdDigis.erase(m_createdDigis.begin(),it);
				  }

			m_OutputDigiCollection->insert(new GateDigi(*inputDigi));

		   	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateNoise::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}


// Return the min-time of all pulses
G4double GateNoise::ComputeStartTime(GateDigiCollection* IDC)
{

  	GateDigi* digi = new GateDigi();

  	std::vector< GateDigi* >* IDCVector = IDC->GetVector ();
  	std::vector<GateDigi*>::iterator iter;

  	G4double startTime = DBL_MAX;
  	digi=0;
  	for (iter = IDCVector->begin(); iter < IDCVector->end() ; ++iter) {
  		if ( (*iter)->GetTime() < startTime ){
  			startTime  = (*iter)->GetTime();
  			digi = *iter;
  		}
  	}

      return digi? digi->GetTime() : DBL_MAX;
}

// Return the max-time of all pulses
G4double GateNoise::ComputeFinishTime(GateDigiCollection* IDC)
{
	std::vector< GateDigi* >* IDCVector = IDC->GetVector ();
    std::vector<GateDigi*>::iterator iter;

    G4double finishTime = 0;
    for (iter = IDCVector->begin(); iter < IDCVector->end() ; ++iter) {
    	if ( (*iter)->GetTime() > finishTime ){
    		finishTime  = (*iter)->GetTime();
      			}
      	}


    return finishTime;
}


void GateNoise::DescribeMyself(size_t indent )
{
  ;
}
