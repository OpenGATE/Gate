
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateNoise
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

	GateDigi* inputDigi;

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
