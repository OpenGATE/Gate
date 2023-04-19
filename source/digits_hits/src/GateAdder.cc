
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

// OK GND 2022
/*!
  \class  GateAdder (prev. GatePulseAdder - by Daniel.Strul@iphe.unil.ch)
  \brief   for adding/grouping pulses per volume.

    - For each volume where there was one or more input pulse, we get exactly
      one output pulse, whose energy is the sum of all the input-pulse energies,
      and whose position is the centroid of the input-pulse positions.

*/

#include "GateAdder.hh"
#include "GateAdderMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"


GateAdder::GateAdder(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer, digitizer->GetSD()),
   m_positionPolicy(kEnergyCentroid),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
{
	G4String colName = digitizer->GetOutputName();
	collectionName.push_back(colName);
	m_Messenger = new GateAdderMessenger(this);
}


GateAdder::~GateAdder()
{
  delete m_Messenger;
}


void GateAdder::Digitize()
{
	if (nVerboseLevel>1)
		G4cout<<"[GateAdder::Digitize] for "<< m_digitizer->GetOutputName() <<G4endl;

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();

	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi = new GateDigi();

	std::vector< GateDigi* >* m_OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;


  if (IDC)
     {
	  if (nVerboseLevel>1)
		G4cout<<"[GateAdder::Digitize] Number of input digis "<< IDC->entries() <<G4endl;

	  G4int n_digi = IDC->entries();
	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  //This part is from ProcessOnePulse
#ifdef GATE_USE_OPTICAL
		  // ignore pulses based on optical photons. These can be added using the opticaladder
		  if (!inputDigi->IsOptical())
#endif
		     for (iter=m_OutputDigiCollectionVector->begin(); iter!= m_OutputDigiCollectionVector->end() ; ++iter)
		     {
		    	 if ( (*iter)->GetVolumeID()   == inputDigi->GetVolumeID() )
		    	 {
		    		 if(m_positionPolicy==kEnergyWinner){
		    			 m_outputDigi = MergePositionEnergyWin(inputDigi,*iter);
		    		 }
		    		 else{
		    			 m_outputDigi = CentroidMerge( inputDigi,*iter );
		    		 }

		    		 if (nVerboseLevel>1)
		    		 	 {
		    			 	 G4cout << " [GateAdder::Digitize] Merged previous digi for volume " << inputDigi->GetVolumeID()
		 						 << " with new digi of energy " << G4BestUnit(inputDigi->GetEnergy(),"Energy") <<".\n"
								 << "Resulting digi is: \n"
								 << **iter << Gateendl << Gateendl ;
		    		 	 }
		    		 break;
		    	 }

		     }//loop over iter

		     if ( iter == m_OutputDigiCollectionVector->end() )
		     {
		    	 m_outputDigi = new GateDigi(*inputDigi);
		    	 m_outputDigi->SetEnergyIniTrack(-1);
		    	 m_outputDigi->SetEnergyFin(-1);
		    	 if (nVerboseLevel>1)
		    		 G4cout << "[GateAdder::Digitize] Created new digi for volume " << inputDigi->GetVolumeID() << ".\n"
					 << "Resulting digi is: \n"
					 << *m_outputDigi << Gateendl << Gateendl ;
		    	 m_OutputDigiCollection->insert(m_outputDigi);
		     }

		//This part is from ProcessPulseList
		     if (nVerboseLevel==1) {
		    	 G4cout << "[GateAdder::Digitizer]: returning output digi collection with " << m_OutputDigiCollectionVector->size() << " entries\n";
		    	 for (iter=m_OutputDigiCollectionVector->begin(); iter!= m_OutputDigiCollectionVector->end() ; ++iter)
		    		 G4cout << **iter << Gateendl;
		    	 G4cout << Gateendl;
		     }
	  }
     }
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateAdder::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);
}


void GateAdder::SetPositionPolicy(const G4String &policy)
{
	if (policy=="takeEnergyWinner")
    {
		m_positionPolicy=kEnergyWinner;
    }
    else {
        if (policy!="energyWeightedCentroid")
            G4cout<<"WARNING : policy not recognized, using default :energyWeightedCentroid\n";
       m_positionPolicy=kEnergyCentroid;
    }


}


///////////////////////////////////////////
////////////// Methods of DM //////////////
///////////////////////////////////////////


/*GateDigi* GateAdder::CentroidMerge(GateDigi* right, GateDigi* output )
{

    // AE : Added in a real pulse no sense
    output->m_Postprocess="NULL";         // PostStep process
    output->m_energyIniTrack=-1;         // Initial energy of the track
    output->m_energyFin=-1;         // final energy of the particle
    output->m_processCreator="NULL";
    output->m_trackID=0;
    //-----------------

    // time: store the minimum time
    output->m_time = std::min ( output->m_time , right->m_time ) ;

    // energy: we compute the sum, but do not store it yet
    // (storing it now would mess up the centroid computations)
    G4double totalEnergy = output->m_energy + right->m_energy;

    if (output->m_sourceEnergy != right->m_sourceEnergy) output->m_sourceEnergy=-1;
    if (output->m_sourcePDG != right->m_sourcePDG) output->m_sourcePDG=0;
    if ( right->m_nCrystalConv > output->m_nCrystalConv ){
        output->m_nCrystalConv 	= right->m_nCrystalConv;
    }
    if ( right->m_nCrystalCompton > output->m_nCrystalCompton ){
        output->m_nCrystalCompton 	= right->m_nCrystalCompton;
    }
    if ( right->m_nCrystalRayleigh > output->m_nCrystalRayleigh ){
        output->m_nCrystalRayleigh 	= right->m_nCrystalRayleigh;
    }

    // Local and global positions: store the controids
    if(totalEnergy>0){
        output->m_localPos  =  ( output->m_localPos  * output->m_energy  + right->m_localPos  * right->m_energy ) / totalEnergy ;
        output->m_globalPos =  ( output->m_globalPos * output->m_energy  + right->m_globalPos * right->m_energy ) / totalEnergy ;
    }
    else{
        output->m_localPos  =  ( output->m_localPos  + right->m_localPos)/2;
        output->m_globalPos =  ( output->m_globalPos + right->m_globalPos)/2 ;
    }

    // Now that the centroids are stored, we can store the energy
    output->m_energy   = totalEnergy;


    // # of compton process: store the max nb
    if ( right->m_nPhantomCompton > output->m_nPhantomCompton )
    {
        output->m_nPhantomCompton 	= right->m_nPhantomCompton;
        output->m_comptonVolumeName = right->m_comptonVolumeName;
    }

    // # of Rayleigh process: store the max nb
    if ( right->m_nPhantomRayleigh > output->m_nPhantomRayleigh )
    {
        output->m_nPhantomRayleigh 	= right->m_nPhantomRayleigh;
        output->m_RayleighVolumeName = right->m_RayleighVolumeName;
    }

    // HDS : # of septal hits: store the max nb
    if ( right->m_nSeptal > output->m_nSeptal )
    {
        output->m_nSeptal 	= right->m_nSeptal;
    }

    // VolumeID: should be identical for both pulses, we do nothing
    // m_scannerPos: identical for both pulses, nothing to do
    // m_scannerRotAngle: identical for both pulses, nothing to do
    // m_outputVolumeID: should be identical for both pulses, we do nothing
    return output;
}
*/
void GateAdder::DescribeMyself(size_t )
{
  ;
}
