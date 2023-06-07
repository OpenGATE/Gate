
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
// OK GND 2022
/*!
  \class  GateAdderCompton

      \brief Digitizer Module for adding/grouping digis per volume.

    - GatedigiAdder - by Daniel.Strul@iphe.unil.ch
	- Exact Compton kinematics changes by jbmichaud@videotron.ca

    - For each volume where there was one or more input digi, we get exactly
      one output digi, whose energy is the sum of all the input-digi energies,
      and whose position is the centroid of the photonic input-digi positions.
	  Electronic digis energy is assigned to the proper photonic digi (the last digi encountered).
	  Wandering photo-electron are discarded, i.e. when no previous photonic interaction
	  has occurred inside the volume ID. This is not EXACT.
	  The case when a photoelectron wanders and lands into a volume ID where

      OK: added to GND in Jan2023
*/


#include "GateAdderCompton.hh"
#include "GateAdderComptonMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"
#include "G4Electron.hh"


GateAdderCompton::GateAdderCompton(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateAdderComptonMessenger(this);
}


GateAdderCompton::~GateAdderCompton()
{
  delete m_Messenger;

}


void GateAdderCompton::Digitize()
{

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();

	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  if ( OutputDigiCollectionVector->empty() )
		  		{
		  			//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " First" ;
		  			if ( inputDigi->GetPDGEncoding() == ( G4Electron::Electron()->GetPDGEncoding() ) )
		  			{
		  				//G4cout <<  " Different Discarded\n";

		  				if (nVerboseLevel>1)
		  					G4cout << "Discarded electronic digi for volume " << inputDigi->GetVolumeID()
		  					<< " with no previous photonic interaction in that Volume ID\n" << Gateendl ;
		  			}
		  			else
		  			{
		  				m_outputDigi = new GateDigi(*inputDigi);
		  					if (nVerboseLevel>1)
		  						G4cout << "Created new digi for volume " << inputDigi->GetVolumeID() << ".\n"
		  						<< "Resulting digi is: \n"
		  						<< *m_outputDigi << Gateendl << Gateendl ;
		  					 m_OutputDigiCollection->insert(m_outputDigi);

		  					//G4cout << " Different Push Back\n";
		  			}
		  		}
		  		else
		  		{
		  			std::vector<GateDigi*>::reverse_iterator currentiter = OutputDigiCollectionVector->rbegin();

		  			while (1)
		  			{
		  				if ( inputDigi->GetPDGEncoding() == ( G4Electron::Electron()->GetPDGEncoding() ) )
		  				{
		  					if ( (inputDigi->GetVolumeID() == (*currentiter)->GetVolumeID()) && (inputDigi->GetEventID() == (*currentiter)->GetEventID()) )
		  					{
		  						CentroidMergeCompton(inputDigi,*currentiter);

		  						if (nVerboseLevel>1)
		  							G4cout << "Merged past photonic digi for volume " << inputDigi->GetVolumeID()
		  							<< " with new electronic digi of energy " << G4BestUnit(inputDigi->GetEnergy(),"Energy") <<".\n"
		  							<< "Resulting digi is: \n"
		  							<< (*currentiter) << Gateendl << Gateendl ;
		  						//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " Merged\n";
		  						break;
		  					}
		  					else
		  					{
		  						//G4cout <<  "Increment " ;
		  						currentiter++;
		  						if (currentiter == OutputDigiCollectionVector->rend())
		  						{
		  							//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " End of list\n";
		  							break;
		  						}
		  					}
		  				}
		  				else
		  				{
		  					//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " Push back\n";
		  					m_outputDigi = new GateDigi(*inputDigi);
		  						if (nVerboseLevel>1)
		  							G4cout << "Created new digi for volume " << inputDigi->GetVolumeID() << ".\n"
		  							<< "Resulting digi is: \n"
		  							<< *m_outputDigi << Gateendl << Gateendl ;
		  						 m_OutputDigiCollection->insert(m_outputDigi);

		  					break;
		  				}

		  			}

		  		}

	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateAdderCompton::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}


GateDigi* GateAdderCompton::CentroidMergeCompton(GateDigi *right, GateDigi *output)
{
    // We define below the fields of the merged pulse

    // runID: identical for both pulses, nothing to do
    // eventID: identical for both pulses, nothing to do
    // sourceID: identical for both pulses, nothing to do
    // source-position: identical for both pulses, nothing to do

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
    output->m_energyIniTrack=-1;         // Initial energy of the track
    output->m_energyFin=-1;

    // time: store the minimum time
    output->m_time = std::min ( output->m_time , right->m_time ) ;

    // energy: we compute the sum
    G4double totalEnergy = output->m_energy + right->m_energy;

    // Local and global positions: keep the original Position

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




void GateAdderCompton::DescribeMyself(size_t )
{
  ;
}
