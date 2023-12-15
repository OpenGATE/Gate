
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateAdderComptPhotIdeal

   Last modification (Adaptation to GND): July 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
 	 !! Important	Maybe not options are tested !!

*/

#include "GateAdderComptPhotIdeal.hh"
#include "GateAdderComptPhotIdealMessenger.hh"
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
#include "GateConfiguration.h"


GateAdderComptPhotIdeal::GateAdderComptPhotIdeal(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_GateAdderComptPhotIdeal(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
{
	m_flgRejActPolicy=0;
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateAdderComptPhotIdealMessenger(this);
}


GateAdderComptPhotIdeal::~GateAdderComptPhotIdeal()
{
  delete m_Messenger;
}


void GateAdderComptPhotIdeal::Digitize()
{

	//G4cout<<"GateAdderComptPhotIdeal::Digitize "<<G4endl;

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;
	std::vector<GateDigi*>::iterator iterIntern;


  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  m_outputDigi = new GateDigi(*inputDigi);

		  //m_OutputDigiCollection->insert(m_outputDigi);

#ifdef GATE_USE_OPTICAL
    // ignore pulses based on optical photons. These can be added using the opticaladder
    if (!inputDigi->IsOptical())
#endif
    	{
    	if(inputDigi->GetParentID()==0)
    	{
    	//G4cout<<inputDigi->GetPostStepProcess()<<G4endl;
    		if(inputDigi->GetPostStepProcess()=="compt" ||inputDigi->GetPostStepProcess()=="phot" ||inputDigi->GetPostStepProcess()=="conv"  )
    		{
    			if(inputDigi->GetPostStepProcess()=="conv"){
    				m_flgEvtRej=1;
    			}

    			if (nVerboseLevel>1)
    					G4cout << "Created new pulse for volume " << inputDigi->GetVolumeID() << ".\n"
    					<< "Resulting pulse is: \n"
    					<< *inputDigi << Gateendl << Gateendl ;

    			m_OutputDigiCollection->insert(m_outputDigi);

    		    m_lastTrackID.push_back(inputDigi->GetTrackID());
    			//G4cout << "inserting a pulse";
    			}
    		else
    		{
    			if(inputDigi->GetPostStepProcess()!="Transportation")
    				m_flgEvtRej=1;
    		}
    		//The Eini of the primaries is their Eini before the first interaction in SD of the layers, not the Eini of the track. This has been changed in the comptoncameraactor in where the hit collection is saved


    	}
    	else
    	{
    		if(m_OutputDigiCollection->entries()==0){
    			//G4cout << " problems  the first hit is not a primary"<<G4endl;
    			// G4cout << " For example primary interaction outside the layers"<<G4endl;
    			//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << G4endl;
    		}
    	    else
    	    {
    	    	//Initial simplification we only take the electron processes    	    	GatePulseList::reverse_iterator iter = outputPulseList.rbegin();
    	    	// GatePulseList::iterator iter = outputPulseList->begin();
    	    	//int posInp;
    	    	while (1){
    	    		//posInp=std::distance(outputPulseList->begin(),iter);
    	    		//if ( inputDigi->GetPDGEncoding() == ( G4Electron::Electron()->GetPDGEncoding() ) )
    	    		//{
    	    		//Esto podria guardarlo xq una vez que tengo los primarios guardados EmaxDepos es fijo
    	    		double EmaxDepos;
    	    		//if(iter==outputPulseList->begin()){
    	    		//EmaxDepos=(*iter)->GetEnergyIniTrack()- (*iter)->GetEnergyFin();
    	    		// }
    	    		//else{
    	    		//EmaxDepos=(*(iter-1))->GetEnergyFin()- (*iter)->GetEnergyFin();
    	    		// }

    	    		if(iter==(OutputDigiCollectionVector->end()-1)){
    	    			EmaxDepos=(*iter)->GetEnergyIniTrack()- (*iter)->GetEnergyFin();
    	    		}
    	    		else{
    	    			EmaxDepos=(*(iter+1))->GetEnergyFin()- (*iter)->GetEnergyFin();
    	    		}

    	    		if ( (inputDigi->GetVolumeID() == (*iter)->GetVolumeID()) && (inputDigi->GetEventID() == (*iter)->GetEventID()) && inputDigi->GetEnergyIniTrack()<=(EmaxDepos+epsilonEnergy))
    	    		{

    	    			//first order secondaries
    	    			if(inputDigi->GetParentID()==1 && inputDigi->GetProcessCreator()==(*iter)->GetPostStepProcess()){
    	    				if( (*iter)->GetTrackID()==1)
    	    				{
    	    					//first secondary for the pulse
    	    					//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " Merged\n";
    	    					// m_lastTrackID.at(posInp)=inputDigi->GetTrackID();
    	    					//add to fill it only if inputDigi->GetTrackID()

    	    					//---Pegote final ???
    	    					//Look if there is an output pulse already with the trackId of the inputDigi
    	    					bool isgood=true;

    	    					for (iterIntern = OutputDigiCollectionVector->begin() ; iterIntern != OutputDigiCollectionVector->end() ; ++iterIntern ){
    	    						if(inputDigi->GetTrackID()==(*iterIntern)->GetTrackID()){
    	    							isgood=false;
    	    						}
    	    					}
    	    					if(isgood==true){
    	    						CentroidMergeComptPhotIdeal(inputDigi,*iter);
    	    						(*iter)->SetTrackID(inputDigi->GetTrackID());
    	    					}
    	    					//---Pegote final fin

    	    					break;
    	    				}
    	    				else{
    	    					//(*iter)->CentroidMergeComptPhotIdeal(inputDigi);;
    	    					// break;
    	    					//This works for EN livermore but not for standard
    	    					if((*iter)->GetTrackID()==inputDigi->GetTrackID()){
    	    						//first secondary for the pulse
    	    						CentroidMergeComptPhotIdeal(inputDigi, *iter);
    	    						break;
    	    					}
    	    					else{

    	    						++iter;
    	    						if (iter == OutputDigiCollectionVector->end())
    	    						{
    	    							//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " End of list\n";
    	    							break;
    	    						}
    	    					}


    	    				}

    	    			}
    	    			//else if(inputDigi->GetParentID()>1 && m_lastTrackID.at(posInp)>=2){//rest of secondaries
    	    			else if(inputDigi->GetParentID()>1 && (*iter)->GetTrackID()>=2 && inputDigi->GetParentID()>=(*iter)->GetTrackID()){//rest of secondaries
    	    				//sECOND CONDITION to avoid to summ to a primary secondaries created in another layer For example a bremstralung that escape
    	    				CentroidMergeComptPhotIdeal(inputDigi, *iter);
    	    				//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " Merged\n";
    	    				//Any conditions but aim for 30548
    	    				//Look good condition!!! (think processcreator based on parentID trackID)

    	    				break;

    	    			}
    	    			else{
    	    				//First secondary without  process creator corresponding to  the saved one. Keep looking

    	    				++iter;
    	    					if (iter == OutputDigiCollectionVector->end())
    	    					{
    	    						//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " End of list\n";
    	    						break;
    	                         }
    	                    }

    	    		}
    	    		else{
    	    			++iter;
    	    			if (iter == OutputDigiCollectionVector->end())
    	    			{
    	    				//G4cout << inputDigi->GetEventID() << " " << inputDigi->GetOutputVolumeID() << " " << inputDigi->GetEnergy() << " " << inputDigi->GetPDGEncoding() << " End of list\n";
    	    				break;
    	    			}

    	    		}
    	    	}


    	    }
    	}





    	}
	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateAdderComptPhotIdeal::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;

    }
  StoreDigiCollection(m_OutputDigiCollection);

}

void GateAdderComptPhotIdeal::DescribeMyself(size_t indent )
{
  ;
}

GateDigi* GateAdderComptPhotIdeal::CentroidMergeComptPhotIdeal(GateDigi *right, GateDigi *output)
{

	// We define below the fields of the merged pulse

    // runID: identical for both pulses, nothing to do
    // eventID: identical for both pulses, nothing to do
    // sourceID: identical for both pulses, nothing to do
    // source-position: identical for both pulses, nothing to do

    // time: store the minimum time
	output->m_time = std::min ( output->m_time , right->m_time ) ;

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
    // energy: we compute the sum
    G4double totalEnergy = output->m_energy + right->m_energy;

    // Local and global positions: keep the original Position

    // n store the energy
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

