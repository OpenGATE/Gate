
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateAdderComptPhotIdeal

   Last modification (Adaptation to GND): July 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
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

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  flgEvtRej=0;

		  if(m_flgRejActPolicy==1 && flgEvtRej==1){
			  while (OutputDigiCollectionVector->size()) {
				  delete OutputDigiCollectionVector->back();
				  OutputDigiCollectionVector->erase(OutputDigiCollectionVector->end()-1);
			  }
		}


		if (nVerboseLevel==1) {
			G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
			for (iter = OutputDigiCollectionVector->begin() ; iter != OutputDigiCollectionVector->end() ; ++iter)
			  G4cout << **iter << Gateendl;
			G4cout << Gateendl;
		}


#ifdef GATE_USE_OPTICAL
    // ignore pulses based on optical photons. These can be added using the opticaladder
    if (!inputDigi->IsOptical())
#endif
    {

    	GateDigi* m_outputDigi = new GateDigi(*inputDigi);
        if(inputDigi->GetParentID()==0)
        {
            if(inputDigi->GetPostStepProcess()=="compt" ||inputDigi->GetPostStepProcess()=="phot" ||inputDigi->GetPostStepProcess()=="conv"  ){

            	if(inputDigi->GetPostStepProcess()=="conv"){
                    flgEvtRej=1;
                }
                m_OutputDigiCollection->insert(m_outputDigi);
                lastTrackID.push_back(inputDigi->GetTrackID());
            }
            else{

                if(inputDigi->GetPostStepProcess()!="Transportation" )
                	flgEvtRej=1;
                delete m_outputDigi;
            }
            //. The Eini of the primaries is her Eini before the first interaction in SD of the layers, not the Eini of the track.. This has been changed in the comptoncameraactor in where the hit coletion is saved

        }
        else{
            if(OutputDigiCollectionVector->empty())
            {
            	;
            }
            else{
                //Initial simplification we only take the electron processes.
            	std::vector<GateDigi*>::reverse_iterator iter = OutputDigiCollectionVector->rbegin();
                while (1){

                    //This could be saved because once I have saved the primaries EmaxDepos is fixed.
                   double EmaxDepos;

                   if(iter==(OutputDigiCollectionVector->rend()-1)){
                       EmaxDepos=(*iter)->GetEnergyIniTrack()- (*iter)->GetEnergyFin();
                   }
                   else{
                       EmaxDepos=(*(iter+1))->GetEnergyFin()- (*iter)->GetEnergyFin();
                   }

                    if ( (inputDigi->GetVolumeID() == (*iter)->GetVolumeID()) && (inputDigi->GetEventID() == (*iter)->GetEventID()) && inputDigi->GetEnergyIniTrack()<=(EmaxDepos+epsilonEnergy))
                    {

                        //first order secondaries
                        if(inputDigi->GetParentID()==1 && inputDigi->GetProcessCreator()==(*iter)->GetPostStepProcess()){
                            if( (*iter)->GetTrackID()==1){
                                 //first secondary for the pulse

                                //---Final paste
                                //Look if there is an output pulse already with the trackId of the inputDigi
                                bool isgood=true;
                                std::vector<GateDigi*>::iterator iterIntern;
                                for (iterIntern = OutputDigiCollectionVector->begin() ; iterIntern != OutputDigiCollectionVector->end() ; ++iterIntern ){
                                     if(inputDigi->GetTrackID()==(*iterIntern)->GetTrackID()){
                                         isgood=false;
                                     }
                                }
                                if(isgood==true){
                                 CentroidMerge(inputDigi,m_outputDigi);
                                 (*iter)->SetTrackID(inputDigi->GetTrackID());

                                }
                                //---Final paste end

                                break;
                            }
                            else{
                                //This works for EMlivermore but not for stantdard.
                                if((*iter)->GetTrackID()==inputDigi->GetTrackID()){
                                    //first secondary for the pulse
                                   CentroidMerge(inputDigi,m_outputDigi);;
                                   break;
                                }
                                else{

                                    ++iter;
                                    if (iter == OutputDigiCollectionVector->rend())
                                    {
                                        break;
                                    }
                                }
                            }

                        }
                        else if(inputDigi->GetParentID()>1 && (*iter)->GetTrackID()>=2 && inputDigi->GetParentID()>=(*iter)->GetTrackID()){//rest of secondaries
                            //sECOND CONDITION to avoid to sum to a primary secondaries created in another layer For example a bremstralung that escape
                            CentroidMerge(inputDigi,m_outputDigi);
                            //Any more information 30548
                            //Search for good condition !!! (think based on processcreator parentID trackID)
                            break;
                        }
                        else{
                            //firs secondary without  process creator corresponding to  the saved one. Keep looking
                            ++iter;
                            if (iter == OutputDigiCollectionVector->rend())
                            {
                                break;
                            }

                        }

                    }
                    else{
                        ++iter;

                        if (iter == OutputDigiCollectionVector->rend())
                        {
                            break;
                        }
                    }
                }
            }
            delete m_outputDigi;
        }
    }


	if (nVerboseLevel==1)
	 {
		G4cout << "[GateAdderComptPhotIdeal::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
		for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
			G4cout << **iter << Gateendl;
		G4cout << Gateendl;
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
