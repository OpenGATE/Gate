

#include "GatePulseAdderComptPhotIdealLocal.hh"

#include "G4UnitsTable.hh"

#include "GatePulseAdderComptPhotIdealLocalMessenger.hh"
#include "G4Electron.hh"
#include "GateConfiguration.h"
#include "GateObjectStore.hh"

GatePulseAdderComptPhotIdealLocal::GatePulseAdderComptPhotIdealLocal(GatePulseProcessorChain* itsChain,
											 const G4String& itsName)
											 : GateVPulseProcessor(itsChain,itsName)
{

    m_messenger = new GatePulseAdderComptPhotIdealLocalMessenger(this);

}

GatePulseAdderComptPhotIdealLocal::~GatePulseAdderComptPhotIdealLocal()
{
	delete m_messenger;
}


G4int GatePulseAdderComptPhotIdealLocal::ChooseVolume(G4String val){
    GateObjectStore* m_store = GateObjectStore::GetInstance();
    if (m_store->FindCreator(val)!=0) {
      m_name=val;
      G4cout << "value inserted in chosse Volume "<<val<<G4endl;
      return 1;
    }
    else {
      G4cout << "Wrong Volume Name\n";
      return 0;
    }
}


GatePulseList* GatePulseAdderComptPhotIdealLocal::ProcessPulseList(const GatePulseList* inputPulseList)
{
  if (!inputPulseList)
    return 0;

  size_t n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
        G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
  if (!n_pulses)
    return 0;

  GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

  GatePulseConstIterator iter;




  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter)
        ProcessOnePulse( *iter, *outputPulseList);

  for(unsigned int k=0; k< primaryPulsesVol.size(); ++k){
            outputPulseList->at(indexPrimVInOut.at(k))->SetEnergy(primaryPulsesVol.at(k).GetEnergy());
            outputPulseList->at(indexPrimVInOut.at(k))->SetTime(primaryPulsesVol.at(k).GetTime());

  }
  while (primaryPulsesVol.size()) {
      primaryPulsesVol.erase(primaryPulsesVol.end()-1);
      indexPrimVInOut.erase(indexPrimVInOut.end()-1);
      indexPrimVInPrim.erase(indexPrimVInPrim.end()-1);
       EDepmaxPrimV.erase( EDepmaxPrimV.end()-1);
      if(primaryPulsesVol.size()!=indexPrimVInOut.size()) G4cout<<"[GatePulseAdderComptPhotIdealLocal:  problems size of vectors]"<<G4endl;
      if(primaryPulsesVol.size()!=indexPrimVInPrim.size()) G4cout<<"[GatePulseAdderComptPhotIdealLocal:  problems size of vectors (total primaries)]"<<G4endl;
  }
  while (primaryPulses.size()) {
      primaryPulses.erase(primaryPulses.end()-1);
  }
  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
        G4cout << **iter << Gateendl;
      G4cout << Gateendl;
  }

  return outputPulseList;
}


void GatePulseAdderComptPhotIdealLocal::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{



#ifdef GATE_USE_OPTICAL
    // ignore pulses based on optical photons. These can be added using the opticaladder
    if (!inputPulse->IsOptical())
#endif
    {
        if(inputPulse->GetParentID()==0)
        {
            if(inputPulse->GetPostStepProcess()=="compt" ||inputPulse->GetPostStepProcess()=="phot" || inputPulse->GetPostStepProcess()=="conv"  ){
                primaryPulses.push_back(*inputPulse);
                if(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName()==m_name){
                    indexPrimVInPrim.push_back( primaryPulses.size()-1);
                    if(primaryPulses.size()==1){
                       EDepmaxPrimV.push_back(inputPulse->GetEnergyIniTrack()-inputPulse->GetEnergyFin());
                    }
                    else{
                        EDepmaxPrimV.push_back(primaryPulses.at(primaryPulses.size()-2).GetEnergyFin()-inputPulse->GetEnergyFin());

                    }

                }

            }

        }

        // G4cout<<"name que uso "<<m_name<<G4endl;
        if(m_name==((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName()){

            //G4cout << " Ideal adder "<<outputPulseList.size()<<G4endl;


            if(inputPulse->GetParentID()==0)
            {
                if(inputPulse->GetPostStepProcess()=="compt" ||inputPulse->GetPostStepProcess()=="phot" ||inputPulse->GetPostStepProcess()=="conv"   ){
                    PulsePushBack(inputPulse, outputPulseList);
                    primaryPulsesVol.push_back(*inputPulse);
                    indexPrimVInOut.push_back(outputPulseList.size()-1);
                }
                //. La Eini de los primaries es su Eini antes de la primera interacci'on en SD of the layers no la Eini del track. This has been changed in the comptoncameraactor in where the hit coletion is saved


            }
            else{
                if(primaryPulsesVol.size()==0){

                    //G4cout << " problems  the first hit  en ese volumen is not a primary"<<G4endl;
                    // G4cout << " For exmaple primary interaction outside the layers"<<G4endl;
                    //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << G4endl;
                }
                else{
                    //Simplificacion  inicial solo cogemos los procesos de electrones
                    std::vector<GatePulse>::reverse_iterator iter = primaryPulsesVol.rbegin();
                    while (1){

                        if ( (inputPulse->GetVolumeID() == (*iter).GetVolumeID()) && (inputPulse->GetEventID() == (*iter).GetEventID())  && inputPulse->GetEnergyIniTrack()<=( EDepmaxPrimV.at((EDepmaxPrimV.size()-1- (iter-primaryPulsesVol.rbegin())))+epsilonEnergy) )
                        //if ( (inputPulse->GetVolumeID() == (*iter).GetVolumeID()) && (inputPulse->GetEventID() == (*iter).GetEventID()) )
                        {


                            //first order secondaries
                            if(inputPulse->GetParentID()==1 && inputPulse->GetProcessCreator()==(*iter).GetPostStepProcess()){
                                if( (*iter).GetTrackID()==1){
                                    //first secondary for the pulse



                                    //---Pegote final
                                    //Look if there is an output pulse already with the trackId of the inputPulse
                                    //For the cases in which we have two C interactions in thesame volume
                                    bool isgood=true;
                                    std::vector<GatePulse>::iterator iterIntern;
                                    for (iterIntern = primaryPulsesVol.begin() ; iterIntern !=primaryPulsesVol.end() ; ++iterIntern ){
                                        if(inputPulse->GetTrackID()==(*iterIntern).GetTrackID()){
                                            isgood=false;
                                        }
                                    }
                                    if(isgood==true){
                                        (*iter).CentroidMergeComptPhotIdeal(inputPulse);
                                        (*iter).SetTrackID(inputPulse->GetTrackID());
                                    }

                                    break;
                                }
                                else{
                                    //(*iter)->CentroidMergeComptPhotIdeal(inputPulse);;
                                    // break;
                                    //Esto funiiona para EMlivermore pero no para stantdard
                                    if((*iter).GetTrackID()==inputPulse->GetTrackID()){
                                        //first secondary for the pulse
                                        (*iter).CentroidMergeComptPhotIdeal(inputPulse);;
                                        break;
                                    }
                                    else{
                                        ++iter;
                                        if (iter == primaryPulsesVol.rend())
                                        {
                                            //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " End of list\n";
                                            break;
                                        }
                                    }


                                }

                            }
                            //else if(inputPulse->GetParentID()>1 && lastTrackID.at(posInp)>=2){//rest of secondaries
                            else if(inputPulse->GetParentID()>1 && (*iter).GetTrackID()>=2 && inputPulse->GetParentID()>=(*iter).GetTrackID()){//rest of secondaries
                                //sECOND CONDITION to avoid to summ to a primary secondaries created in another layer For example a bremstralung that escape
                                (*iter).CentroidMergeComptPhotIdeal(inputPulse);
                                //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " Merged\n";
                                //Alguna cndicion mas mirar 30548
                                //Buscar buena condicion !!! (pensar en base a processcreator parentID trackID)

                                break;

                            }
                            else{
                                //firs secondary without  process creator corresponding to  the saved one. Keep looking

                                ++iter;
                                if (iter == primaryPulsesVol.rend())
                                {
                                    //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " End of list\n";
                                    break;
                                }
                            }




                        }
                        else{
                            ++iter;
                            if (iter == primaryPulsesVol.rend())
                            {
                                //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " End of list\n";
                                break;
                            }

                        }

                    }


                }
            }
        }

        else{

            PulsePushBack(inputPulse, outputPulseList);
        }

    }


}




void GatePulseAdderComptPhotIdealLocal::DescribeMyself(size_t )
{
	;
}

//this is standalone only because it repeats twice in processOnePulse()
inline void GatePulseAdderComptPhotIdealLocal::PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList)
{
	GatePulse* outputPulse = new GatePulse(*inputPulse);
    if (nVerboseLevel>1)
		G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		<< "Resulting pulse is: \n"
		<< *outputPulse << Gateendl << Gateendl ;
	outputPulseList.push_back(outputPulse);
}


