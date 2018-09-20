

#include "GatePulseAdderComptPhotIdeal.hh"

#include "G4UnitsTable.hh"

#include "GatePulseAdderComptPhotIdealMessenger.hh"
#include "G4Electron.hh"
#include "GateConfiguration.h"

GatePulseAdderComptPhotIdeal::GatePulseAdderComptPhotIdeal(GatePulseProcessorChain* itsChain,
											 const G4String& itsName)
											 : GateVPulseProcessor(itsChain,itsName)
{
    m_flgRejActPolicy=0;
	m_messenger = new GatePulseAdderComptPhotIdealMessenger(this);

}

GatePulseAdderComptPhotIdeal::~GatePulseAdderComptPhotIdeal()
{
    G4cout<<"value taken for flgRejActPolicy="<<m_flgRejActPolicy<<G4endl;
	delete m_messenger;
}



void GatePulseAdderComptPhotIdeal::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
#ifdef GATE_USE_OPTICAL
    // ignore pulses based on optical photons. These can be added using the opticaladder
    if (!inputPulse->IsOptical())
#endif
    {

         //G4cout << " Input hit pulse "<<"parentID="<<inputPulse->GetParentID()<<"  posStep"<<inputPulse->GetPostStepProcess()<<"  trackID="<<inputPulse->GetTrackID()<<"   energy="<<inputPulse->GetEnergy()<<" processCreator="<<inputPulse->GetProcessCreator()<<"  eIni"<<inputPulse->GetEnergyIniTrack()<<"  eFin="<<inputPulse->GetEnergyFin()<<G4endl;


        if(inputPulse->GetParentID()==0)
        {
            if(inputPulse->GetPostStepProcess()=="compt" ||inputPulse->GetPostStepProcess()=="phot" ||inputPulse->GetPostStepProcess()=="conv"  ){
                if(inputPulse->GetPostStepProcess()=="conv"){
                    flgEvtRej=1;
                }
                PulsePushBack(inputPulse, outputPulseList);
                lastTrackID.push_back(inputPulse->GetTrackID());            
                //G4cout << "inserting a pulse";
            }
            else{
                if(inputPulse->GetPostStepProcess()!="Transportation" )flgEvtRej=1;
            }
            //. La Eini de los primaries es su Eini antes de la primera interacci'on en SD of the layers no la Eini del track. This has been changed in the comptoncameraactor in where the hit coletion is saved


        }
        else{
            if(outputPulseList.empty()){
                //G4cout << " problems  the first hit is not a primary"<<G4endl;
                // G4cout << " For exmaple primary interaction outside the layers"<<G4endl;
                //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << G4endl;
            }
            else{
                //Simplificacion  inicial solo cogemos los procesos de electrones
                GatePulseList::reverse_iterator iter = outputPulseList.rbegin();
                   // GatePulseList::iterator iter = outputPulseList.begin();
                //int posInp;
                while (1){
                    //posInp=std::distance(outputPulseList.begin(),iter);
                    //if ( inputPulse->GetPDGEncoding() == ( G4Electron::Electron()->GetPDGEncoding() ) )
                    //{
                    //Esto podria guardarlo xq una vez que tengo los primarios guardados EmaxDepos es fijo
                    double EmaxDepos;
                    //if(iter==outputPulseList.begin()){
                    //EmaxDepos=(*iter)->GetEnergyIniTrack()- (*iter)->GetEnergyFin();
                    // }
                    //else{
                    //EmaxDepos=(*(iter-1))->GetEnergyFin()- (*iter)->GetEnergyFin();
                   // }

                   if(iter==(outputPulseList.rend()-1)){
                       EmaxDepos=(*iter)->GetEnergyIniTrack()- (*iter)->GetEnergyFin();
                   }
                   else{
                       EmaxDepos=(*(iter+1))->GetEnergyFin()- (*iter)->GetEnergyFin();
                   }

                    if ( (inputPulse->GetVolumeID() == (*iter)->GetVolumeID()) && (inputPulse->GetEventID() == (*iter)->GetEventID()) && inputPulse->GetEnergyIniTrack()<=(EmaxDepos+epsilonEnergy))
                    {

                        //first order secondaries
                        if(inputPulse->GetParentID()==1 && inputPulse->GetProcessCreator()==(*iter)->GetPostStepProcess()){
                            if( (*iter)->GetTrackID()==1){
                                 //first secondary for the pulse

                                //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " Merged\n";

                                // lastTrackID.at(posInp)=inputPulse->GetTrackID();
                                //anadir que lo llne solo si inputPulse->GetTrackID()

                                //---Pegote final
                                //Look if there is an output pulse already with the trackId of the inputPulse
                                bool isgood=true;
                                 GatePulseList::iterator iterIntern;
                                for (iterIntern = outputPulseList.begin() ; iterIntern != outputPulseList.end() ; ++iterIntern ){
                                     if(inputPulse->GetTrackID()==(*iterIntern)->GetTrackID()){
                                         isgood=false;
                                     }
                                }
                                if(isgood==true){                             
                                 (*iter)->CentroidMergeComptPhotIdeal(inputPulse);
                                 (*iter)->SetTrackID(inputPulse->GetTrackID());
                                }
                                //---Pegote final fin

                                break;
                            }
                            else{
                                //(*iter)->CentroidMergeComptPhotIdeal(inputPulse);;
                               // break;
                                //Esto funiiona para EMlivermore pero no para stantdard
                                if((*iter)->GetTrackID()==inputPulse->GetTrackID()){
                                    //first secondary for the pulse
                                   (*iter)->CentroidMergeComptPhotIdeal(inputPulse);;
                                   break;
                                }
                                else{

                                    ++iter;
                                    if (iter == outputPulseList.rend())
                                    {
                                        //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " End of list\n";
                                        break;
                                    }
                                }


                            }

                        }
                        //else if(inputPulse->GetParentID()>1 && lastTrackID.at(posInp)>=2){//rest of secondaries
                        else if(inputPulse->GetParentID()>1 && (*iter)->GetTrackID()>=2 && inputPulse->GetParentID()>=(*iter)->GetTrackID()){//rest of secondaries
                            //sECOND CONDITION to avoid to summ to a primary secondaries created in another layer For example a bremstralung that escape
                            (*iter)->CentroidMergeComptPhotIdeal(inputPulse);
                            //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " Merged\n";
                            //Alguna cndicion mas mirar 30548
                            //Buscar buena condicion !!! (pensar en base a processcreator parentID trackID)

                            break;

                        }
                        else{
                            //firs secondary without  process creator corresponding to  the saved one. Keep looking

                            ++iter;
                            if (iter == outputPulseList.rend())
                            {
                                //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " End of list\n";
                                break;
                            }
                        }




                    }
                    else{
                        ++iter;
                        if (iter == outputPulseList.rend())
                        {
                            //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " End of list\n";
                            break;
                        }

                    }
                }


            }
        }
    }



}



GatePulseList* GatePulseAdderComptPhotIdeal::ProcessPulseList(const GatePulseList* inputPulseList)
{
  if (!inputPulseList)
    return 0;
   flgEvtRej=0;
  size_t n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
        G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
  if (!n_pulses)
    return 0;


  GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

  GatePulseConstIterator iter;


  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter){
        ProcessOnePulse( *iter, *outputPulseList);
        //if(m_flgRejActPolicy==1 && flgEvtRej==1)break;
  }
  if(m_flgRejActPolicy==1 && flgEvtRej==1){
      //vaciar el outputLisr
     // G4cout<<"Rejecting the eventID ="<<outputPulseList->at(0)->GetEventID()<<G4endl;
      while (outputPulseList->size()) {
          delete outputPulseList->back();
          outputPulseList->erase(outputPulseList->end()-1);
      }
      return 0;
  }


  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
        G4cout << **iter << Gateendl;
      G4cout << Gateendl;
  }

  return outputPulseList;
}



void GatePulseAdderComptPhotIdeal::DescribeMyself(size_t )
{
	;
}

//this is standalone only because it repeats twice in processOnePulse()
inline void GatePulseAdderComptPhotIdeal::PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList)
{
	GatePulse* outputPulse = new GatePulse(*inputPulse);
    if (nVerboseLevel>1)
		G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		<< "Resulting pulse is: \n"
		<< *outputPulse << Gateendl << Gateendl ;
	outputPulseList.push_back(outputPulse);
}


