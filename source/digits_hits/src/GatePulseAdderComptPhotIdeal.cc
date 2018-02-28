

#include "GatePulseAdderComptPhotIdeal.hh"

#include "G4UnitsTable.hh"

#include "GatePulseAdderComptPhotIdealMessenger.hh"
#include "G4Electron.hh"
#include "GateConfiguration.h"

GatePulseAdderComptPhotIdeal::GatePulseAdderComptPhotIdeal(GatePulseProcessorChain* itsChain,
											 const G4String& itsName)
											 : GateVPulseProcessor(itsChain,itsName)
{
	m_messenger = new GatePulseAdderComptPhotIdealMessenger(this);
}

GatePulseAdderComptPhotIdeal::~GatePulseAdderComptPhotIdeal()
{
	delete m_messenger;
}


///REESCRIBIR!!!!!!!!!!!!!!!!!!
void GatePulseAdderComptPhotIdeal::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
#ifdef GATE_USE_OPTICAL
    // ignore pulses based on optical photons. These can be added using the opticaladder
    if (!inputPulse->IsOptical())
#endif
    {

        //  G4cout << "Entering adder";

        if(inputPulse->GetParentID()==0)
        {
            if(inputPulse->GetPostStepProcess()=="compt" ||inputPulse->GetPostStepProcess()=="phot"  ){
                PulsePushBack(inputPulse, outputPulseList);
                lastTrackID.push_back(inputPulse->GetTrackID());
                //G4cout << "inserting a pulse";
            }


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

                    if ( (inputPulse->GetVolumeID() == (*iter)->GetVolumeID()) && (inputPulse->GetEventID() == (*iter)->GetEventID()) && inputPulse->GetEnergyIniTrack()< EmaxDepos)
                    {
                        //first order secondaries
                        if(inputPulse->GetParentID()==1 && inputPulse->GetProcessCreator()==(*iter)->GetPostStepProcess()){
                            if( (*iter)->GetTrackID()==1){
                                 //first secondary for the pulse

                                //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " Merged\n";

                                // lastTrackID.at(posInp)=inputPulse->GetTrackID();
                                //anadir que lo llne solo si inputPulse->GetTrackID()

                                //---Pegote final
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
                                if((*iter)->GetTrackID()==inputPulse->GetTrackID()){
                                    //first secondary for the pulse
                                   (*iter)->CentroidMergeComptPhotIdeal(inputPulse);;
                                   break;
                                }
                                else{
                                    iter++;
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

                            iter++;
                            if (iter == outputPulseList.rend())
                            {
                                //G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " End of list\n";
                                break;
                            }
                        }




                    }
                    else{
                        iter++;
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


