


#include "GateMessageManager.hh"
#include "G4UImanager.hh"
#include "GateDigitizer.hh"
#include "GateCCRootDefs.hh"
#include "GateCCCoincidencesFileReader.hh"
#include "GateCCCoincidenceDigi.hh"


//in order to have volumeID Geomtery neeede
#include "GateDetectorConstruction.hh"
#include "GateRunManager.hh"
#include "GateSignalHandler.hh"

#include <cstdlib>

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{

   // Usage
   std::ostringstream usage;
   usage << std::endl
         << "Gate_CC_coincidence_procesor" << std::endl
         << "Gate for Compton Camera" << std::endl
         << "Process coincidence  to provide  sequence coincidences" << std::endl
         << "Usage : " << argv[0] << " <coincidenceInput.root> <sequenceCoincidenceOutput.root> <options.mac>" << std::endl;


   // Get user parameters
   if (argc != 4) {
       std::cout << "Need 4 parameters" << std::endl
                 << usage.str() << std::endl;
       exit(0);
   }
   std::string coinc_filePathName=argv[1];
   std::string sequenCoinc_filePathName = argv[2];
   std::string options_macrofile = argv[3];


   // GATE Initialisation
   // First of all, set the G4cout to our message manager
   GateMessageManager* theGateMessageManager = GateMessageManager::GetInstance();
   G4UImanager::GetUIpointer()->SetCoutDestination( theGateMessageManager );
   GateSignalHandler::Install();

   //To be avaible to interpret the volumeID  of the pulses the geometry of the system is needed.
   GateRunManager* runManager = new GateRunManager;
   // Set the DetectorConstruction
   GateDetectorConstruction* gateDC = new GateDetectorConstruction();
   runManager->SetUserInitialization( gateDC );
   // Set the PhysicsList is needed altough I do not set any list
   runManager->SetUserInitialization( GatePhysicsList::GetInstance() );
   // Initialize G4 kernel
   //runManager->InitializeAll();



   //Digitizer
   GateDigitizer*  digitizer =    GateDigitizer::GetInstance();
   //#######################################################
   G4double coincidenceWindow = 10.* ns;
   bool IsCCSorter=1;
   const G4String thedigitizerSorterName="Coincidences";
   GateCoincidenceSorterOld*coincidenceSorter = new GateCoincidenceSorterOld(digitizer,thedigitizerSorterName,coincidenceWindow,"layers",IsCCSorter);
   digitizer->StoreNewCoincidenceSorter(coincidenceSorter);
   //I  am not sure if it is necesaary or no to do the store  of coincidneces in the digitizer from the tree to process them
   //##########################################################33



   // Get the pointer to the User Interface manager
   G4UImanager* UImanager = G4UImanager::GetUIpointer();
   // Launching Gate  macro file
   std::cout << "Reading " << options_macrofile << " ..." << std::endl;
   G4String command = "/control/execute ";
   UImanager->ApplyCommand( command + options_macrofile );


   TFile* pTfile = new TFile(sequenCoinc_filePathName.c_str(),"RECREATE");

   std::vector<G4String> coincidenceChainNames;
   // std::cout << "Number of coincidence chains loaded" <<digitizer->GetmCoincChainListSize()<<std::endl;
   for(unsigned int i=0; i<digitizer->GetmCoincChainListSize(); i++){
       coincidenceChainNames.push_back(digitizer->GetCoincChain(i)->GetOutputName());
       std::cout << "Name of chains=" <<coincidenceChainNames.back()<<std::endl;
   }

   std::vector<std::unique_ptr<GateCCCoincTree>> m_coincChainTree;
   GateCCRootCoincBuffer  m_CoincBuffer;
   for(unsigned int i=0; i<digitizer->GetmCoincChainListSize(); i++){
       m_coincChainTree.emplace_back(new GateCCCoincTree(coincidenceChainNames.at(i), "CoincidenceChain tree"));
       m_coincChainTree.back()->Init(m_CoincBuffer);
   }



   GateCCCoincidencesFileReader* m_coincFileReader= GateCCCoincidencesFileReader::GetInstance(coinc_filePathName);
   m_coincFileReader->PrepareAcquisition();

   while( m_coincFileReader->HasNextEvent()){

       int isgood=m_coincFileReader->PrepareNextEvent();

       if(isgood==1){
           digitizer->StoreCoincidencePulse(m_coincFileReader->PrepareEndOfEvent());//meterlo en el digitizer
           //           auto coincidencePulseV=digitizer->FindCoincidencePulse(thedigitizerSorterName);
           //            if(coincidencePulseV.size()>0){
           //                //std::cout<<" coinc Pulse Size "<<coincidencePulseV.size()<<std::endl;
           //                for (std::vector<GateCoincidencePulse*>::const_iterator it = coincidencePulseV.begin();it != coincidencePulseV.end() ; ++it){
           //                    //Here I have my coincidences (Singles fill a buffer  every event and when the size is above a THR  coincidences are processed and then we have a ocincidnece pulse output after the event which is erase at hte beginning of the next event
           //                    G4cout<<"pulses in the coincidence"<< (*it)->size()<<G4endl;
           //                }
           //            }
           for(unsigned int i=0; i<digitizer->GetmCoincChainListSize(); i++){
               digitizer->GetCoincChain(i)->ProcessCoincidencePulses();
           }


           for(unsigned int iChain=0; iChain<digitizer->GetmCoincChainListSize(); iChain++){


               std::vector<GateCoincidencePulse*> coincidencePulseChain = digitizer->FindCoincidencePulse(digitizer->GetCoincChain(iChain)->GetOutputName());


               if(coincidencePulseChain.size()>0){
                   // G4cout<<" Ncoincidence vector found in the digitizer ="<<coincidencePulseChain.size()<<G4endl;
                   for (std::vector<GateCoincidencePulse*>::const_iterator it = coincidencePulseChain.begin();it < coincidencePulseChain.end() ; ++it){

                       GateCoincidencePulse* coincPulse=*it;

                       unsigned int numCoincPulses=coincPulse->size();
                       // G4cout<<" coinID "<<coincPulse->GetCoincID()<<G4endl;


                       for(unsigned int i=0;i<numCoincPulses;i++){
                           GateCCCoincidenceDigi* aCoinDigi=new GateCCCoincidenceDigi(coincPulse->at(i),coincPulse->GetCoincID());

                           m_CoincBuffer.Fill(aCoinDigi);

                           m_coincChainTree.at(iChain)->Fill();
                           m_CoincBuffer.Clear();

                           if(aCoinDigi){
                               delete aCoinDigi;
                               aCoinDigi=0;
                           }


                       }


                   }


                   // G4cout<<"Size the input pCoin pulse non zero"<<G4endl;

               }




           }
           digitizer->ErasePulseListVector();

       }
       else{
           //Not good maybe delte GateCoincidnece vector
       }
   }





    pTfile->Write();


    m_coincFileReader->TerminateAfterAcquisition();
   std::cout<<"Terminate acquisition"<<std::endl;


   return 0;
}

