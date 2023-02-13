/*
 *	\file Gate_CC_singles_sorter.cc
 */



#include "GateMessageManager.hh"
#include "G4UImanager.hh"
#include "GateDigitizer.hh"
#include "GateCCRootDefs.hh"
#include "GateCCSinglesFileReader.hh"
#include "GateCCCoincidenceDigi.hh"


//in order to have volumeID Geomtery needed
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
          << "Gate_CC_singles_sorter" << std::endl
          << "Gate for Compton Camera" << std::endl
          << "Process single to provide coincidences" << std::endl
          << "Usage : " << argv[0] << " <singlesInput.root> <coincidenceOutput.root> <options.mac> absorberSDVolName" << std::endl;


    // Get user parameters
    if (argc != 5) {
        std::cout << "Need 5 parameters" << std::endl
                  << usage.str() << std::endl;
        exit(0);
    }
    std::string singles_filePathName=argv[1];
    std::string coinc_filePathName = argv[2];
    std::string options_macrofile = argv[3];
    std::string absorberSDName = argv[4];



    // GATE Initialisation
    // First of all, set the G4cout to our message manager
    GateMessageManager* theGateMessageManager = GateMessageManager::GetInstance();
    G4UImanager::GetUIpointer()->SetCoutDestination( theGateMessageManager );
    GateSignalHandler::Install();

    //To be avaible to interpret the volumeID  of the singles the geometry of the system is needed. LayerName is also stored at first.
    GateRunManager* runManager = new GateRunManager;
    // Set the DetectorConstruction
    GateDetectorConstruction* gateDC = new GateDetectorConstruction();
    runManager->SetUserInitialization( gateDC );
    // Set the PhysicsList is needed altough I do not set any list
    runManager->SetUserInitialization( GatePhysicsList::GetInstance() );
    // Initialize G4 kernel
    //runManager->InitializeAll();



    GateDigitizer*  digitizer =    GateDigitizer::GetInstance();


    //Include a coincidence sorter into the digitizer with a default coincidence window that can be changed with macro commands
    G4double coincidenceWindow = 10.* ns;
    bool IsCCSorter=1;
    GateCoincidenceSorterOld* coincidenceSorter = new GateCoincidenceSorterOld(digitizer,"Coincidences",coincidenceWindow,"layers",IsCCSorter);
    digitizer->StoreNewCoincidenceSorter(coincidenceSorter);


    // Get the pointer to the User Interface manager
    G4UImanager* UImanager = G4UImanager::GetUIpointer();
    // Launching Gate  macro file
    std::cout << "Reading " << options_macrofile << " ..." << std::endl;
    G4String command = "/control/execute ";
    UImanager->ApplyCommand( command + options_macrofile );
    std::cout << "Done" << std::endl;

     coincidenceSorter->SetAbsorberSDVol(absorberSDName);


     //Prepare output file
     TFile* pTfile = new TFile(coinc_filePathName.c_str(),"RECREATE");
     //Prepare Tree
     GateCCCoincTree* m_CoincTree=new GateCCCoincTree("Coincidences");
     GateCCRootCoincBuffer  m_CoincBuffer;
     m_CoincTree->Init(m_CoincBuffer);


     int coincID=0;

     //Read singles file
     GateCCSinglesFileReader* m_singlesFileReader= GateCCSinglesFileReader::GetInstance(singles_filePathName);
     m_singlesFileReader->PrepareAcquisition();
     while(m_singlesFileReader->HasNextEvent()){
         //cout<<"prepareNext"<<endl;
         m_singlesFileReader->PrepareNextEvent();
         // cout<<"ErasepulseListt"<<endl;
         digitizer->ErasePulseListVector();
         //cout<<"ProcessingSingleList"<<endl;
         coincidenceSorter->ProcessSinglePulseList(m_singlesFileReader->PrepareEndOfEvent());
         std::vector<GateCoincidencePulse*> coincPulseVector=digitizer->FindCoincidencePulse("Coincidences");

         //std::cout<<" from digitizer coind pulse LsitAlias vetcot="<<digitizer->m_coincidencePulseListAliasVector.size()<<G4endl;
         if(coincPulseVector.size()>=1){
             for (std::vector<GateCoincidencePulse*>::const_iterator it = coincPulseVector.begin();it != coincPulseVector.end() ; ++it){

                 GateCoincidencePulse* coincPulse=*it;
                 //GAteCoincidencePulse compose of several GatePulse

                 unsigned int numCoincPulses=coincPulse->size();
                 //cout<<"number of pulses"<<numCoincPulses<<endl;

                 //cout<<"first pulse evtID"<<coincPulse->at(0)->GetEventID()<<"first pulse time+"<<coincPulse->at(0)->GetTime()<<G4endl;
                 for(unsigned int i=0;i<numCoincPulses;i++){
                     GateCCCoincidenceDigi* aCoinDigi=new GateCCCoincidenceDigi(coincPulse->at(i),coincID);
                     m_CoincBuffer.Fill(aCoinDigi);
                     //cout<<"buffer filled"<<endl;
                     m_CoincTree->Fill();
                     //cout<<"tree filled"<<endl;
                     m_CoincBuffer.Clear();
                     //std::cout<<"Filling coincidences"<<std::endl;
                 }

                 coincID++;
             }
         }
     }


     pTfile->Write();
     m_singlesFileReader->TerminateAfterAcquisition();


    return 0;
}


