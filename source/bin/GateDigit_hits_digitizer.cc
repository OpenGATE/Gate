

#include "G4SystemOfUnits.hh"
#include "GateMessageManager.hh"
#include "G4UImanager.hh"
#include "GateCCHitFileReader.hh"
#include "GateDigitizer.hh"
#include "GateDigi.hh"
#include "GateRandomEngine.hh"

#include "GateDetectorConstruction.hh"
#include "GateRunManager.hh"
#include "GateSignalHandler.hh"



#ifdef G4ANALYSIS_USE_ROOT
#include "TROOT.h"
#include "TRint.h"
#include "TPluginManager.h"
#endif
#include <getopt.h>
#include <cstdlib>
#include <queue>





int main(int argc, char *argv[])
{
    // Usage
    std::ostringstream usage;
    usage << std::endl
          << "Gate_CC_hits_digitizer" << std::endl
          << "Gate for Compton Camera" << std::endl
          << "Process hits to provide singles" << std::endl
          << "Usage : " << argv[0] << " <hit.root> <singles.root> <options.mac>" << std::endl;

    // Get user parameters
    if (argc != 4) {
        std::cout << "Need 4 parameters" << std::endl
                  << usage.str() << std::endl;
        exit(0);
    }


    std::string hits_filePathName=argv[1];
    std::string singles_filePathName = argv[2];
    std::string options_macrofile = argv[3];

    size_t foundPoint =  options_macrofile.find_last_of( "." );
    // Finding suffix
    G4String suffix = "";
    if( foundPoint != G4String::npos )
        suffix =  options_macrofile.substr( foundPoint + 1 );
    if( suffix != "mac" )
    {
        std::cout << "problemas last argument is not a macro file" << std::endl;
        exit(0);
    }

    // GATE Initialisation
    // First of all, set the G4cout to our message manager
    GateMessageManager* theGateMessageManager = GateMessageManager::GetInstance();
    G4UImanager::GetUIpointer()->SetCoutDestination( theGateMessageManager );
    GateSignalHandler::Install();

    GateRandomEngine* randomEngine = GateRandomEngine::GetInstance();

    //To have volumeID in the pulses and use properly the adder, the geometry ids needed to interpret volumeD of hits.
    //I have in the tree also layer name I could use that but to this end it should be inserted somehow in the pulses and taken into account in the adders
    // Construct the default run manager
    GateRunManager* runManager = new GateRunManager;
    // Set the DetectorConstruction
    GateDetectorConstruction* gateDC = new GateDetectorConstruction();
    runManager->SetUserInitialization( gateDC );
    // Set the PhysicsList
    runManager->SetUserInitialization( GatePhysicsList::GetInstance() );
    // Initialize G4 kernel
    //runManager->InitializeAll();

    //With these lines I enable /gate/digitizer/layers  command because the pointer is called digitizer and the chain layers.
    GateDigitizer*  digitizer =    GateDigitizer::GetInstance();
    const G4String thechainName="layers";
    GatePulseProcessorChain* chain=new GatePulseProcessorChain(digitizer, thechainName);
    digitizer->StoreNewPulseProcessorChain(chain);

    // Get the pointer to the User Interface manager
    G4UImanager* UImanager = G4UImanager::GetUIpointer();
    // Launching Gate  macro file
    //GateMessage( "Core", 0, "Starting macro " << options_macrofile << G4endl);
    std::cout << "Reading " << options_macrofile << " ..." << std::endl;
    G4String command = "/control/execute ";
    UImanager->ApplyCommand( command + options_macrofile );
    std::cout << "Done" << std::endl;



    //Prepare output file
    TFile* pTfile = new TFile(singles_filePathName.c_str(),"RECREATE");
    //Prepare Tree
    GateCCSingleTree* m_SingleTree=new GateCCSingleTree("Singles");
    GateCCRootSingleBuffer  m_SinglesBuffer;
    m_SingleTree->Init(m_SinglesBuffer);



    //Read Hits tree
    GateCCHitFileReader* m_hitFileReader= GateCCHitFileReader::GetInstance(hits_filePathName);
    m_hitFileReader->PrepareAcquisition();
    while(m_hitFileReader->HasNextEvent()){

        m_hitFileReader->PrepareNextEvent();
        digitizer->Digitize(m_hitFileReader->PrepareEndOfEvent());
        //SaveInto Single tree the pulses
        GatePulseList* pPulseList=digitizer->FindPulseList(thechainName);
        if(pPulseList){
            if(pPulseList->size()>0){


               GatePulseConstIterator iterIn;
                for (iterIn = pPulseList->begin() ; iterIn != pPulseList->end() ; ++iterIn){

                    GateDigi* aSingleDigi=new GateDigi(*iterIn);


                    m_SinglesBuffer.Fill(aSingleDigi);
                    m_SingleTree->Fill();
                    m_SinglesBuffer.Clear();

	             if(aSingleDigi){
                        delete aSingleDigi;
                        aSingleDigi=0;
                    }
                }
            }
        }

    }
    pTfile->Write();
    m_hitFileReader->TerminateAfterAcquisition();

    delete  randomEngine;
    delete runManager;
    delete digitizer;




    return 0;
}

