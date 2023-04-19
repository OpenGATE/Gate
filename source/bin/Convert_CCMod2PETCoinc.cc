


#include "GateMessageManager.hh"
#include "G4UImanager.hh"
#include "GateDigitizer.hh"
#include "GateCCRootDefs.hh"
#include "GateRootDefs.hh"
#include "GateCCCoincidencesFileReader.hh"
#include "GateCCCoincidenceDigi.hh"
#include"GateCoincidenceDigi.hh"


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
         << "ConvertCCMod2PETCoin" << std::endl
         << "Read  the Coincidence tree (CCMod actor format)  and generate  a Coincidence tree file with 2-single coincidence following PET systems structure   " << std::endl
         << "Usage : " << argv[0] << " <CCModCoincidenceInputFile.root>" << "<PETFormatCoincFilename.root> <geom.mac>" << std::endl;


   // Get user parameters
   if (argc != 4) {
       std::cout << "Need 4 parameters" << std::endl
                 << usage.str() << std::endl;
       exit(0);
   }
   std::string coinc_filePathName=argv[1];
   std::string sequenCoinc_filePathName = argv[2];
   std::string options_macrofile = argv[3];


   // First of all, set the G4cout to our message manager
   GateMessageManager* theGateMessageManager = GateMessageManager::GetInstance();
   G4UImanager::GetUIpointer()->SetCoutDestination( theGateMessageManager );
   GateSignalHandler::Install();

   //To be avaible to interpret the volumeID  of the pulses the geometry of the system is needed.
   GateRunManager* runManager = new GateRunManager;
   // Set the DetectorConstruction
   GateDetectorConstruction* gateDC = new GateDetectorConstruction();
   runManager->SetUserInitialization( gateDC );
   // Set the PhysicsList is needed although I do not set any list
   runManager->SetUserInitialization( GatePhysicsList::GetInstance() );


   // Get the pointer to the User Interface manager
   G4UImanager* UImanager = G4UImanager::GetUIpointer();
   std::cout << "Reading " << options_macrofile << " ..." << std::endl;
   G4String command = "/control/execute ";
   UImanager->ApplyCommand( command + options_macrofile );


   TFile* pTfile = new TFile(sequenCoinc_filePathName.c_str(),"RECREATE");
   GateCoincTree* m_PETcoincTree=new GateCoincTree("Coincidences");
   GateRootCoincBuffer  m_PETcoincBuffer;
   m_PETcoincTree->Init(m_PETcoincBuffer);


   GateCCCoincidencesFileReader* m_coincFileReader= GateCCCoincidencesFileReader::GetInstance(coinc_filePathName);
   m_coincFileReader->PrepareAcquisition();


   while( m_coincFileReader->HasNextEvent()){
       int isgood=m_coincFileReader->PrepareNextEvent();

       if(isgood==1){
           GateCoincidencePulse* m_coincidencePulse=m_coincFileReader->PrepareEndOfEvent();
           if (m_coincidencePulse->size()==2){
        	//OK GND CC 2022 TODO
        	  /* GateCoincidenceDigiOld* aPETCoincDigi= new GateCoincidenceDigiOld(m_coincidencePulse);
               m_PETcoincBuffer.Fill(aPETCoincDigi);
               m_PETcoincTree->Fill();
               m_PETcoincBuffer.Clear();

               if(aPETCoincDigi){
                   delete aPETCoincDigi;
                   aPETCoincDigi=0;
               }
        	 */
           }
           else{
                if (m_coincidencePulse->size()<2){

                    cout<<"Error Coincidences of less than two singles"<<endl;
                }
                else{cout<<"Multiple coinc not written in the outputfile"<<endl;}

           }
       }
   }
   pTfile->Write();


   m_coincFileReader->TerminateAfterAcquisition();
   std::cout<<"Terminate acquisition"<<std::endl;


   return 0;
}
