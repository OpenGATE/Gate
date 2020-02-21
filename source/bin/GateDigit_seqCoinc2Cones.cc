#include <cstdlib>
#include "GateSequenceCoincidenceTreeReader.hh"
#include "GateCCRootDefs.hh"
#include "GateCCCoincidenceDigi.hh"




#include "GateMiscFunctions.hh"



using std::cout;
using std::endl;




int main(int argc, char *argv[])
{

    // Usage
    std::ostringstream usage;
    usage << std::endl
          << "Gate_CC_seqCoinc2Cones" << std::endl
          << "Read   the SequenceCoincidenceTree  and generate a tree file with cones info " << std::endl
          << "Usage : " << argv[0] << " <SequenceCoincidenceInputFile.root>" << "<ConesOutputFile.root> "<<std::endl;


    // Get user parameters
    if (argc != 3) {
        std::cout << "Need 3 parameters" << std::endl
                  << usage.str() << std::endl;
        exit(0);
    }


	std::string input_filePathName=argv[1];
     //Prepare output file
     std::string cones_filePathName = argv[2];
     TFile* pTfile = new TFile(cones_filePathName.c_str(),"RECREATE");
     GateCCConesTree* m_ConesTree=new GateCCConesTree("Cones");
     GateCCRootConesBuffer  m_ConesBuffer;
     m_ConesTree->Init(m_ConesBuffer);


   int coincCounter=0;
    GateSequenceCoincidenceTreeReader* m_coincFileReader= new GateSequenceCoincidenceTreeReader(input_filePathName);
         m_coincFileReader->PrepareAcquisition();
	 GateComptonCameraCones aCon;
         while( m_coincFileReader->hasNext()){
             //cout<<"prepareNext"<<endl;
              int isgood=m_coincFileReader->PrepareNextEvent();
	     //if(isgood==1){
	     aCon =m_coincFileReader->PrepareEndOfEvent();
             //if(isgood==1 && GateComptonCameraCones.GetTrueFlag()==true){
	     if(isgood==1){
                 coincCounter++;
		     m_ConesBuffer.Fill(&aCon);
                     //cout<<"buffer filled"<<endl;
                     m_ConesTree->Fill();
                     //cout<<"tree filled"<<endl;
                     m_ConesBuffer.Clear();		    
                   //writeTraEvent( m_coincFileReader->PrepareEndOfEvent(), coincCounter,ossCones);
             }
           }
     pTfile->Write();
    return 0;
}

