#include <cstdlib>
#include "GateSequenceCoincidenceTreeReader.hh"
#include "GateCCRootDefs.hh"
#include "GateCCCoincidenceDigi.hh"




#include "GateMiscFunctions.hh"



using std::cout;
using std::endl;
void writeEvent(GateComptonCameraCones aCone, int nLayers, std::ofstream &oss){
    if(nLayers==2){
        oss<<"2"<<'\t'<<"1"<<'\t'<<aCone.GetPosition1().getX()/mm<<'\t'<<aCone.GetPosition1().getY()/mm<<'\t'<<aCone.GetPosition1().getZ()/mm<<'\t'<<aCone.GetEnergy1()/keV<<'\t'<<"2"<<'\t'<<aCone.GetPosition2().getX()/mm<<'\t'<<aCone.GetPosition2().getY()/mm<<'\t'<<aCone.GetPosition2().getZ()/mm<<'\t'<<aCone.GetEnergyR()/keV<<'\t'<<"3"<<'\t'<<"0"<<'\t'<<"0"<<'\t'<<"0"<<'\t'<<"0"<<'\n';
    }
    else if (nLayers==3){
        double E3= aCone.GetEnergyR()/keV-aCone.GetEnergy2()/keV;
        oss<<"2"<<'\t'<<"1"<<'\t'<<aCone.GetPosition1().getX()/mm<<'\t'<<aCone.GetPosition1().getY()/mm<<'\t'<<aCone.GetPosition1().getZ()/mm<<'\t'<<aCone.GetEnergy1()/keV<<'\t'<<"2"<<'\t'<<aCone.GetPosition2().getX()/mm<<'\t'<<aCone.GetPosition2().getY()/mm<<'\t'<<aCone.GetPosition2().getZ()/mm<<'\t'<<aCone.GetEnergy2()/keV<<'\t'<<"3"<<'\t'<<aCone.GetPosition3().getX()/mm<<'\t'<<aCone.GetPosition3().getY()/mm<<'\t'<<aCone.GetPosition3().getZ()/mm<<'\t'<<E3<<'\n';

    }
}


int main(int argc, char *argv[])
{

    // Usage
    std::ostringstream usage;
    usage << std::endl
          << "Gate_CC_RootSeqCoinTree2MACACOListMode" << std::endl
          << "Read   the SequenceCoincidenceTree  and with the information of a cone generate a .dat   file for MACACO " << std::endl
          << "Usage : " << argv[0] << " <SequenceCoincidenceInput.root>" << "numberofLayers(2/3)"<<std::endl;


    // Get user parameters
    if (argc != 3) {
        std::cout << "Need 3 parameters" << std::endl
                  << usage.str() << std::endl;
        exit(0);
    }
    std::string input_filePathName=argv[1];
    //Prepare output file

    std::string outputFilename = removeExtension(input_filePathName);
    outputFilename = outputFilename+"nLayers_"+argv[2]+ ".dat";

    // Create output file
    std::ofstream ossCones;
    OpenFileOutput(outputFilename, ossCones);

    //int nLayers=atoi(argv[2]);
    std::istringstream ss(argv[2]);
    int nLayers;
    if (!(ss >> nLayers)) {
      std::cerr << "Invalid number: " << argv[2] << '\n';
    } else if (!ss.eof()) {
      std::cerr << "Trailing characters after number: " << argv[2] << '\n';
    }


    //int coincCounter=0;
    GateSequenceCoincidenceTreeReader* m_coincFileReader= new GateSequenceCoincidenceTreeReader(input_filePathName);
    m_coincFileReader->PrepareAcquisition();
    //Saca un pulseList por cada eventID with the corresponfing number of pulses
    while( m_coincFileReader->hasNext()){
        //cout<<"prepareNext"<<endl;
        int isgood=m_coincFileReader->PrepareNextEvent();
        if(isgood==1){
            //coincCounter++;
            writeEvent( m_coincFileReader->PrepareEndOfEvent(), nLayers,ossCones);
        }
    }
    return 0;
}

