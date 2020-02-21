#include "GateCoincidenceTreeReader.hh"

using std::cout;
using std::endl;

GateCoincidenceTreeReader::GateCoincidenceTreeReader(const std::string path)
    :filepath(path)
    ,f(0)
    ,m_coincTree(0)
    ,nentries(0)
    ,currentEntry(0)

{

    oldCoincID=0;
    m_coincBuffer.Clear();
}

void GateCoincidenceTreeReader::PrepareAcquisition(){
    gSystem->Load("libTree");
    // Open the input file
    f = new TFile(filepath.c_str(),"READ");

    if (f->IsZombie()) {
        cout << "Error opening file" << endl;
        exit(EXIT_FAILURE);
    }
    else{
        cout<<"Warning: Ideal coincidence reader. Information from the ideal digitizer (Eini, Efin) is employed to create cones"<<endl;
    }

    currentEntry=0;
    m_coincTree = (TTree*)f->Get("Coincidences");

    if (! m_coincTree)
    {
        cout<<"Could not find a tree of hits in the ROOT file '" +filepath<<endl;
    }

    nentries=(int)m_coincTree->GetEntries();

    // Set the addresses of the branch buffers: each buffer is a field of the root-hit structure
    GateCCCoincTree::SetBranchAddresses(m_coincTree,m_coincBuffer);

    LoadCoincData();


}


void GateCoincidenceTreeReader::LoadCoincData()
{
    // We've reached the end of file: set indicators to tell the caller that the reading failed
    if (currentEntry>=nentries){
        m_coincBuffer.runID=-1;
        m_coincBuffer.eventID=-1;
        return;
    }


    if (m_coincTree->GetEntry(currentEntry++)<=0) {
        G4cerr << "[GateCoincidenceFileReader::LoadCoincData]:\n"
               << "\tCould not read the next Coincidence!\n";
        m_coincBuffer.runID=-1;
        m_coincBuffer.eventID=-1;
    }
}



bool GateCoincidenceTreeReader::hasNext(){

    if(currentEntry>=nentries){

        return false;
    }
    else{
        return true;
    }
}

G4int GateCoincidenceTreeReader::PrepareNextEvent(){
    G4int currentCoincID = m_coincBuffer.coincID;
    G4int currentRunID = m_coincBuffer.runID;


    // We've reached the end-of-file
    if ( (currentCoincID==-1) && (currentRunID==-1) )
        return 0;
    int  counter=0;
    int firstEvtID=0;
    bool isTrueCoinc=true;
    // We loop until the data that have been read are found to be for a different event or run

    while ( (currentCoincID == m_coincBuffer.coincID) && (currentRunID == m_coincBuffer.runID) ) {
        if(counter==0){
            aCone.SetEnergy1(m_coincBuffer.energyIni-m_coincBuffer.energyFin);
            //aCone.SetEnergy1(m_coincBuffer.energy);
            aCone.SetEnergyR(m_coincBuffer.energyFin);
            G4ThreeVector pos1;
            pos1.setX(m_coincBuffer.globalPosX);
            pos1.setY(m_coincBuffer.globalPosY);
            pos1.setZ(m_coincBuffer.globalPosZ);
            aCone.SetPosition1(pos1);

            //First event od the coincidence
            firstEvtID=m_coincBuffer.eventID;
        }
        else{
            if(m_coincBuffer.eventID!=firstEvtID)isTrueCoinc=false;
            if(counter==1){

                G4ThreeVector pos2;
                pos2.setX(m_coincBuffer.globalPosX);
                pos2.setY(m_coincBuffer.globalPosY);
                pos2.setZ(m_coincBuffer.globalPosZ);
                aCone.SetPosition2(pos2);
                //aCone.SetEnergy2(m_coincBuffer.energy);
            }
        }
        counter++;//Only first two interaction info i nthe cones


        LoadCoincData();
    }
    aCone.SetNumSingles(counter);
    aCone.SetTrueFlag(isTrueCoinc);
    if (currentRunID==m_coincBuffer.runID){
        // We got a set of hits for the current run -> return 1
        return 1;
    }
    else
    {
        // We got a set of hits for a later run -> return 0
        return 0;
    }
}



GateComptonCameraCones GateCoincidenceTreeReader::PrepareEndOfEvent(){

    return aCone;
}

