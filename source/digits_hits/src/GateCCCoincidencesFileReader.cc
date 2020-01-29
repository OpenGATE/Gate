/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateCCCoincidencesFileReader.hh"


#include <TBranch.h>
#include "GateOutputVolumeID.hh"
#include "GateTools.hh"
#include "GateCCRootDefs.hh"
#include "GateCCCoincidenceDigi.hh"

GateCCCoincidencesFileReader* GateCCCoincidencesFileReader::instance = 0;



// Private constructor: this function should only be called from GetInstance()
GateCCCoincidencesFileReader::GateCCCoincidencesFileReader(G4String file)
    : m_coincFile(0)
    , m_coincTree(0)
    , m_entries(0)
    , m_currentEntry(0)
{

    m_coincBuffer.Clear();

    unsigned lastPointPos=file.rfind(".");
    std::string pathExt = file.substr(lastPointPos);
    m_fileName=file.substr(0,lastPointPos);
    unsigned lastDirect=file.rfind("/");
    m_filePath= file.substr(0,lastDirect+1);

    std::string nameExt=file.substr(lastDirect+1);
    std::cout<<"name "<<m_fileName<<std::endl;

}




// Public destructor
GateCCCoincidencesFileReader::~GateCCCoincidencesFileReader()
{
    TerminateAfterAcquisition();

}





GateCCCoincidencesFileReader* GateCCCoincidencesFileReader::GetInstance(G4String filename)
{
    if (instance == 0)
        instance = new GateCCCoincidencesFileReader(filename);
    return instance;
}




void GateCCCoincidencesFileReader::PrepareAcquisition()
{
    std::cout<<"Preparing acquisition"<<std::endl;
    // Open the input file
    m_coincFile = new TFile((m_fileName+".root").c_str(),"READ");
    if (!m_coincFile)
    {
        G4String msg = "Could not open the requested Coincidences file '" + m_fileName + ".root'!";
        G4Exception( "GateCCCoincidencesFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg );
    }
    if (!(m_coincFile->IsOpen()))
    {
        G4String msg = "Could not open the requested Coincidences file '" + m_fileName + ".root'!";
        G4Exception( "GateCCCoincidencesFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg );
    }
    // Get the Coincidences tree
    m_coincTree = (TTree*)( m_coincFile->Get("Coincidences") );
    if (!m_coincTree)
    {
        G4String msg = "Could not find a tree of Coincidences in the ROOT file '" + m_fileName + ".root'!";
        G4Exception( "GateCCCoincidencesFileReader::PrepareBeforeAcquisition", "PrepareBeforeAcquisition", FatalException, msg);
    }
    // Reset the entry counters
    m_currentEntry=0;
    m_entries = m_coincTree->GetEntries();

    GateCCCoincTree::SetBranchAddresses(m_coincTree,m_coincBuffer);

    //  Load the first Coincidences into the root-Coincidences structure
    LoadCoincData();

}





G4int GateCCCoincidencesFileReader::PrepareNextEvent( )
{

    G4int currentCoincID = m_coincBuffer.coincID;
    G4int currentRunID = m_coincBuffer.runID;


    if ( (currentCoincID==-1) && (currentRunID==-1) )
        return 0;


    int counter=0;
    while ( (currentCoincID == m_coincBuffer.coincID) && (currentRunID == m_coincBuffer.runID) ) {
        GateCCCoincidenceDigi* acoinDigi=m_coincBuffer.CreateCoincidence();
        // G4cout<<"CoinID "<<acoinDigi->GetCoincidenceID()<<G4endl;

        if(counter==0){
            // Need to create a new pulse Digitizer delete them
            m_coincidencePulse= new GateCoincidencePulse("Coincidences",new GatePulse(&acoinDigi->GetPulse()),10,0);
            m_coincidencePulse->SetCoincID(acoinDigi->GetCoincidenceID());
        }
        else{
            m_coincidencePulse->push_back(new GatePulse(&acoinDigi->GetPulse()));
        }

        counter++;

        if(acoinDigi){
            delete acoinDigi;
            acoinDigi=0;
        }

        if(m_currentEntry==(m_entries)) break;//Not to lose the last coincidence
        LoadCoincData();
        if ( (currentCoincID==-1) && (currentRunID==-1) ) break;
    }



    if (currentRunID==m_coincBuffer.runID){
        // We got a set of Coincidencess for the current run -> return 1
        return 1;
    }
    else
    {
        // We got a set of Coincidencess for a later run -> return 0
        return 0;
    }
}


GateCoincidencePulse*  GateCCCoincidencesFileReader::PrepareEndOfEvent()
{

    return m_coincidencePulse;
}




void GateCCCoincidencesFileReader::TerminateAfterAcquisition()
{
    // Close the file
    if (m_coincFile) {
        delete m_coincFile;
        m_coincFile=0;
    }

    m_coincTree=0;
}


G4bool GateCCCoincidencesFileReader::HasNextEvent(){

    if (m_currentEntry>=m_entries){

        return false;
    }
    else{
        return true;
    }
}


void GateCCCoincidencesFileReader::LoadCoincData()
{
    // We've reached the end of file: set indicators to tell the caller that the reading failed
    if (m_currentEntry>=m_entries){
        m_coincBuffer.runID=-1;
        m_coincBuffer.eventID=-1;
        m_coincBuffer.coincID=-1;
        return;
    }

    if (m_coincTree->GetEntry(m_currentEntry++)<=0) {
        G4cerr << "[GateCoincidenceFileReader::LoadCoincData]:\n"
               << "\tCould not read the next Coincidence!\n";
        m_coincBuffer.runID=-1;
        m_coincBuffer.eventID=-1;
        m_coincBuffer.coincID=-1;
    }
}







void GateCCCoincidencesFileReader::Describe(size_t indent)
{
    // GateClockDependent::Describe(indent);
    G4cout << GateTools::Indent(indent) << "Coincidences-file name:    " << m_fileName << Gateendl;
    G4cout << GateTools::Indent(indent) << "Coincidences-file status:  " << (m_coincFile ? "open" : "closed" ) << Gateendl;
    if (m_coincTree) {
        G4cout << GateTools::Indent(indent) << "Coincidences-tree entries: " << m_entries << Gateendl;
        G4cout << GateTools::Indent(indent) << "Current entry:    " << m_currentEntry << Gateendl;
    }
}


//#endif
