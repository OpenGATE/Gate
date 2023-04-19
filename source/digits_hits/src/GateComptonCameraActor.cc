/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateComptonCameraActor
*/

#include <G4EmCalculator.hh>
#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include <G4PhysicalConstants.hh>
#include <G4VProcess.hh>
#include <G4DigiManager.hh>
#include "GateComptonCameraActor.hh"
#include "GateComptonCameraActorMessenger.hh"
#include "GateMiscFunctions.hh"
#include "GateDigitizer.hh"
#include "GateDigi.hh"





typedef std::pair<G4String,GatePulseList*> 	GatePulseListAlias;

const G4String GateComptonCameraActor::thedigitizerName="layers";
const G4String GateComptonCameraActor::thedigitizerSorterName="Coincidences";

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateComptonCameraActor::GateComptonCameraActor(G4String name, G4int depth):
    GateVActor(name,depth)
{
    GateDebugMessageInc("Actor",4,"GateComptonamerActor() -- begin\n");

    G4cout<<"###########################CONSTRUCTOR of GATEComptonCameraActor#############################################"<<G4endl;

    Ei = 0.;
    Ef = 0.;
    newEvt = true;
    newTrack = true;
    edepEvt = 0.;
    slayerID=-1;


    //Default values for the variables that will be loaded in the messenger
    mNumberTotScattLayers=0;
    mNameOfAbsorberSDVol="absorber";
    mNameOfScattererSDVol="scatterer";
    mSaveHitsTreeFlag=0;
    mSaveSinglesTreeFlag=1;
    mSaveCoincidencesTreeFlag=1;
    mSaveCoincidenceChainsTreeFlag=1;
    mSaveEventInfoTreeFlag=false;
    mParentIDSpecificationFlag=false;

    //Messenger load values
    pMessenger = new GateComptonCameraActorMessenger(this);

    EnableEnergy=1;
    EnableEnergyIni=1;
    EnableEnergyFin=1;
    EnableTime=1;
    EnableXPosition=1;
    EnableYPosition=1;
    EnableZPosition=1;
    EnableXLocalPosition=1;
    EnableYLocalPosition=1;
    EnableZLocalPosition=1;
    EnableXSourcePosition=1;
    EnableYSourcePosition=1;
    EnableZSourcePosition=1;
    EnableVolumeID=1;
    EnableSourceEnergy=1;
    EnableSourcePDG=1;
    EnablenCrystalCompt=1;
    EnablenCrystalConv=1;
    EnablenCrystalRayl=1;

    emcalc = new G4EmCalculator;
    m_digitizer = GateDigitizer::GetInstance();

    //I enable digitizer/layers chain. digitizer/Singles chain is already created in Gate.cc  within  if G4Analysis_use_general.
    //digitizer() function can be applied independently to both chains
    chain=new GatePulseProcessorChain(m_digitizer, thedigitizerName);
    m_digitizer->StoreNewPulseProcessorChain(chain);
    //Include a coincidence sorter into the digitizer with a default coincidence window that can be changed with macro commands
    G4double coincidenceWindow = 10.* ns;
    //Flag to identify when the sorte is applied to the CC
    bool IsCCSorter=1;
    coincidenceSorter = new GateCoincidenceSorterOld(m_digitizer,thedigitizerSorterName,coincidenceWindow,thedigitizerName,IsCCSorter);
    m_digitizer->StoreNewCoincidenceSorter(coincidenceSorter);

    GateDebugMessageDec("Actor",4,"GateComptonCamera() -- end\n");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateComptonCameraActor::~GateComptonCameraActor(){
    G4cout<<"destructor CCActor"<<G4endl;
    GateDebugMessageInc("Actor",4,"~GateComptonCameraActor() -- begin\n");
    GateDebugMessageDec("Actor",4,"~GateComptonCameraActor() -- end\n");

    //    delete pMessenger;
    delete m_digitizer;
    delete emcalc;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct`
void GateComptonCameraActor::Construct()
{

    G4cout<<"construction"<<G4endl;
    GateVActor::Construct();

    coincidenceSorter->SetAbsorberSDVol(mNameOfAbsorberSDVol);
    // Enable callbacks
    EnableBeginOfRunAction(true);
    EnableBeginOfEventAction(true);

    EnablePreUserTrackingAction(true);
    EnablePostUserTrackingAction(true);

    EnableUserSteppingAction(true);

    EnableEndOfEventAction(true); // for save every n


    //I think I can rm it It was for a test
    attachPhysVolumeName=mVolume->GetPhysicalVolumeName();//Name of  the BB where the actor is attached
    G4cout<<"GAteComptonCameraActor:Construct  numTotScatt"<<mNumberTotScattLayers<<G4endl;


    //Layernames are set to store in those volumes the hits in the collection
    if(mNumberTotScattLayers==0){//I am not using the messenger to specify number of layer and names

        //By default one absorber one scatter
        //This should be changed to accomodate CC composed of one single block acting as absorber and scatterer. Coincidnecesorter conditions need to be changed for  this situation
        nDaughterBB=1;
        layerNames.push_back(mNameOfAbsorberSDVol);
    }
    else{
        nDaughterBB=mNumberTotScattLayers+1;

        for(unsigned int i=0; i < nDaughterBB; i++) {
            if(i==(nDaughterBB-1)){
                layerNames.push_back(mNameOfAbsorberSDVol);
            }
            else{
                if(i==0){
                    layerNames.push_back(mNameOfScattererSDVol);
                }
                else{
                    layerNames.push_back(mNameOfScattererSDVol+std::to_string(i));

                }
            }
            // G4cout<<"layerNames "<<layerNames.back()<<G4endl;
        }

    }


    //To select primary particles for example in ion sources. Otherwise the associated to the primary particle is set to the particle with parentID=0
    if(mParentIDSpecificationFlag){
        // Read the file and load the values in the vector called mParentIDFileName
        std::ifstream textfile;
        std::string line;
        textfile.open(mParentIDFileName.c_str(), std::ios_base::in);  // open data
        if(textfile.is_open()!=true) G4cout<<"[GateComptonCameraActor::Construct]: parentID specification file not correctly opened"<<G4endl;
        int tmpParentID;
        while(textfile.is_open()){
            getline(textfile,line); //read stream line by line
            std::istringstream in(line);//make a stream for the line itself
            in >> tmpParentID;
            if(textfile.eof()){
                textfile.close();
                break;
            }
            specfParentID.push_back(tmpParentID);

        }
    }



   G4String extension = getExtension(mSaveFilename);
   if(extension!="root"&&extension!="txt" &&extension!="npy") GateError("Unknown extension for CC actor output");
    std::string filename = removeExtension(mSaveFilename);
   //set a file for each tree. Filemanager it is not prepared to add several trees in a file
    if(mSaveHitsTreeFlag){
        G4String filenameH=filename+"_Hits."+extension;
        if(extension == "root")
            mFileHits.add_file(filenameH,  "root");
        else if(extension == "npy")
            mFileHits.add_file(filenameH, "npy");
        else if(extension == "txt")
            mFileHits.add_file(filenameH, "txt");
        else
            GateError("Unknown extension for CC actor output");
        mFileHits.set_tree_name("Hits");
        //equivalent to Tree.Init(buffer)
        mFileHits.write_variable("runID",&m_HitsBuffer.runID);
        mFileHits.write_variable("eventID", &m_HitsBuffer.eventID);

        //someflags missing. I need to define them
        mFileHits.write_variable("PDGEncoding", &m_HitsBuffer.PDGEncoding);
        mFileHits.write_variable("trackID", &m_HitsBuffer.trackID);
        mFileHits.write_variable("parentID", &m_HitsBuffer.parentID);
        mFileHits.write_variable("trackLocalTime", &m_HitsBuffer.trackLocalTime);
        if(EnableTime)mFileHits.write_variable("time", &m_HitsBuffer.time);
        if(EnableEnergy)mFileHits.write_variable("edep", &m_HitsBuffer.edep);
        mFileHits.write_variable("stepLength", &m_HitsBuffer.stepLength);
        mFileHits.write_variable("trackLength", &m_HitsBuffer.trackLength);
        if(EnableXPosition) mFileHits.write_variable("posX", &m_HitsBuffer.posX);
        if(EnableYPosition)mFileHits.write_variable("posY", &m_HitsBuffer.posY);
        if(EnableZPosition)mFileHits.write_variable("posZ", &m_HitsBuffer.posZ);
        if(EnableXLocalPosition)mFileHits.write_variable("localPosX", &m_HitsBuffer.localPosX);
        if(EnableYLocalPosition)mFileHits.write_variable("localPosY", &m_HitsBuffer.localPosY);
        if(EnableZLocalPosition)mFileHits.write_variable("localPosZ", &m_HitsBuffer.localPosZ);
        if(EnableXSourcePosition)mFileHits.write_variable("sourcePosX", &m_HitsBuffer.sPosX);
        if(EnableYSourcePosition)mFileHits.write_variable("sourcePosY", &m_HitsBuffer.sPosY);
        if(EnableZSourcePosition)mFileHits.write_variable("sourcePosZ", &m_HitsBuffer.sPosZ);
        if(EnableSourceEnergy)mFileHits.write_variable("sourceEnergy", &m_HitsBuffer.sourceEnergy);
        if(EnableSourcePDG)mFileHits.write_variable("sourcePDG", &m_HitsBuffer.sourcePDG);
        if(EnablenCrystalConv)mFileHits.write_variable("nCrystalConv",&m_HitsBuffer.nCrystalConv);
        if(EnablenCrystalCompt)mFileHits.write_variable("nCrystalCompt", &m_HitsBuffer.nCrystalCompt);
        if(EnablenCrystalRayl) mFileHits.write_variable("nCrystalRayl", &m_HitsBuffer.nCrystalRayl);
        if(EnableEnergyFin)mFileHits.write_variable("energyFinal",&m_HitsBuffer.energyFin);
        if(EnableEnergyIni)mFileHits.write_variable("energyIniT",&m_HitsBuffer.energyIniT);
        mFileHits.write_variable("postStepProcess", m_HitsBuffer.postStepProcess, sizeof(m_HitsBuffer.postStepProcess));
        mFileHits.write_variable("processName", m_HitsBuffer.processName, sizeof(m_HitsBuffer.processName));
        //It is not necessary
        mFileHits.write_variable("layerName", m_HitsBuffer.layerName, sizeof(m_HitsBuffer.layerName));
        if(EnableVolumeID)mFileHits.write_variable("volumeID", m_HitsBuffer.volumeID,ROOT_VOLUMEIDSIZE);

        mFileHits.write_header();

    }
   if( mSaveSinglesTreeFlag ){//Equivalent to the SingleTree->Init(SingleBuffer)

       G4String filenameS=filename+"_Singles."+extension;
       if(extension == "root")
           mFileSingles.add_file(filenameS,  "root");
       else if(extension == "npy")
           mFileSingles.add_file(filenameS, "npy");
       else if(extension == "txt")
           mFileSingles.add_file(filenameS, "txt");
       else
           GateError("Unknown extension for CC actor output");

       mFileSingles.set_tree_name("Singles");
       //equivalent to Tree.Init(buffer)
       mFileSingles.write_variable("runID",&m_SinglesBuffer.runID);
       mFileSingles.write_variable("eventID", &m_SinglesBuffer.eventID);

       if(EnableTime)mFileSingles.write_variable("time",&m_SinglesBuffer.time);
       if (EnableEnergy) mFileSingles.write_variable("energy", &m_SinglesBuffer.energy);
       //Sometimes they are not worth writing No info
       if(EnableEnergyFin)mFileSingles.write_variable("energyFinal",&m_SinglesBuffer.energyFin);
       if(EnableEnergyIni) mFileSingles.write_variable("energyIni",&m_SinglesBuffer.energyIni);
       if(EnableXPosition) mFileSingles.write_variable("globalPosX", &m_SinglesBuffer.globalPosX);
       if(EnableYPosition)mFileSingles.write_variable("globalPosY", &m_SinglesBuffer.globalPosY);
       if(EnableZPosition) mFileSingles.write_variable("globalPosZ", &m_SinglesBuffer.globalPosZ);
       if(EnableXLocalPosition)mFileSingles.write_variable("localPosX", &m_SinglesBuffer.localPosX);
       if(EnableYLocalPosition)mFileSingles.write_variable("localPosY", &m_SinglesBuffer.localPosY);
       if(EnableZLocalPosition)mFileSingles.write_variable("localPosZ", &m_SinglesBuffer.localPosZ);
       //source
       if(EnableXSourcePosition)mFileSingles.write_variable("sourcePosX", &m_SinglesBuffer.sourcePosX);
       if(EnableYSourcePosition)mFileSingles.write_variable("sourcePosY", &m_SinglesBuffer.sourcePosY);
       if(EnableZSourcePosition)mFileSingles.write_variable("sourcePosZ", &m_SinglesBuffer.sourcePosZ);
       if(EnableSourceEnergy)mFileSingles.write_variable("sourceEnergy",&m_SinglesBuffer.sourceEnergy);
       if(EnableSourcePDG)mFileSingles.write_variable("sourcePDG",&m_SinglesBuffer.sourcePDG);
       //interactions
       if(EnablenCrystalConv)mFileSingles.write_variable("nCrystalConv", &m_SinglesBuffer.nCrystalConv);
       if(EnablenCrystalCompt)mFileSingles.write_variable("nCrystalCompt",&m_SinglesBuffer.nCrystalCompt);
       if(EnablenCrystalRayl)mFileSingles.write_variable("nCrystalRayl",&m_SinglesBuffer.nCrystalRayl);
       //volume identification
       mFileSingles.write_variable("layerName", m_SinglesBuffer.layerName, sizeof(m_SinglesBuffer.layerName));
        //it works only for root output
       if(EnableVolumeID)mFileSingles.write_variable("volumeID", m_SinglesBuffer.volumeID,ROOT_VOLUMEIDSIZE);


       mFileSingles.write_header();


   }
   if(mSaveCoincidencesTreeFlag){
       G4String filenameC=filename+"_Coincidences."+extension;
       if(extension == "root")
           mFileCoinc.add_file(filenameC,  "root");
       else if(extension == "npy")
           mFileCoinc.add_file(filenameC, "npy");
       else if(extension == "txt")
           mFileCoinc.add_file(filenameC, "txt");
       else
           GateError("Unknown extension for CC actor output");

       mFileCoinc.set_tree_name("Coincidences");

       mFileCoinc.write_variable("runID",&m_CoincBuffer.runID);
       mFileCoinc.write_variable("coincID",&m_CoincBuffer.coincID);
       mFileCoinc.write_variable("eventID", &m_CoincBuffer.eventID);

       if(EnableTime)mFileCoinc.write_variable("time",&m_CoincBuffer.time);
       if (EnableEnergy) mFileCoinc.write_variable("energy", &m_CoincBuffer.energy);
       //Sometimes they are not worth writing No info
       if(EnableEnergyFin)mFileCoinc.write_variable("energyFinal",&m_CoincBuffer.energyFin);
       if(EnableEnergyIni) mFileCoinc.write_variable("energyIni",&m_CoincBuffer.energyIni);
       if(EnableXPosition) mFileCoinc.write_variable("globalPosX", &m_CoincBuffer.globalPosX);
       if(EnableYPosition)mFileCoinc.write_variable("globalPosY", &m_CoincBuffer.globalPosY);
       if(EnableZPosition) mFileCoinc.write_variable("globalPosZ", &m_CoincBuffer.globalPosZ);
       if(EnableXLocalPosition)mFileCoinc.write_variable("localPosX", &m_CoincBuffer.localPosX);
       if(EnableYLocalPosition)mFileCoinc.write_variable("localPosY", &m_CoincBuffer.localPosY);
       if(EnableZLocalPosition)mFileCoinc.write_variable("localPosZ", &m_CoincBuffer.localPosZ);
       //source
       if(EnableXSourcePosition)mFileCoinc.write_variable("sourcePosX", &m_CoincBuffer.sourcePosX);
       if(EnableYSourcePosition)mFileCoinc.write_variable("sourcePosY", &m_CoincBuffer.sourcePosY);
       if(EnableZSourcePosition)mFileCoinc.write_variable("sourcePosZ", &m_CoincBuffer.sourcePosZ);
       if(EnableSourceEnergy)mFileCoinc.write_variable("sourceEnergy",&m_CoincBuffer.sourceEnergy);
       if(EnableSourcePDG)mFileCoinc.write_variable("sourcePDG",&m_CoincBuffer.sourcePDG);
       //interactions
       if(EnablenCrystalConv)mFileCoinc.write_variable("nCrystalConv", &m_CoincBuffer.nCrystalConv);
       if(EnablenCrystalCompt)mFileCoinc.write_variable("nCrystalCompt",&m_CoincBuffer.nCrystalCompt);
       if(EnablenCrystalRayl)mFileCoinc.write_variable("nCrystalRayl",&m_CoincBuffer.nCrystalRayl);
       //volume identification
       mFileCoinc.write_variable("layerName", m_CoincBuffer.layerName, sizeof(m_CoincBuffer.layerName));
         //it works only for root output
       if(EnableVolumeID)mFileCoinc.write_variable("volumeID", m_CoincBuffer.volumeID,ROOT_VOLUMEIDSIZE);

       mFileCoinc.write_header();
   }
   //Coincidence chains for CSR for example
   for(unsigned int i=0; i<m_digitizer->GetmCoincChainListSize(); i++){
       coincidenceChainNames.push_back(m_digitizer->GetCoincChain(i)->GetOutputName());
   }
   if(mSaveCoincidenceChainsTreeFlag){
       for(unsigned int i=0; i<m_digitizer->GetmCoincChainListSize(); i++){
           G4String filenameCSR=filename+"_"+coincidenceChainNames.at(i)+"."+extension;


           //GateOutputTreeFileManager tempFile;
           mVectorFileCoinChain.emplace_back(new GateOutputTreeFileManager());
           std::cout<<"size chain file manager vector ="<<mVectorFileCoinChain.size()<<std::endl;

           if(extension == "root")
               mVectorFileCoinChain.back()->add_file(filenameCSR,  "root");
           else if(extension == "npy")
               mVectorFileCoinChain.back()->add_file(filenameCSR,  "npy");
           else if(extension == "txt")
               mVectorFileCoinChain.back()->add_file(filenameCSR,  "txt");
           else
                GateError("Unknown extension for CC actor output");
           mVectorFileCoinChain.back()->set_tree_name(coincidenceChainNames.at(i));

           mVectorFileCoinChain.back()->write_variable("runID",&m_CoincBuffer.runID);
           mVectorFileCoinChain.back()->write_variable("coincID",&m_CoincBuffer.coincID);
           mVectorFileCoinChain.back()->write_variable("eventID", &m_CoincBuffer.eventID);

           if(EnableTime)mVectorFileCoinChain.back()->write_variable("time",&m_CoincBuffer.time);
           if (EnableEnergy) mVectorFileCoinChain.back()->write_variable("energy", &m_CoincBuffer.energy);
           //Sometimes they are not worth writing No info
           if(EnableEnergyFin)mVectorFileCoinChain.back()->write_variable("energyFinal",&m_CoincBuffer.energyFin);
           if(EnableEnergyIni) mVectorFileCoinChain.back()->write_variable("energyIni",&m_CoincBuffer.energyIni);
           if(EnableXPosition) mVectorFileCoinChain.back()->write_variable("globalPosX", &m_CoincBuffer.globalPosX);
           if(EnableYPosition)mVectorFileCoinChain.back()->write_variable("globalPosY", &m_CoincBuffer.globalPosY);
           if(EnableZPosition) mVectorFileCoinChain.back()->write_variable("globalPosZ", &m_CoincBuffer.globalPosZ);
           if(EnableXLocalPosition)mVectorFileCoinChain.back()->write_variable("localPosX", &m_CoincBuffer.localPosX);
           if(EnableYLocalPosition)mVectorFileCoinChain.back()->write_variable("localPosY", &m_CoincBuffer.localPosY);
           if(EnableZLocalPosition)mVectorFileCoinChain.back()->write_variable("localPosZ", &m_CoincBuffer.localPosZ);
           //source
           if(EnableXSourcePosition)mVectorFileCoinChain.back()->write_variable("sourcePosX", &m_CoincBuffer.sourcePosX);
           if(EnableYSourcePosition)mVectorFileCoinChain.back()->write_variable("sourcePosY", &m_CoincBuffer.sourcePosY);
           if(EnableZSourcePosition)mVectorFileCoinChain.back()->write_variable("sourcePosZ", &m_CoincBuffer.sourcePosZ);
           if(EnableSourceEnergy)mVectorFileCoinChain.back()->write_variable("sourceEnergy",&m_CoincBuffer.sourceEnergy);
           if(EnableSourcePDG)mVectorFileCoinChain.back()->write_variable("sourcePDG",&m_CoincBuffer.sourcePDG);
           //interactions
           if(EnablenCrystalConv)mVectorFileCoinChain.back()->write_variable("nCrystalConv", &m_CoincBuffer.nCrystalConv);
           if(EnablenCrystalCompt)mVectorFileCoinChain.back()->write_variable("nCrystalCompt",&m_CoincBuffer.nCrystalCompt);
           if(EnablenCrystalRayl)mVectorFileCoinChain.back()->write_variable("nCrystalRayl",&m_CoincBuffer.nCrystalRayl);
           //volume identification
           mVectorFileCoinChain.back()->write_variable("layerName", m_CoincBuffer.layerName, sizeof(m_CoincBuffer.layerName));
             //it works only for root output
           if(EnableVolumeID)mVectorFileCoinChain.back()->write_variable("volumeID", m_CoincBuffer.volumeID,ROOT_VOLUMEIDSIZE);

           mVectorFileCoinChain.back()->write_header();

       }
   }

  // General event Info. In this case Electron escape info
   if(mSaveEventInfoTreeFlag){
       G4String filenameC=filename+"_eventGlobalInfo."+extension;
       if(extension == "root")
           mFileEvent.add_file(filenameC,  "root");
       else if(extension == "npy")
           mFileEvent.add_file(filenameC, "npy");
       else if(extension == "txt")
           mFileEvent.add_file(filenameC, "txt");
       else
           GateError("Unknown extension for CC actor output");

       mFileEvent.set_tree_name("EventGlobalInfo");

       mFileEvent.write_variable("runID",&runID);
       mFileEvent.write_variable("eventID", &evtID);
       mFileEvent.write_variable("energyElectronEscaped", &energyElectronEscapedEvt);
       //Is exiting a volume or entering a colume
       mFileEvent.write_variable("isElectronExitingSD", &IseExitingSDVol);
       //which volume (entering or exiting)
       mFileEvent.write_variable("SDVolName", eEspVolName, sizeof(eEspVolName));


       mFileEvent.write_header();
   }






    ResetData();
    G4cout<<"end ofconstruction"<<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateComptonCameraActor::SaveData()
{
    // It seems that  Vactor calls by default  call for EndOfRunAction allowing to call Save
    GateVActor::SaveData();

    if(mSaveHitsTreeFlag){
        mFileHits.close();
    }
    if(mSaveSinglesTreeFlag){
        mFileSingles.close();
    }
    if(mSaveCoincidencesTreeFlag){
        mFileCoinc.close();
    }
    if(mSaveCoincidenceChainsTreeFlag){
        for(unsigned int i=0; i<mVectorFileCoinChain.size(); i++){
            mVectorFileCoinChain.at(i)->close();
        }
    }
    if(mSaveEventInfoTreeFlag){
        mFileEvent.close();
    }


}
//-----------------------------------------------------------------------------




//-----------------------------------------------------------------------------
void GateComptonCameraActor::ResetData()
{


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::BeginOfRunAction(const G4Run * run)
{
    // G4cout<<"###Start of BEGIN ACTION#############################################"<<G4endl;
    GateDebugMessage("Actor", 3, "GateComptonCameraActor -- Begin of Run\n");
    ResetData();
    runID=run->GetRunID();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::BeginOfEventAction(const G4Event* evt)
{

    G4PrimaryVertex* pvertex=evt->GetPrimaryVertex();
    //The posiiton is related to the parentID=0 even if we set a differnt value for primary. The enrgy and PDGEncoding are wel set.
    sourcePos=pvertex->GetPosition();

    GateDebugMessage("Actor", 3, "GateComptonCameraActor -- Begin of Event\n");
    // G4cout<<"######STARTA OF :begin OF EVENT ACTION####################################"<<G4endl;
    newEvt = true;
    edepEvt = 0.;
    Ef_oldPrimary=0;

    sourceEnergy=-1;



    while (hitsList.size()) {
        delete hitsList.back();
        hitsList.erase(hitsList.end()-1);
    }

    if(mSaveHitsTreeFlag){
        m_HitsBuffer.Clear();
    }

    evtID = evt->GetEventID();

    nCrystalConv=0;
    nCrystalRayl=0;
    nCrystalCompt=0;
    //nCrystalCompt_posZ.clear();
    //nCrystalConv_posZ.clear();
    nCrystalCompt_gTime.clear();
    nCrystalConv_gTime.clear();
    nCrystalRayl_gTime.clear();

    energyElectronEscapedEvt=0.0;
    IseExitingSDVol=false;
    eEspVolName="NULL";


    //std::cout<<"eventID="<<evtID<<std::endl;

    //G4cout<<"###### Begin OF EVENT ACTION################################-----eventID="<<evt->GetEventID()<<G4endl;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::EndOfEventAction(const G4Event* )
{
    //G4cout<<"######:END OF EVENT ACTION#################################### "  <<G4endl;
    GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Event\n");
    if (edepEvt > 0) {
        // if(!crystalCollection)G4cout<<"problems with crystalCollection  pointer"<<G4endl;
        if(hitsList.size()>0){
            m_digitizer->Digitize(hitsList);
            processPulsesIntoSinglesTree();
            std::vector<GateCoincidencePulse*> coincidencePulseV =m_digitizer->FindCoincidencePulse(thedigitizerSorterName);
            //// std::vector<std::shared_ptr<GateCoincidencePulse>> coincidencePulseV = m_digitizer->FindCoincidencePulse(thedigitizerSorterName);
            if(coincidencePulseV.size()>0){
                //std::cout<<" coinc Pulse Size "<<coincidencePulseV.size()<<std::endl;
                for (std::vector<GateCoincidencePulse*>::const_iterator it = coincidencePulseV.begin();it != coincidencePulseV.end() ; ++it){
                    unsigned int numCoincPulses=(*it)->size();
                    for(unsigned int i=0;i<numCoincPulses;i++){
                        GateCCCoincidenceDigi* aCoinDigi=new GateCCCoincidenceDigi((*it)->at(i), (*it)->GetCoincID());

                        if(mSaveCoincidencesTreeFlag){
                            m_CoincBuffer.Fill(aCoinDigi);
                            mFileCoinc.fill();
                            m_CoincBuffer.Clear();
                        }
                        if(aCoinDigi){
                            delete aCoinDigi;
                            aCoinDigi=0;
                        }
                    }


                }
                while (coincidencePulseV.size()) {
                    coincidencePulseV.erase(coincidencePulseV.end()-1);
                }

            }
            //Find processed coincidences after applying a coincidenceChain
            for(unsigned int iChain=0; iChain<m_digitizer->GetmCoincChainListSize(); iChain++){
                // std::vector<std::shared_ptr<GateCoincidencePulse>> coincidencePulseChain = m_digitizer->FindCoincidencePulse(m_digitizer->GetCoincChain(iChain)->GetOutputName());
                std::vector<GateCoincidencePulse*> coincidencePulseChain = m_digitizer->FindCoincidencePulse(m_digitizer->GetCoincChain(iChain)->GetOutputName());

                if(coincidencePulseChain.size()>0){
                    for (std::vector<GateCoincidencePulse*>::const_iterator it = coincidencePulseChain.begin();it < coincidencePulseChain.end() ; ++it){
                        GateCoincidencePulse* coincPulse=*it;
                        unsigned int numCoincPulses=coincPulse->size();

                        for(unsigned int i=0;i<numCoincPulses;i++){
                            //G4cout<<"coincID of chain"<<m_coincIDChain.at(iChain)<<G4endl;

                            //GateCCCoincidenceDigi* aCoinDigi=new GateCCCoincidenceDigi(coincPulse->at(i),m_coincIDChain.at(iChain));
                            GateCCCoincidenceDigi* aCoinDigi=new GateCCCoincidenceDigi(coincPulse->at(i),coincPulse->GetCoincID());


                            if(mSaveCoincidenceChainsTreeFlag){
                                m_CoincBuffer.Fill(aCoinDigi);
                                mVectorFileCoinChain.at(iChain)->fill();
                                m_CoincBuffer.Clear();
                            }
                            if(aCoinDigi){
                                delete aCoinDigi;
                                aCoinDigi=0;
                            }


                        }


                    }


                }




            }

        }
        else{
            G4cout<<"#########################################"<<G4endl;
            G4cout<<"no hits with energy deposition in layer collection ="<<edepEvt<<G4endl;
            G4cout<<"######################################"<<G4endl;
        }

    }




    //G4cout<<"######END OF :END OF EVENT ACTION####################################"<<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::PreUserTrackingAction(const GateVVolume * , const G4Track* t)
{
    //G4cout<<"#####BEGIN OF :PreuserTrackingAction####################################"<<G4endl;
    GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Track\n");
    newTrack = true;

    //PrimarySetInfo to secondaires.
    if(t->GetUserInformation()==0){
        //real primary not UserInfoSet.  This is equivalent to do it for tracks with parentID=0
        GatePrimTrackInformation* trackInfo1 = new  GatePrimTrackInformation(t);
        // G4cout<<"setInfo from this track"<<G4endl;
        trackInfo1->SetEPrimTrackInformation(t);
        t->SetUserInformation(trackInfo1);
    }
    else{
        if(mParentIDSpecificationFlag){
            itPrtID = find ( specfParentID.begin(),  specfParentID.end(), t->GetParentID());
            if (itPrtID !=  specfParentID.end()){
                //We Need to change the info
                ((GatePrimTrackInformation*)(t->GetUserInformation()))->SetEPrimTrackInformation(t);
                //G4cout<<"<modifying tracks info"<<G4endl;

            }
        }

    }



    // G4cout<<"eventID="<<evtID<<G4endl;
    //  G4cout<<"CCActor::PreUserTrackingAction: energy from trackInfo "<<((GatePrimTrackInformation*)t->GetUserInformation())->GetSourceEini()<<G4endl;
    // G4cout<<"CCActor::PreUserTrackingAction: PDG from trackInfo "<<((GatePrimTrackInformation*)t->GetUserInformation())->GetSourcePDG()<<G4endl;
    //   G4cout<<"CCActor::PreUserTrackingAction: total energy  from track "<<t->GetTotalEnergy()<<G4endl;
    //  G4cout<<"CCActor::PreUserTrackingAction: track parentID  "<<t->GetParentID() <<G4endl;
    //  G4cout<<"CCActor::PreUserTrackingAction: track PDGEncoding  "<<t->GetParticleDefinition()->GetPDGEncoding() <<G4endl;

    edepTrack = 0.;
    //G4cout<<"######END OF :PreuserTrackingAction####################################"<<G4endl;
}
//-----------------------------------------------------------------------------


void GateComptonCameraActor::PostUserTrackingAction(const GateVVolume *, const G4Track * atrack){

    // check memory
    const G4TrackVector* secondaries=atrack->GetStep()->GetSecondary();

    if(secondaries)
    {

        size_t nSecon = secondaries->size();
        if(nSecon>0){
            //G4cout<<"CCActor:PostUserTracking: energy value in the info of secondary tracks= "<<info.GetSourceEini()<<G4endl;
            GatePrimTrackInformation* infoNew;
            for(size_t i=0;i<nSecon;i++){


                /*if((*secondaries)[i]->GetParentID()==0){
                       G4cout<<"#-----nsecondaries="<<nSecon<<G4endl;
                       G4cout<<"#-----PostUserTracking for tracks with parentID="<<(*secondaries)[i]->GetParentID()<<"-------#"<<G4endl;
                       G4cout<<"#-----PostUserTracking: Creator process="<<(*secondaries)[i]->GetCreatorProcess()->GetProcessName() <<G4endl;
                        //const G4VProcess* processTrack = aTrack->GetCreatorProcess();
                       //G4String processName = ( (processTrack != NULL) ? processTrack->GetProcessName() : G4String() ) ;
                        G4cout<<"#-----PostUserTracking: trackID"<<(*secondaries)[i]->GetTrackID()<<G4endl;
                         G4cout<<"#-----PostUserTracking:PDG secon="<<(*secondaries)[i]->GetParticleDefinition()->GetPDGEncoding()<<G4endl;
                          G4cout<<"#-----PostUserTracking:position ="<<(*secondaries)[i]->GetPosition().getZ()<<G4endl;
                         G4cout<<"#- current track parentID="<<atrack->GetParentID()<<G4endl;
                          G4cout<<"#-Current (all) Number of total (prim) Compt ="<<nCrystalCompt<<G4endl;


                //}

                G4cout<<"#######################################"<<G4endl;*/
                GatePrimTrackInformation info=  *((GatePrimTrackInformation*)(atrack->GetUserInformation()));

                // std::shared_ptr<GatePrimTrackInformation> infoNew( new GatePrimTrackInformation(info));
                infoNew = new GatePrimTrackInformation(info);


                if(mParentIDSpecificationFlag){
                    G4cout<<" nComptos for secondaries. Check parentD values. Selected parentID  plus 1? To be DONE##########################################"<<G4endl;
                }
                else{
                    if((*secondaries)[i]->GetParentID()==1){//For First secondary tracks set number of comptons. Then it is inherited.
                        //En principio en la segunda interaccion primaria (ej Comp+ COnv. los secundarios de la segunda llevan un 1 en cada proceso. Mientras los secundarios de la primera solo del primero)
                        if(nCrystalCompt>0){
                          //Check it with  a vector a 3dPosVector
                          /*std::vector<double>::iterator it= find (nCrystalCompt_posZ.begin(), nCrystalCompt_posZ.end(), (*secondaries)[i]->GetPosition().getZ());
                          if (it != nCrystalCompt_posZ.end()){
                              int indexCompt=it-nCrystalCompt_posZ.begin();
                              //G4cout<<"indexCompt="<<indexCompt<<G4endl;
                              infoNew->setNCompton(indexCompt+1);

                          }*/
                          auto it =std::upper_bound(nCrystalCompt_gTime.begin(),nCrystalCompt_gTime.end(), (*secondaries)[i]->GetGlobalTime());
                          if (it != nCrystalCompt_gTime.end()){
                              int indexCompt=it-nCrystalCompt_gTime.begin();
                              infoNew->setNCompton(indexCompt);
                          }
                          else{
                             // G4cout<<"indexConv not found"<<G4endl;
                              infoNew->setNCompton(nCrystalCompt_gTime.size());
                            }
                        }
                        if(nCrystalConv>0){
                          //Meterlo en una funcion que devuelva indexConv+1
                         /* std::vector<double>::iterator it= find (nCrystalConv_posZ.begin(), nCrystalConv_posZ.end(), (*secondaries)[i]->GetPosition().getZ());
                          if (it != nCrystalConv_posZ.end()){
                              int indexConv=it-nCrystalConv_posZ.begin();
                              infoNew->setNConv(indexConv+1);

                          }*/
                          auto it =std::upper_bound(nCrystalConv_gTime.begin(),nCrystalConv_gTime.end(), (*secondaries)[i]->GetGlobalTime());
                          if (it != nCrystalConv_gTime.end()){
                              int indexConv=it-nCrystalConv_gTime.begin();
                              infoNew->setNConv(indexConv);
                          }
                          else{
                             // G4cout<<"indexConv not found"<<G4endl;
                              infoNew->setNConv(nCrystalConv_gTime.size());

                          }

                        }
                        if(nCrystalRayl>0){
                            auto it =std::upper_bound(nCrystalRayl_gTime.begin(),nCrystalRayl_gTime.end(), (*secondaries)[i]->GetGlobalTime());
                            //for(unsigned int i=0; i<nCrystalRayl_gTime.size();i++){
                            //    G4cout<<"svector rayl time"<<nCrystalRayl_gTime.at(i)<<G4endl;
                            //}
                            if (it != nCrystalRayl_gTime.end()){
                                int indexRay=it-nCrystalRayl_gTime.begin();
                                infoNew->setNRayl(indexRay);

                            }
                            else{
                                infoNew->setNRayl(nCrystalRayl_gTime.size());

                            }
                        }
                    }


                }


                // G4cout<<"PDG secon="<<(*secondaries)[i]->GetParticleDefinition()->GetPDGEncoding()<<G4endl;
                // (*secondaries)[i]->SetUserInformation(infoNew.get());
                (*secondaries)[i]->SetUserInformation(infoNew);

            }

        }
    }

}

//-----------------------------------------------------------------------------
void GateComptonCameraActor::UserSteppingAction(const GateVVolume *  , const G4Step* step)
{
    //G4cout<<"######START OF :UserSteppingAction####################################"<<G4endl;
    assert(step->GetTrack()->GetWeight() == 1.); // edep doesnt handle weight

    //======================== info of the track ==========================
    G4Track* aTrack = step->GetTrack();
    trackID      = aTrack->GetTrackID();
    parentID     = aTrack->GetParentID();
    trackLength  = aTrack->GetTrackLength();
    trackLocalTime = aTrack->GetLocalTime();

    G4String partName = aTrack->GetDefinition()->GetParticleName();
    G4int  PDGEncoding= aTrack->GetDefinition()->GetPDGEncoding();

    //============info of current step ======================================
    hitPostPos = step->GetPostStepPoint()->GetPosition()/mm;
    hitPrePos = step->GetPreStepPoint()->GetPosition()/mm;
    hitEdep=step->GetTotalEnergyDeposit()/MeV;

    //Volume name od the step tosave only the hits in SD of the layers
    G4TouchableHandle touchable=step->GetPreStepPoint()->GetTouchableHandle();
    int copyNumberStep=touchable->GetVolume(0)->GetCopyNo();
    G4String nameCurrentV=touchable->GetVolume(0)->GetName();
    unsigned extPos=nameCurrentV.rfind("_phys");
    nameCurrentV=nameCurrentV.substr(0,extPos);
    //VolNameStep=nameCurrentV;
    if(copyNumberStep!=0 && nameCurrentV!=mNameOfAbsorberSDVol){
        VolNameStep=nameCurrentV+std::to_string(copyNumberStep);
    }
    else{

        VolNameStep=nameCurrentV;
    }

    //G4cout<<VolNameStep<<G4endl;
    const G4TouchableHistory*  touchableH = (const G4TouchableHistory*)(step->GetPreStepPoint()->GetTouchable() );
    GateVolumeID volumeID(touchableH);

    hitPreLocalPos=volumeID.MoveToBottomVolumeFrame(hitPrePos);
    hitPostLocalPos=volumeID.MoveToBottomVolumeFrame(hitPostPos);
    G4ThreeVector hitMeanLocalPos=(hitPreLocalPos+hitPostLocalPos)/2;
    // std::cout << "VolumNameStep " <<VolNameStep <<"energy="<<hitEdep<<'\n';
    //========================track (step) =========================================
    const G4VProcess* processTrack = aTrack->GetCreatorProcess();
    G4String processName = ( (processTrack != NULL) ? processTrack->GetProcessName() : G4String() ) ;
    if(step->GetPostStepPoint()->GetProcessDefinedStep()!=0){
        processPostStep=step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
    }
    else{
        processPostStep="NULL";
    }
    //=================================================

    if (newEvt) {
        //      double pretof = step->GetPreStepPoint()->GetGlobalTime();
        //      double posttof = step->GetPostStepPoint()->GetGlobalTime();
        //      tof = pretof + posttof;
        //      tof /= 2;
        newEvt = false;
    }


    //  //Energy of particles
    Ef=step->GetPostStepPoint()->GetKineticEnergy();
    //     G4cout<<"Ef="<<step->GetPostStepPoint()->GetKineticEnergy()<<G4endl;
    if(newTrack){
        //G4cout<<"new track Ei="<<step->GetPreStepPoint()->GetKineticEnergy()<<G4endl;
        Ei=step->GetPreStepPoint()->GetKineticEnergy();
        newTrack=false;
    }

    //if(evtID==1741||evtID==6254 ||evtID==7647||evtID==23650||evtID==43942 ||evtID==11962){
    // if(evtID==692930||evtID==847511 ||evtID==1591796){
    //if(evtID==8134111||evtID==26175664 ||evtID==32789427||evtID==40127247){
    //if(evtID==218394||evtID==338367 ||evtID==2088191||evtID==80014851||evtID==51582910){
    //  G4cout<<"evtID="<<evtID<<"  PDGEncoding="<<PDGEncoding<<" parentID="<<parentID<<" trackID="<<trackID<<" energyDep"<<hitEdep<<"  Ei="<<Ei<<" EF="<<Ef<<"  volName="<<VolNameStep<<" posPosZ="<<hitPostPos<<"  posStepProcess="<<processPostStep<<"  trackCreator="<<processName<<"  time="<<aTrack->GetGlobalTime()<<G4endl;
    //  G4cout<<"CCActor::UserSteppingAction: sourceEnergy= "<<((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetSourceEini()<<G4endl;
    //  G4cout<<"CCActor::UserSteppingAction: sourcePDG= "<<((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetSourcePDG()<<G4endl;
    // }



    if(mParentIDSpecificationFlag){
        //itPrtID = find ( specfParentID.begin(),  specfParentID.end(), t->GetParentID());//Specific photon set to primary for ion sources
        //This info (Ei, Ef) is only for the ideal adder that for the moment does not work for ion sources. It only works for photon sources
    }
    else if (parentID==0){
        //Like that The initial energy of the primaries it is their initial energy and not the initial energy of the track Useful for AdderComptPhotIdeal
        if(Ef_oldPrimary>0)Ei=Ef_oldPrimary;
        Ef_oldPrimary=Ef;

    }


    std::vector<G4String>::iterator it= find (layerNames.begin(), layerNames.end(), VolNameStep);
    if (it != layerNames.end()){
        //Hits with preStep in sensitive volumes.

        // step ends in the  boundary and it is an electron. Here is the case in which the pre-step is in a sensitive volume
        if(step->GetPostStepPoint()->GetStepStatus() == fGeomBoundary && PDGEncoding==11){
            //This is an electron  with a preStep in sensitive volumes and the post-step in the boundary (escaping from the SD)
            energyElectronEscapedEvt=Ef/MeV;
            IseExitingSDVol=true;
            eEspVolName=VolNameStep;
            //Fill file each time that an electron exits a SD volume. There can be several entries per event.
            mFileEvent.fill();

        }


        //I add energy to the event only if it is deposited in the layers
        edepEvt += step->GetTotalEnergyDeposit();


        //Here nCompt, nRayl, nConv only for  for primaries (07/02/2021). UPDATE doc?
        int nCurrentHitCompton=0;
        int nCurrentHitConv=0;
        int nCurrentHitRayl=0;

        if(mParentIDSpecificationFlag){
            //not checked for ions
            itPrtID = find ( specfParentID.begin(),  specfParentID.end(), parentID);
            if (itPrtID !=  specfParentID.end()){
                if(processPostStep=="conv"){
                    nCrystalConv++;
                   //nCrystalConv_posZ.push_back(hitPostPos.getZ());
                    nCrystalConv_gTime.push_back(step->GetPostStepPoint()->GetGlobalTime());
                }
                else if(processPostStep=="compt") {
                    nCrystalCompt++;
                    //nCrystalCompt_posZ.push_back(hitPostPos.getZ());
                     nCrystalCompt_gTime.push_back(step->GetPostStepPoint()->GetGlobalTime());
                }
                else if (processPostStep=="Rayl"){
                    nCrystalRayl++;
                    nCrystalRayl_gTime.push_back(step->GetPostStepPoint()->GetGlobalTime());
                }
                nCurrentHitCompton=nCrystalCompt;
                nCurrentHitConv=nCrystalConv;
                nCurrentHitRayl=nCrystalRayl;
            }
            else{
                nCurrentHitCompton=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetNCompton();
                nCurrentHitConv=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetNConv();
                nCurrentHitRayl=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetNRayl();
             }
        }
        else {
            if (parentID==0){
                if(processPostStep=="conv"){
                    nCrystalConv++;
                    //nCrystalConv_posZ.push_back(hitPostPos.getZ());
                    nCrystalConv_gTime.push_back(step->GetPostStepPoint()->GetGlobalTime());
                }
                else if(processPostStep=="compt"){
                    nCrystalCompt++;
                    //nCrystalCompt_posZ.push_back(hitPostPos.getZ());
                    nCrystalCompt_gTime.push_back(step->GetPostStepPoint()->GetGlobalTime());
                }
                else if (processPostStep=="Rayl"){
                    nCrystalRayl++;
                    nCrystalRayl_gTime.push_back(step->GetPostStepPoint()->GetGlobalTime());
                }
                nCurrentHitCompton=nCrystalCompt;
                nCurrentHitConv=nCrystalConv;
                nCurrentHitRayl=nCrystalRayl;
            }
            else{
                nCurrentHitCompton=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetNCompton();
                nCurrentHitConv=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetNConv();
                nCurrentHitRayl=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetNRayl();
            }
        }

        //G4cout<< "#-------------------------------------------------------------- "<<G4endl;
        //G4cout<<" trackID= "<<trackID<<"  Currenthit nCompton="<<nCurrentHitCompton<<" nConv= "<<nCurrentHitConv<<"  nRayl="<<nCurrentHitRayl<<G4endl;
        //G4cout<< "#-------------------------------------------------------------- "<<G4endl;

        //G4cout<<"Currenthit nCompton="<<nCurrentHitCompton<<" trackID= "<<trackID<<" PDG= "<<PDGEncoding<<"  posZ="<<hitPostPos.getZ()<<G4endl;
        sourceEnergy=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetSourceEini();
        sourcePDG=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetSourcePDG();

        //if (hitEdep!=0.){
        // Create a new crystal hit (maybe better an object hit
        GateHit* aHit=new GateHit();
        //GateHit aHit;
        //std::shared_ptr<GateHit> aHit(new GateHit());
        aHit->SetEdep(hitEdep);
        aHit->SetEnergyFin(Ef);
        aHit->SetEnergyIniTrack(Ei);
        //aHit->SetNCrystalConv(nCrystalConv);
        aHit->SetNCrystalConv(nCurrentHitConv);
        //aHit->SetNCrystalCompton(nCrystalCompt);
        aHit->SetNCrystalCompton(nCurrentHitCompton);
        aHit->SetNCrystalRayleigh(nCurrentHitRayl);
        //    G4ThreeVector hitPos;
        //    hitPos.setX((hitPrePos.getX()+hitPostPos.getX())/2);
        //    hitPos.setY((hitPrePos.getY()+hitPostPos.getY())/2);
        //    hitPos.setZ((hitPrePos.getZ()+hitPostPos.getZ())/2);
        //aHit.SetGlobalPos(hitPos);
        aHit->SetGlobalPos(hitPostPos);
        //aHit.SetGlobalPos(hitPrePos);
        //aHit.SetLocalPos(hitPreLocalPos);
        aHit->SetLocalPos( hitPostLocalPos);
        //aHit.SetLocalPos(hitMeanLocalPos);
        aHit->SetTime(aTrack->GetGlobalTime());
        aHit->SetTrackID(trackID );
        aHit->SetParentID(parentID );
        aHit->SetTrackLength(trackLength );
        aHit->SetTrackLocalTime(trackLocalTime );
        aHit->SetVolumeID(volumeID);
        aHit->SetEventID(evtID);
        aHit->SetRunID(runID);
        aHit->SetPDGEncoding(PDGEncoding);
        aHit->SetPostStepProcess( processPostStep);
        aHit->SetSourceEnergy(sourceEnergy);
        aHit->SetSourcePDG(sourcePDG);
        aHit->SetSourcePosition(sourcePos);
        //track creator
        aHit->SetProcess(processName);
        aHit->SetStepLength(step->GetStepLength());
        aHit->SetMomentumDir( aTrack->GetMomentumDirection());
        //Except for this volume related information (volume, touchable, material, ...)
        //the information kept by 'track' is same as one kept by 'PostStepPoint'.
        if(hitEdep!=0. ||(parentID==0 && processPostStep!="Transportation")){
            hitsList.push_back(aHit);
            if(mSaveHitsTreeFlag){
                m_HitsBuffer.Fill(aHit,VolNameStep);
                // m_hitsBuffer.Fill(aHit.get(),VolNameStep);
                mFileHits.fill();
                m_HitsBuffer.Clear();

            }

        }
        else{
            delete aHit;
        }

    }
    else{

        //int nCurrentHitCompton=0;
       //if(parentID!=0){
         //  nCurrentHitCompton=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetNCompton();
        //}
        //G4cout<<"Not in LAYERS Currenthit nCompton="<<nCurrentHitCompton<<" trackID= "<<trackID<<" PDG= "<<PDGEncoding<<"  posZ="<<hitPostPos.getZ()<<G4endl;
        //pre-step not in a SD
        if(step->GetPostStepPoint()->GetStepStatus() == fGeomBoundary && PDGEncoding==11){
            //Is the post-step in the boundary of a SD volume (entering to he SD)
            energyElectronEscapedEvt=Ef/MeV;
            IseExitingSDVol=false;

           // eEspVolName=
            G4TouchableHandle touchable_pos=step->GetPostStepPoint()->GetTouchableHandle();
            G4String vName=touchable_pos->GetVolume(0)->GetName();
            unsigned extPos=vName.rfind("_phys");
            vName=vName.substr(0,extPos);
            int nCp_pos=touchable_pos->GetVolume(0)->GetCopyNo();
            if(nCp_pos>0 && vName!=mNameOfAbsorberSDVol){

                vName=vName+std::to_string(nCp_pos);
                //G4cout<<"!!"<<vName<<G4endl;
            }
            eEspVolName=vName;

            //If the post step volume is a SD: store info of electron entering a SD
            std::vector<G4String>::iterator it= find (layerNames.begin(), layerNames.end(), eEspVolName);
            if (it != layerNames.end()){
               //Fill file each time that an electron enters a SD volume.
                mFileEvent.fill();
            }

        }
    }

    // G4cout<<"######END OF :UserSteppingAction####################################"<<G4endl;
}
//-----------------------------------------------------------------------------



//---------------------------------------------------
void GateComptonCameraActor::readPulses(GatePulseList* pPulseList)
{
    if(!pPulseList)G4cout<<"problems with the pPulseList"<<G4endl;
    GatePulseConstIterator iterIn;
    for (iterIn = pPulseList->begin() ; iterIn != pPulseList->end() ; ++iterIn){
        GatePulse* inputPulse = *iterIn;
        G4cout << "volID"<<inputPulse->GetVolumeID()<<G4endl;
        G4cout << "energy"<< inputPulse->GetEnergy()<<G4endl;
        G4cout << " una lista eventID"<< inputPulse->GetEventID()<<G4endl;
    }
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void GateComptonCameraActor::processPulsesIntoSinglesTree()
{
    //    G4cout <<  m_digitizer->m_pulseListAliasVector.size()<<G4endl;
    //    //size of the vector related to the number of chains in out case one for Hits, another Singles other layers
    //        for(int i=0;i<m_digitizer->m_pulseListAliasVector.size();i++){
    //            G4cout <<  m_digitizer->m_pulseListAliasVector.at(i).first<<G4endl;
    //        }

    GatePulseList* pPulseList=m_digitizer->FindPulseList("layers");

    if(pPulseList){
        if(pPulseList->size()>0){
            GatePulseConstIterator iterIn;
            for (iterIn = pPulseList->begin() ; iterIn != pPulseList->end() ; ++iterIn){
                //GatePulse* inputPulse = *iterIn;
                GateDigi* aSingleDigi=new GateDigi( *iterIn);
                if(mSaveSinglesTreeFlag){

                    m_SinglesBuffer.Fill(aSingleDigi);
                    // G4cout<< " time units after Singlesbuffer="<<m_SinglesBuffer.time<<G4endl;
                    //m_SingleTree->Fill();
                    mFileSingles.fill();
                    m_SinglesBuffer.Clear();
                }

                if(aSingleDigi){
                    delete aSingleDigi;
                    aSingleDigi=0;
                }

            }
        }
    }
    else{
        //std::cout<<"No pulse list"<<std::endl;
    }
}
//----------------------------------------------------------------------------

