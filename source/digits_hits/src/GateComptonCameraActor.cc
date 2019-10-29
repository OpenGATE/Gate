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
#include "GateSingleDigi.hh"








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

    m_hitsTree=0;
    m_SingleTree=0;
    m_CoincTree=0;


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
    mSaveSinglesTextFlag=false;
    mSaveCoincTextFlag=false;
    mSaveCoinChainsTextFlag=false;
    mParentIDSpecificationFlag=false;

    //Messenger load values
    pMessenger = new GateComptonCameraActorMessenger(this);

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
    coincidenceSorter = new GateCoincidenceSorter(m_digitizer,thedigitizerSorterName,coincidenceWindow,thedigitizerName,IsCCSorter);
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


    //To select primary particles for example in ion sources. Otherwiser the associated to the primary particle is set to the particle with parentID=0
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

        // G4cout<<"vector parentID size"<< specfParentID.size()<<G4endl;
        //for(unsigned int i=0; i<specfParentID.size(); i++){
        //     G4cout<<"value="<< specfParentID.at(i)<<G4endl;
        // }
    }

    //##################################################################################3
    //root files
    pTfile = new TFile(mSaveFilename,"RECREATE");
    //A tree for the hits
    if(mSaveHitsTreeFlag){
        //hits tree
        m_hitsTree=new GateCCHitTree("Hits");
        m_hitsTree->Init(m_hitsBuffer);
    }
    if(mSaveSinglesTreeFlag){
        // singles tree
        m_SingleTree=new GateCCSingleTree("Singles");
        m_SingleTree->Init(m_SinglesBuffer);
    }
    if(mSaveCoincidencesTreeFlag){
        // coincidence tree
        m_CoincTree=new GateCCCoincTree("Coincidences");
        m_CoincTree->Init(m_CoincBuffer);
    }
    //For the processed coincidence there can be multiple chains. Find how many
    for(unsigned int i=0; i<m_digitizer->GetmCoincChainListSize(); i++){
        coincidenceChainNames.push_back(m_digitizer->GetCoincChain(i)->GetOutputName());
    }
    if(mSaveCoincidenceChainsTreeFlag){
        for(unsigned int i=0; i<m_digitizer->GetmCoincChainListSize(); i++){
            m_coincChainTree.emplace_back(new GateCCCoincTree(coincidenceChainNames.at(i), "CoincidenceChain tree"));
            //m_coincChainTree.back()->Init(m_coincChainBuffer.at(i));
            m_coincChainTree.back()->Init(m_CoincBuffer);

        }
    }

    //Text files
    if(mSaveHitsTextFlag){
        OpenTextFile(mSaveFilename,"Hits", ossHits);
    }
    if(mSaveSinglesTextFlag){
        OpenTextFile(mSaveFilename,"Singles", ossSingles);
    }
    if(mSaveCoincTextFlag){
        OpenTextFile(mSaveFilename,"Coincidences", ossCoincidences);
    }
    if(mSaveCoinChainsTextFlag){
        OpenTextFile(mSaveFilename, coincidenceChainNames, ossCoincidenceChains);
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
    pTfile->Write();
    closeTextFiles();

    //pTfile->Close();
}
//-----------------------------------------------------------------------------




//-----------------------------------------------------------------------------
void GateComptonCameraActor::ResetData()
{
    //nEvent = 0;

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
        m_hitsBuffer.Clear();
    }

    evtID = evt->GetEventID();

    nCrystalConv=0;
    nCrystalRayl=0;
    nCrystalCompt=0;
    //std::cout<<"eventID="<<evtID<<std::endl;

    //G4cout<<"######end OF :begin OF EVENT ACTION####################################"<<G4endl;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::EndOfEventAction(const G4Event* )
{
    //G4cout<<"######start of  :END OF EVENT ACTION####################################   "<<nEvent<<G4endl;
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
                        if(mSaveCoincTextFlag){
                            SaveAsTextCoincEvt(aCoinDigi, ossCoincidences);
                        }
                        if(mSaveCoincidencesTreeFlag){
                            m_CoincBuffer.Fill(aCoinDigi);
                            m_CoincTree->Fill();
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

                            if(mSaveCoinChainsTextFlag){
                                SaveAsTextCoincEvt(aCoinDigi, *(ossCoincidenceChains.at(iChain)));
                            }
                            if(mSaveCoincidenceChainsTreeFlag){
                                //m_coincChainBuffer.at(i).Fill(aCoinDigi);
                                m_CoincBuffer.Fill(aCoinDigi);
                                m_coincChainTree.at(iChain)->Fill();
                                //m_coincChainBuffer.at(i).Clear();
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

    //nEvent++;


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
            GatePrimTrackInformation info=  *((GatePrimTrackInformation*)(atrack->GetUserInformation()));
            //G4cout<<"CCActor:PostUserTracking: energy value in the info of secondary tracks= "<<info.GetSourceEini()<<G4endl;
            GatePrimTrackInformation* infoNew;
            for(size_t i=0;i<nSecon;i++){
                // std::shared_ptr<GatePrimTrackInformation> infoNew( new GatePrimTrackInformation(info));
                infoNew = new GatePrimTrackInformation(info);
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
    //To check if  step ends in the  boundary
    // step->GetPostStepPoint()->GetStepStatus() == fGeomBoundary

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



    //specfParentID.size()=!=0 (parentId not null, we have ions,  )
    if(mParentIDSpecificationFlag){
        //itPrtID = find ( specfParentID.begin(),  specfParentID.end(), t->GetParentID());
        //Ei and Ef only set for parentID=0. Whne primaries are considered other particles it should b included here
        //This info is only for the ideal adder that for the moment does not work for ion sources
    }
    else if (parentID==0){
        //Like that The initial energy of the primaries it is their initial energy and not the initial energy of the track Useful for AdderComptPhotIdeal
        if(Ef_oldPrimary!=0)Ei=Ef_oldPrimary;
        Ef_oldPrimary=Ef;

    }


    std::vector<G4String>::iterator it= find (layerNames.begin(), layerNames.end(), VolNameStep);
    if (it != layerNames.end()){
        //I add energy to the event only if it is deposited in the layers
        edepEvt += step->GetTotalEnergyDeposit();
        if(processPostStep=="conv")  nCrystalConv++;
        else if(processPostStep=="compt") nCrystalCompt++;
        else if (processPostStep=="rayl")nCrystalRayl++;


        sourceEnergy=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetSourceEini();
        sourcePDG=((GatePrimTrackInformation*)(aTrack->GetUserInformation()))->GetSourcePDG();

        //if (hitEdep!=0.){
        // Create a new crystal hit (maybe better an object hit
        GateCrystalHit* aHit=new GateCrystalHit();
        //GateCrystalHit aHit;
        //std::shared_ptr<GateCrystalHit> aHit(new GateCrystalHit());
        aHit->SetEdep(hitEdep);
        aHit->SetEnergyFin(Ef);
        aHit->SetEnergyIniTrack(Ei);
        aHit->SetNCrystalConv(nCrystalConv);
        aHit->SetNCrystalCompton(nCrystalCompt);
        aHit->SetNCrystalRayleigh(nCrystalRayl);
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
                m_hitsBuffer.Fill(aHit,VolNameStep);
                // m_hitsBuffer.Fill(aHit.get(),VolNameStep);
                m_hitsTree->Fill();
                m_hitsBuffer.Clear();

            }
            if(mSaveHitsTextFlag){
                //SaveAsTextHitsEvt(&aHit,VolNameStep);
                SaveAsTextHitsEvt(aHit,VolNameStep);
            }
        }
        else{
            delete aHit;
        }

    }

    // G4cout<<"######END OF :UserSteppingAction####################################"<<G4endl;
}
//-----------------------------------------------------------------------------




//---------------------------------------------------------------


void GateComptonCameraActor::OpenTextFile(G4String initial_filename, G4String specificName, std::ofstream & oss){

    std::string filename = removeExtension(initial_filename);
    filename = filename + "_"+specificName+".txt";
    OpenFileOutput(filename, oss);
    G4cout<<"Open "<<filename<<G4endl;
}

void GateComptonCameraActor::OpenTextFile(G4String initial_filename, std::vector<G4String> specificN, std::vector<std::shared_ptr<std::ofstream> >  &oss){
    std::string filename = removeExtension(initial_filename);
    std::string name;
    for(unsigned int i=0; i<specificN.size(); i++){
        name=filename + "_"+specificN.at(i)+".txt";
        std::shared_ptr<std::ofstream> out(new std::ofstream);
        out->open(name.c_str());
        oss.push_back(out);
        //OpenFileOutput(filename, *oss.at(i));
        G4cout<<"Open "<<filename<<G4endl;

    }
    if(specificN.size()==0)G4cout<<"Text output asked for a coincidence Pulse processor but no processor is applied "<<filename<<G4endl;
}

//-----------------------------------------------------------------

void GateComptonCameraActor::SaveAsTextHitsEvt(GateCrystalHit* aHit, std::string layerN)
{

    int copyCrys=-1;
    copyCrys=aHit->GetVolumeID().GetBottomVolume()->GetCopyNo();
    ossHits<<"    evtID="<< aHit->GetEventID()<<"  PDG="<<aHit->GetPDGEncoding()<<" processPost="<<aHit->GetPostStepProcess()<<"  pID="<<aHit->GetParentID()<<"     edep="<<aHit->GetEdep()<<"  copyCry="<<copyCrys<<"  "<<aHit->GetGlobalPos().getX()<<"    "<<aHit->GetGlobalPos().getY()<<"    "<<aHit->GetGlobalPos().getZ()<<"    "<<layerN<< '\n';

}

//-----------------------------------------------------------------------------
void GateComptonCameraActor::SaveAsTextSingleEvt(GateSingleDigi *aSin)
{

    int copyN=aSin->GetPulse().GetVolumeID().GetBottomVolume()->GetCopyNo();
    std::string layerName;
    if(copyN==0){
        layerName=aSin->GetPulse().GetVolumeID().GetVolume(2)->GetName();
    }
    else{
        layerName=aSin->GetPulse().GetVolumeID().GetVolume(2)->GetName()+std::to_string(copyN);
    }


    ossSingles<<aSin->GetEventID()<<"    "<<"    "<<std::setprecision(8)<<aSin->GetTime()<<"    "<<aSin->GetEnergy()<<"    globalPos="<<aSin->GetGlobalPos().getX()<<"    "<<aSin->GetGlobalPos().getY()<<"    "<<aSin->GetGlobalPos().getZ()<<"    "<<"    globalPos="<<aSin->GetLocalPos().getX()<<"    "<<aSin->GetLocalPos().getY()<<"    "<<aSin->GetLocalPos().getZ()<<"    "<<layerName<< '\n';


}
//--------------------------------------------------------------


void GateComptonCameraActor::SaveAsTextCoincEvt(GateCCCoincidenceDigi* aCoin, std::ofstream& ossC)
{


    int copyN=aCoin->GetPulse().GetVolumeID().GetBottomVolume()->GetCopyNo();

    std::string layerName;
    if(copyN==0){
        layerName=aCoin->GetPulse().GetVolumeID().GetBottomCreator()->GetObjectName();
    }
    else{
        //layerName=aCoin->GetPulse().GetVolumeID().GetVolume(2)->GetName()+std::to_string(copyN);
        layerName=aCoin->GetPulse().GetVolumeID().GetBottomCreator()->GetObjectName()+std::to_string(copyN);
    }


    ossC<<aCoin->GetCoincidenceID()<<"    "<<aCoin->GetEventID()<<"    "<<std::setprecision(8)<<aCoin->GetTime()<<"    "<<aCoin->GetEnergy()<<"    "<<aCoin->GetGlobalPos().getX()<<"    "<<aCoin->GetGlobalPos().getY()<<"    "<<aCoin->GetGlobalPos().getZ()<<"    "<<layerName<< std::endl;


}
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
                GateSingleDigi* aSingleDigi=new GateSingleDigi( *iterIn);

                if(mSaveSinglesTreeFlag){
                    m_SinglesBuffer.Fill(aSingleDigi);
                    // G4cout<< " time units after Singlesbuffer="<<m_SinglesBuffer.time<<G4endl;
                    m_SingleTree->Fill();
                    m_SinglesBuffer.Clear();
                }
                if(mSaveSinglesTextFlag){
                    SaveAsTextSingleEvt(aSingleDigi);
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




void GateComptonCameraActor::closeTextFiles(){
    if(ossHits.is_open())ossHits.close();
    if(ossSingles.is_open())ossSingles.close();
    if(ossCoincidences.is_open())ossCoincidences.close();
    for(unsigned int i =0; i<ossCoincidenceChains.size(); i++){
        if(ossCoincidenceChains.at(i)->is_open())ossCoincidenceChains.at(i)->close();

    }
}
//-----------------------------------------------------------------------------
