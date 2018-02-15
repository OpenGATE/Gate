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

const G4String GateComptonCameraActor::theCrystalCollectionName="crystalCollection";
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

  Ei = 0.;
  Ef = 0.;
  newEvt = true;
  newTrack = true;

  nTrack=0;
  edepEvt = 0.;
  slayerID=-1;
  coincID=0;

  mSaveSinglesManualTreeFlag=false;

  emcalc = new G4EmCalculator;
  m_digitizer =    GateDigitizer::GetInstance();

  //With this two lines I enable digitizer/layers chain. digitizer/Singles chain is already created in Gate.cc. within  if G4Analysis_use_general.
  //digitizer() function applied independently both chains
  chain=new GatePulseProcessorChain(m_digitizer, thedigitizerName);
  m_digitizer->StoreNewPulseProcessorChain(chain);
  //Include a coincidence sorte into the digitizer with a default coincidence window that can be changed with macro commands
  G4double coincidenceWindow = 10.* ns;
  bool IsCCSorter=1;
  coincidenceSorter = new GateCoincidenceSorter(m_digitizer,thedigitizerSorterName,coincidenceWindow,thedigitizerName,IsCCSorter);
  m_digitizer->StoreNewCoincidenceSorter(coincidenceSorter);



  pMessenger = new GateComptonCameraActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateComptonCamera() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateComptonCameraActor::~GateComptonCameraActor(){
  GateDebugMessageInc("Actor",4,"~GateComptonCameraActor() -- begin\n");
  GateDebugMessageDec("Actor",4,"~GateComptonCameraActor() -- end\n");
  delete m_digitizer;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateComptonCameraActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true); // for save every n

  nDaughterBB=mVolume->GetLogicalVolume()->GetNoDaughters();
  attachPhysVolumeName=mVolume->GetPhysicalVolumeName();

  //Arrays created to test singles outputs
  edepInEachLayerEvt=new double [nDaughterBB];
  xPos_InEachLayerEvt=new double [nDaughterBB];
  yPos_InEachLayerEvt=new double [nDaughterBB];
  zPos_InEachLayerEvt=new double [nDaughterBB];

  //Get the names of the physical volumes of the layers
  int daughter_cpN=0;
  for(unsigned int i=0; i < nDaughterBB; i++) {
    edepInEachLayerEvt[i]=0.0;
    daughter_cpN=mVolume->GetLogicalVolume()->GetDaughter(i)->GetCopyNo();

    if(daughter_cpN==0){
      layerNames.push_back(mVolume->GetLogicalVolume()->GetDaughter(i)->GetName());
    }
    else{
      layerNames.push_back( mVolume->GetLogicalVolume()->GetDaughter(i)->GetName()+std::to_string(daughter_cpN) );
    }
  }

  //############################3
  //root file
  pTfile = new TFile(mSaveFilename,"RECREATE");
  //A tree for the hits
  if(mSaveHitsTreeFlag){
    m_hitsTree=new GateCCHitTree("Hits");
    m_hitsTree->Init(m_hitsBuffer);
    m_hitsAbsTree=new GateCCHitTree("AbsorberHits");
    m_hitsAbsTree->Init(m_hitsAbsBuffer);
    m_hitsScatTree=new GateCCHitTree("ScattererHits");
    m_hitsScatTree->Init(m_hitsScatBuffer);
  }
  // singles tree
  m_SingleTree=new GateCCSingleTree("Singles");
  m_SingleTree->Init(m_SinglesBuffer);

  // coincidence tree
  m_CoincTree=new GateCCCoincTree("Coincidences");
  m_CoincTree->Init(m_CoincBuffer);

  //output for the manual singles ree
  //A tree for each layer
  if(mSaveSinglesManualTreeFlag){

      for(unsigned int i=0; i<nDaughterBB;i++){
          pSingles.emplace_back(new TTree(layerNames.at(i), "Singles tree"));
          pSingles.at(i)->Branch("edepEvt",&edepInEachLayerEvt[i],"edepEvt/D");
          pSingles.at(i)->Branch("xPosEvt",&xPos_InEachLayerEvt[i],"xPosEvt/D");
          pSingles.at(i)->Branch("yPosEvt",&yPos_InEachLayerEvt[i],"yPosEvt/D");
          pSingles.at(i)->Branch("zPosEvt",&zPos_InEachLayerEvt[i],"zPosEvt/D");
      }
      //This line does not work I do not know how to put the units in the histograms of the branch
      //pSingles2.at(i)->GetBranch("xPosEvt")->SetTitle(" xPosEvt (mm)");
  }

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateComptonCameraActor::SaveData()
{
  // It seems that  Vactor calls by default to  the endofRun y default callback for EndOfRunAction allowing to call Save
  GateVActor::SaveData();

  pTfile->Write();

  //pTfile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::EndOfEvent(G4HCofThisEvent*)
{

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::Initialize(G4HCofThisEvent* )
{
  //  In SD the hitCollection ID is stored intothe static variable HCID using  GetColectionID.
  //I can not use it since I do not have a system or sensitive volume. Returns  MFD_e actorname notfound.
  // Then in SD it add the hit collection to the G4HCofThisEvent
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::ResetData()
{
  nEvent = 0;
  //If I do  delete segementation fault
  //if(edepInEachLayerEvt) delete[edepInEachLayerEvt;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::BeginOfRunAction(const G4Run * run)
{
  //G4cout<<"###Start of BEGIN ACTION#############################################"<<G4endl;
  GateDebugMessage("Actor", 3, "GateComptonCameraActor -- Begin of Run\n");
  ResetData();
  runID=run->GetRunID();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::BeginOfEventAction(const G4Event* evt)
{
  GateDebugMessage("Actor", 3, "GateComptonCameraActor -- Begin of Event\n");
 // G4cout<<"######STARTA OF :begin OF EVENT ACTION####################################"<<G4endl;
  newEvt = true;
  edepEvt = 0.;
  //edepInEachLayerEvt.assign(nDaughterBB,0.0);
  for(unsigned int i=0;i<nDaughterBB;i++){
    edepInEachLayerEvt[i]=0.0;
    xPos_InEachLayerEvt[i]=0.0;
    yPos_InEachLayerEvt[i]=0.0;
    zPos_InEachLayerEvt[i]=0.0;

  }
  if(crystalCollection){
    crystalCollection=0;\
    delete crystalCollection;
  }
  m_hitsBuffer.Clear();
  m_hitsScatBuffer.Clear();
  m_hitsAbsBuffer.Clear();

  evtID = evt->GetEventID();
  crystalCollection = new GateCrystalHitsCollection(attachPhysVolumeName,theCrystalCollectionName);

  //G4cout<<"######end OF :begin OF EVENT ACTION####################################"<<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::EndOfEventAction(const G4Event*)
{
  //G4cout<<"######start of  :END OF EVENT ACTION####################################"<<G4endl;
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Event\n");
  if (edepEvt > 0) {
    //Manually constructed singles
      if(mSaveSinglesManualTreeFlag){
          for(unsigned int i=0; i<nDaughterBB; i++){
              //I let divide by zero because it gives me null value for the position in the layer when there is no energy deposition
              if(edepInEachLayerEvt[i]!=0.){
                  xPos_InEachLayerEvt[i]=xPos_InEachLayerEvt[i]/edepInEachLayerEvt[i];
                  yPos_InEachLayerEvt[i]=yPos_InEachLayerEvt[i]/edepInEachLayerEvt[i];
                  zPos_InEachLayerEvt[i]=zPos_InEachLayerEvt[i]/edepInEachLayerEvt[i];
                  // }
                  pSingles.at(i)->Fill();
              }
          }
      }
    if(!crystalCollection)G4cout<<"problems with crystalCollection  pointer"<<G4endl;
    G4cout<<"entries of CC before digitizer "<<crystalCollection->entries()<< "  edep  "<<edepEvt <<G4endl;
    if(crystalCollection->entries()>0){
        m_digitizer->Digitize(crystalCollection);
        processPulsesIntoSinglesTree();
        G4cout<<"coincidenceVectorPulse size="<<m_digitizer->FindCoincidencePulse(thedigitizerSorterName).size()<<G4endl;
        if(m_digitizer->FindCoincidencePulse(thedigitizerSorterName).size()==1){

            //Here I have my coincidences (Singles fill a buffer  every event and when the size is above a THR  coincidences are processed and then we have a ocincidnece pulse output after the event which is erase at hte beginning of the next event

            GateCoincidencePulse* coincPulse=m_digitizer->FindCoincidencePulse(thedigitizerSorterName).at(0);
            //GAteCoincidencePulse compose of several GatePulse

                   unsigned int numCoincPulses=coincPulse->size();
                   for(unsigned int i=0;i<numCoincPulses;i++){
                         GateCCCoincidenceDigi* aCoinDigi=new GateCCCoincidenceDigi(coincPulse->at(i),coincID);
                         //Me falta crear su tree su buffer  y llenarlo con todos los pulsos de la coincidencia Muchos seran pares otros no
                         m_CoincBuffer.Fill(aCoinDigi);
                         m_CoincTree->Fill();
                         m_CoincBuffer.Clear();
                   }

            G4cout<<"size of the  coincidence Pulse"<<coincPulse->size()<<G4endl;
            if(coincPulse->size()>2){
                //Sometimes 3 or more pulses (singles) in a coincidence
                G4cout<<"###############################################################"<<G4endl;
                G4cout<<"number PULSES IN THE COINCIDENCEPULSE the vector ."<<coincPulse->size()<<G4endl;
                G4cout<<"###############################################################"<<G4endl;


            }


           coincID++;
        }
        else if(m_digitizer->FindCoincidencePulse(thedigitizerSorterName).size()>1){
            G4cout<<"###############################################################"<<G4endl;
            G4cout<<"More than one coincidencePulse in the vector . If I have defined only the sorter PROBLEMS"<<G4endl;
            G4cout<<"###############################################################"<<G4endl;
            //para que se salga con un break
            m_digitizer->FindCoincidencePulse(thedigitizerSorterName).at(15);

        }
    }
    else{
        G4cout<<"#########################################"<<G4endl;
        G4cout<<"no hits with energy deposition in layer collection ="<<edepEvt<<G4endl;
        G4cout<<"######################################"<<G4endl;
    }
    // crystalPulseList=GateHitConvertor::GetInstance()->ProcessHits(crystalCollection);
    // const G4String adderName="adder";
    // GatePulseList* pPulseFinalList=GatePulseAdder(chain,adderName).ProcessPulseList(crystalPulseList);
    //readPulses(pPulseFinalList);

  }
  nEvent++;
  //G4cout<<"######END OF :END OF EVENT ACTION####################################"<<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::PreUserTrackingAction(const GateVVolume * , const G4Track* t)
{
  //G4cout<<"#####BEGIN OF :PreuserTrackingAction####################################"<<G4endl;
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Track\n");
  newTrack = true;
  //AE  no hace nada con esto
  if (t->GetParentID()==1) nTrack++;
  edepTrack = 0.;
  //G4cout<<"######END OF :PreuserTrackingAction####################################"<<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::PostUserTrackingAction(const GateVVolume *, const G4Track* /*t*/)
{
  //G4cout<<"#####BEGIN OF :PostuserTrackingAction####################################"<<G4endl;
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Track\n");
  //double eloss = Ei-Ef;
  //if (edepTrack > 0)  pEdepTrack->Fill(edepTrack/MeV,t->GetWeight() );
  //G4cout<<"######END OF :PostuserTrackingAction####################################"<<G4endl;
}
//-----------------------------------------------------------------------------


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

  //Volume name od the step
  G4TouchableHandle touchable=step->GetPreStepPoint()->GetTouchableHandle();
  int copyNumberStep=touchable->GetVolume(0)->GetCopyNo();
  if(copyNumberStep==0){
    VolNameStep=touchable->GetVolume(0)->GetName();
  }
  else{
    VolNameStep=touchable->GetVolume(0)->GetName()+std::to_string(copyNumberStep);
  }

  const G4TouchableHistory*  touchableH = (const G4TouchableHistory*)(step->GetPreStepPoint()->GetTouchable() );
  GateVolumeID volumeID(touchableH);
  hitPreLocalPos=volumeID.MoveToBottomVolumeFrame(hitPrePos);
  // std::cout << "VolumNameStep " <<VolNameStep <<"energy="<<hitEdep<<'\n';
  //========================track (step) =========================================
  ///SD hit position from postStep
  ///processName from posStep GetProccessDefinedStep (this works with post and pre step)
  /// To do analyze difference trackcreatorProcess and  preStep/postStepProcessDefinedStep()
  const G4VProcess* processTrack = aTrack->GetCreatorProcess();
  // const G4VProcess* process=step->GetPreStepPoint()->GetProcessDefinedStep();
  //const G4VProcess* process=step->GetPostStepPoint()->GetProcessDefinedStep();
  G4String processName = ( (processTrack != NULL) ? processTrack->GetProcessName() : G4String() ) ;

  //=================================================

  //To check if  step ends in the  boundary
  // step->GetPostStepPoint()->GetStepStatus() == fGeomBoundary

  //Manual singles
  std::vector<G4String>::iterator it;
  it = find (layerNames.begin(), layerNames.end(), VolNameStep);
  int poslayer=-1;
  if (it != layerNames.end()){
    poslayer=std::distance(layerNames.begin(),it);
    // std::cout << "Element found in myvector: " << *it <<"posicion="<<std::distance(layerNames.begin(),it)<<'\n';
    if(hitEdep!=0.){
      edepInEachLayerEvt[poslayer]+=hitEdep;
      xPos_InEachLayerEvt[poslayer]+=hitEdep*(hitPrePos.getX()+hitPostPos.getX())/2;
      yPos_InEachLayerEvt[poslayer]+=hitEdep*(hitPrePos.getY()+hitPostPos.getY())/2;
      zPos_InEachLayerEvt[poslayer]+=hitEdep*(hitPrePos.getZ()+hitPostPos.getZ())/2;
    }
    //I add energy to the event only if it is deposited in the layers
     edepEvt += step->GetTotalEnergyDeposit();
  }


  //  edepTrack += step->GetTotalEnergyDeposit();

  //  //First hit of a new event tof (time) and for the rest ltof
  if (newEvt) {
    //      double pretof = step->GetPreStepPoint()->GetGlobalTime();
    //      double posttof = step->GetPostStepPoint()->GetGlobalTime();
    //      tof = pretof + posttof;
    //      tof /= 2;
    //      //cout << "****************** new event tof=" << pretof << "/" << posttof << "/" << tof << " edep=" << edep << endl;
    newEvt = false;
  } else {
    //      double pretof = step->GetPreStepPoint()->GetGlobalTime();
    //      double posttof = step->GetPostStepPoint()->GetGlobalTime();
    //      double ltof = pretof + posttof;
    //      ltof /= 2;
  }

  //  //Energy of particles
  //  Ef=step->GetPostStepPoint()->GetKineticEnergy();
  if(newTrack){
    //      Ei=step->GetPreStepPoint()->GetKineticEnergy();
    newTrack=false;
  }

  if (hitEdep!=0.){
    // Create a new crystal hit (maybe better an object hit
    GateCrystalHit* aHit = new GateCrystalHit();
    aHit->SetEdep(hitEdep);
    aHit->SetGlobalPos(hitPrePos);
    aHit->SetLocalPos(hitPreLocalPos);
    //step->GetPreStepPoint()->GetGlobalTime()
    /////Which time do I need to save. Prestep or the time of the track?
    /// also possiblitiy of track local time.
    /// Singles takes the time of the first hit
    /// Track global time
    aHit->SetTime(aTrack->GetGlobalTime());
    aHit->SetTrackID(trackID );
    aHit->SetParentID(parentID );
    aHit->SetTrackLength(trackLength );
    aHit->SetTrackLocalTime(trackLocalTime );
    aHit->SetVolumeID(volumeID);
    aHit->SetEventID(evtID);
    aHit->SetRunID(runID);
    aHit->SetPDGEncoding(PDGEncoding);
    //track creator
    aHit->SetProcess(processName);
    aHit->SetStepLength(step->GetStepLength());
    aHit->SetMomentumDir( aTrack->GetMomentumDirection());
    //Except for this volume related information (volume, touchable, material, ...) the information kept by 'track' is same as one kept by 'PostStepPoint'.

    //Only hits in the layer are saved
    //if(aHit->GetProcess()!="Transportation"){
    if (it != layerNames.end()){
      //Maybe I need different collection for each layer to apply different digitization chains
      crystalCollection->insert(aHit);
      //G4cout<<"inserting a hit"<<G4endl;
     // G4cout<<"layer+"<<VolNameStep<<G4endl;
      m_hitsBuffer.Fill(aHit,VolNameStep);
      m_hitsTree->Fill();
      m_hitsBuffer.Clear();

      //Test: create a separate output tree for the hit in the absorber and hits in the scatterer
      if(nDaughterBB>1){
        if(poslayer==(signed int)nDaughterBB-1){
          // Mattia does not save hits in the absoreber without track processCreator. But that depend on the cuts
          //if(aHit->GetProcess()!=G4String()){
          m_hitsAbsBuffer.Fill(aHit,VolNameStep);
          m_hitsAbsTree->Fill();
          m_hitsAbsBuffer.Clear();
          // }
        }
        else{
          m_hitsScatBuffer.Fill(aHit,VolNameStep);
          m_hitsScatTree->Fill();
          m_hitsScatBuffer.Clear();
        }
      }
      else{
        G4cout<<"problems our CC has less than two layers"<<G4endl;
      }
    }
  }
  // G4cout<<"######END OF :UserSteppingAction####################################"<<G4endl;
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateComptonCameraActor::readPulses(GatePulseList* pPulseList)
{
  if(!pPulseList)G4cout<<"problems with the pPulseList"<<G4endl;
  //size_t n_pulses = pPulseList->size();
  // G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
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
  // G4cout <<  m_digitizer->m_pulseListAliasVector.size()<<G4endl;
  //size of the vector related to the number of chains in out case one for Hits, another Singles other layers
  //     for(int i=0;i<m_digitizer->m_pulseListAliasVector.size();i++){
  //         G4cout <<  m_digitizer->m_pulseListAliasVector.at(i).first<<G4endl;

  //     }

  //Choose the pulses of the desired chain
  //GatePulseList* pPulseList=m_digitizer->FindPulseList("Singles");
  GatePulseList* pPulseList=m_digitizer->FindPulseList("layers");

  GatePulseConstIterator iterIn;
  for (iterIn = pPulseList->begin() ; iterIn != pPulseList->end() ; ++iterIn){
    GatePulse* inputPulse = *iterIn;
    GateSingleDigi* aSingleDigi=new GateSingleDigi(inputPulse);
    // G4cout << "eventID"<< inputPulse->GetEventID()<<"singleDigi evtID="<<aSingleDigi->GetEventID()<<G4endl;

    //LAYER NAMES MUST BE INTRODUCED AS INPUT IN ACTOR MESSENGER ang HERE some generalization
    //Possible layer. Physical name Not info about copy (0+world, 1-BB, 2--layers, 3-sublayer (4 example segmented crys))
    //G4cout << "vol name nivel 2"<<inputPulse->GetVolumeID().GetVolume(2)->GetName()<<G4endl;
    // G4cout << "vol ID nivel 2="<<inputPulse->GetVolumeID().GetVolume(2)->GetCopyNo()<<G4endl;
    ///This identification not good like that. It depends on the arbitrary name that I have given to the different layer.
    /// Oblige    to use those names or maybe  better    insert the chosen names for the layers with macro commands so that I can load them using the messenger
    if(inputPulse->GetVolumeID().GetVolume(2)->GetName()=="absorber_phys" && nDaughterBB>1){
    //if(inputPulse->GetVolumeID().GetVolume(2)->GetName()!="scatterer_phys"){
     // slayerID=nDaughterBB-1;
        slayerID=0;
    }
    else if(inputPulse->GetVolumeID().GetVolume(2)->GetName()=="scatterer_phys"){
      slayerID=inputPulse->GetVolumeID().GetVolume(2)->GetCopyNo()+1;
    }
    else{
      G4cout << "problems of layer identification"<<G4endl;
    }

    //if(inputPulse->GetVolumeID().GetVolume(2)->GetName()=="absorber_phys"){
    m_SinglesBuffer.Fill(aSingleDigi,slayerID);
    m_SingleTree->Fill();
    m_SinglesBuffer.Clear();
    //}
  }
}
//-----------------------------------------------------------------------------

