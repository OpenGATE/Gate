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

 //Is this needed? I put later the address
  m_hitsTree=0;
  m_SingleTree=0;
  m_CoincTree=0;


  Ei = 0.;
  Ef = 0.;
  newEvt = true;
  newTrack = true;

  nTrack=0;
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

  //Messenger load values
   pMessenger = new GateComptonCameraActorMessenger(this);

  emcalc = new G4EmCalculator;
  m_digitizer = GateDigitizer::GetInstance();

//  //With this two lines I enable digitizer/layers chain. digitizer/Singles chain is already created in Gate.cc. within  if G4Analysis_use_general.
//  //digitizer() function applied independently both chains
    chain=new GatePulseProcessorChain(m_digitizer, thedigitizerName);
    m_digitizer->StoreNewPulseProcessorChain(chain);
//  //Include a coincidence sorte into the digitizer with a default coincidence window that can be changed with macro commands
//  //G4double coincidenceWindow = 1;
   G4double coincidenceWindow = 10.* ns;
 bool IsCCSorter=1;
  coincidenceSorter = new GateCoincidenceSorter(m_digitizer,thedigitizerSorterName,coincidenceWindow,thedigitizerName,IsCCSorter);
  m_digitizer->StoreNewCoincidenceSorter(coincidenceSorter);




  GateDebugMessageDec("Actor",4,"GateComptonCamera() -- end\n");
 // G4cout<< "Singles save flag"<<mSaveSinglesTextFlag<<G4endl;
  //G4cout<< "CoincChain save txt flag"<<mSaveCoinChainsTextFlag<<G4endl;

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
  //EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);

  EnableEndOfEventAction(true); // for save every n




  if(mNumberTotScattLayers==0){//I am not using the messenger to specify number of layer and names

    // nDaughterBB=mVolume->GetLogicalVolume()->GetNoDaughters();
      //By default one absorber one scatter (scatterer all the copies that you want) but one kind
      nDaughterBB=2;
      //mVolume->GetLogicalVolume()->GetDaughter(mNumberTotScattLayers)

  }
  else{
    nDaughterBB=mNumberTotScattLayers+1;
  }
  attachPhysVolumeName=mVolume->GetPhysicalVolumeName();

  G4cout<<"GAteComptonCameraActor:Construct  numTotScatt"<<mNumberTotScattLayers<<G4endl;

  if(mNumberTotScattLayers==0){//I am not using the messenger to specify number of layer and names

      //Get the names of the physical volumes of the layers
      //
      int daughter_cpN=0;
      G4String name;
      for(unsigned int i=0; i < nDaughterBB; i++) {
        daughter_cpN=mVolume->GetLogicalVolume()->GetDaughter(i)->GetCopyNo();
         name=mVolume->GetLogicalVolume()->GetDaughter(i)->GetName();
        unsigned extPos=name.rfind("_phys");
        name=name.substr(0,extPos);

        if(daughter_cpN==0){
          layerNames.push_back(name);
          G4cout<<"layerN="<< layerNames.back()<<G4endl;
        }
        else{
          layerNames.push_back( name+std::to_string(daughter_cpN) );
           G4cout<<"layerN="<< layerNames.back()<<G4endl;
        }
      }
  }
  else{
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



  //############################3
   G4cout<<"files"<<G4endl;
  //root file
  pTfile = new TFile(mSaveFilename,"RECREATE");
  //A tree for the hits
  if(mSaveHitsTreeFlag){
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
  for(unsigned int i=0; i<m_digitizer->GetmCoincChainListSize(); i++){
      coincidenceChainNames.push_back(m_digitizer->GetCoincChain(i)->GetOutputName());
      // GateCCRootCoincBuffer coincChainB;
      // m_coincChainBuffer.push_back(coincChainB);
  }
  //m_coincChainBuffer.resize(coincidenceChainNames.size());
 if(mSaveCoincidenceChainsTreeFlag){
     for(unsigned int i=0; i<m_digitizer->GetmCoincChainListSize(); i++){
         m_coincChainTree.emplace_back(new GateCCCoincTree(coincidenceChainNames.at(i), "CoincidenceChain tree"));
         //m_coincChainTree.back()->Init(m_coincChainBuffer.at(i));
         m_coincChainTree.back()->Init(m_CoincBuffer);

     }
 }

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
  // It seems that  Vactor calls by default to  the endofRun y default callback for EndOfRunAction allowing to call Save
  GateVActor::SaveData();

  pTfile->Write();


  closeTextFiles();


  //pTfile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::ResetData()
{
  nEvent = 0;

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



    sourcePos=evt->GetPrimaryVertex()->GetPosition();

    GateDebugMessage("Actor", 3, "GateComptonCameraActor -- Begin of Event\n");
   // G4cout<<"######STARTA OF :begin OF EVENT ACTION####################################"<<G4endl;
    newEvt = true;
    edepEvt = 0.;
    Ef_oldPrimary=0;



    while (hitsList.size()) {
      delete hitsList.back();
     hitsList.erase(hitsList.end()-1);
    }


    if(mSaveHitsTreeFlag){
        m_hitsBuffer.Clear();

    }

    evtID = evt->GetEventID();
    //std::cout<<"eventID="<<evtID<<std::endl;

 // G4cout<<"######end OF :begin OF EVENT ACTION####################################"<<G4endl;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActor::EndOfEventAction(const G4Event* )
{
 // G4cout<<"######start of  :END OF EVENT ACTION####################################   "<<nEvent<<G4endl;
    GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Event\n");
    if (edepEvt > 0) {
     // if(!crystalCollection)G4cout<<"problems with crystalCollection  pointer"<<G4endl;
      if(hitsList.size()>0){
          m_digitizer->Digitize(hitsList);
            // G4cout<<"Singles"<<G4endl;
          //AQUI EMPIEZA UN NUEVO MEMORY LEAK si copio el vector!! (porqque se trat
          processPulsesIntoSinglesTree();
          // G4cout<<"findcoinc"<<G4endl;

          //Solo haciendo el find creo que ya tengo memeory leak!!!!
          //m_digitizer->FindCoincidencePulse(thedigitizerSorterName);
          std::vector<GateCoincidencePulse*> coincidencePulseV =m_digitizer->FindCoincidencePulse(thedigitizerSorterName);
         // std::vector<std::shared_ptr<GateCoincidencePulse>> coincidencePulseV = m_digitizer->FindCoincidencePulse(thedigitizerSorterName);
                  //G4cout<<"coincidenceVectorPulse size="<<coincidencePulseV.size()<<G4endl;
                 if(coincidencePulseV.size()>0){
                      //std::cout<<" coinc Pulse Size "<<coincidencePulseV.size()<<std::endl;
                      for (std::vector<GateCoincidencePulse*>::const_iterator it = coincidencePulseV.begin();it != coincidencePulseV.end() ; ++it){
                          //Here I have my coincidences (Singles fill a buffer  every event and when the size is above a THR  coincidences are processed and then we have a ocincidnece pulse output after the event which is erase at hte beginning of the next event
                          unsigned int numCoincPulses=(*it)->size();
                          for(unsigned int i=0;i<numCoincPulses;i++){
                              //coincPulse->SetCoincID(coincID);
                              //std::cout<<"evtID coinc "<<(*it)->at(i)->GetEventID()<<std::endl;
                              //GateCCCoincidenceDigi* aCoinDigi=new GateCCCoincidenceDigi(coincPulse->at(i), coincPulse->GetCoincID());
                               GateCCCoincidenceDigi* aCoinDigi=new GateCCCoincidenceDigi((*it)->at(i), (*it)->GetCoincID());
                              //Me falta crear su tree su buffer  y llenarlo con todos los pulsos de la coincidencia Muchos seran pares otros no
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
                          //Si borro el puntero se estropea todo   (PEro sino epueza mi memory leak)
                       // delete coincidencePulseV.back();
                        coincidencePulseV.erase(coincidencePulseV.end()-1);
                      }

                   }
                      //Find sequence coincidences or other coincidences after applying coincidenceChain If any coincidenceChain has been applied
                     for(unsigned int iChain=0; iChain<m_digitizer->GetmCoincChainListSize(); iChain++){
                         // G4cout<<"list size"<<m_digitizer->GetmCoincChainListSize()<<G4endl;

                             // std::vector<std::shared_ptr<GateCoincidencePulse>> coincidencePulseChain = m_digitizer->FindCoincidencePulse(m_digitizer->GetCoincChain(iChain)->GetOutputName());
                              std::vector<GateCoincidencePulse*> coincidencePulseChain = m_digitizer->FindCoincidencePulse(m_digitizer->GetCoincChain(iChain)->GetOutputName());

                              //G4cout<<"Find coinc in digitiazer for output"<<G4endl;
                               // G4cout<<"vector size "<<coincidencePulseChain.size()<<G4endl;

                                if(coincidencePulseChain.size()>0){
                                    for (std::vector<GateCoincidencePulse*>::const_iterator it = coincidencePulseChain.begin();it < coincidencePulseChain.end() ; ++it){
                                        //G4cout<<"1"<<G4endl;
                                        GateCoincidencePulse* coincPulse=*it;

                                        unsigned int numCoincPulses=coincPulse->size();
                                       // G4cout<<" coinIDchain "<<coincPulse->GetCoincID()<<G4endl;

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


                                    // G4cout<<"Size the input pCoin pulse non zero"<<G4endl;

                                }




                     }

              }
              else{
                  G4cout<<"#########################################"<<G4endl;
                  G4cout<<"no hits with energy deposition in layer collection ="<<edepEvt<<G4endl;
                  G4cout<<"######################################"<<G4endl;
              }

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
  ///SD hit position from postStep
  ///processName from posStep GetProccessDefinedStep (this works with post and pre step)
  /// To do analyze difference trackcreatorProcess and  preStep/postStepProcessDefinedStep()
  const G4VProcess* processTrack = aTrack->GetCreatorProcess();
  // const G4VProcess* process=step->GetPreStepPoint()->GetProcessDefinedStep();
  //const G4VProcess* process=step->GetPostStepPoint()->GetProcessDefinedStep();
  G4String processName = ( (processTrack != NULL) ? processTrack->GetProcessName() : G4String() ) ;
  if(step->GetPostStepPoint()->GetProcessDefinedStep()!=0){
       processPostStep=step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
  }
  else{
      processPostStep="NULL";
  }
  //=================================================

//  //check volumeID diff with crystalSD
//  std::cout << "posStepZG: " << hitPostPos.getZ() <<"  preStepZG: " << hitPrePos.getZ()<<'\n';
//  std::cout << "posSteprocess " <<processPostStep  <<  "energy: " << hitEdep<<'\n';
//  std::cout << "volume name cogido del pre  "<<  VolNameStep<<std::endl;
//  std::cout << "volume name cogido del post "<<  step->GetPostStepPoint()->GetTouchableHandle()->GetVolume(0)->GetName()<<std::endl;
//  std::cout << "volume name Hitroey pre "<<std::endl;

//  for(unsigned int i=0;i<volumeID.size(); i++){
//      std::cout<< volumeID.at(i);
//  }
// std::cout << " " << '\n';
//  std::cout << "volume name Hitroey post "<<std::endl;

//  const G4TouchableHistory*  touchableHTest = (const G4TouchableHistory*)(step->GetPostStepPoint()->GetTouchable() );
//  GateVolumeID volumeIDTest(touchableHTest);
//  for(unsigned int i=0;i<volumeIDTest.size(); i++){
//      std::cout<< volumeIDTest.at(i);
//  }
//  std::cout << " " << '\n';


  //To check if  step ends in the  boundary
  // step->GetPostStepPoint()->GetStepStatus() == fGeomBoundary

  //Manual singles
  std::vector<G4String>::iterator it;
  it = find (layerNames.begin(), layerNames.end(), VolNameStep);
  //int poslayer=-1;
  if (it != layerNames.end()){
   // poslayer=std::distance(layerNames.begin(),it);
    // std::cout << "Element found in myvector: " << *it <<"posicion="<<std::distance(layerNames.begin(),it)<<'\n';
    //I add energy to the event only if it is deposited in the layers
     edepEvt += step->GetTotalEnergyDeposit();
  }


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
    Ef=step->GetPostStepPoint()->GetKineticEnergy();
  //     G4cout<<"Ef="<<step->GetPostStepPoint()->GetKineticEnergy()<<G4endl;
  if(newTrack){
      //G4cout<<"new track Ei="<<step->GetPreStepPoint()->GetKineticEnergy()<<G4endl;
          Ei=step->GetPreStepPoint()->GetKineticEnergy();
    newTrack=false;
  }

//  if(evtID==79){

 //G4cout<<"evtID="<<evtID<<"  PDGEncoding="<<PDGEncoding<<" parentID="<<parentID<<" trackID="<<trackID<<" energyDep"<<hitEdep<<"  Ei="<<Ei<<" EF="<<Ef<<"  volName="<<VolNameStep<<" posPosZ="<<hitPostPos<<"  posStepProcess="<<processPostStep<<"  trackCreator="<<processName<<G4endl;
//  }


  if(parentID==0){
      //Like that The initial energy of the primaries it is their initial energy and not the initial energy of the track Useful for AdderComptPhotIdeal
      if(Ef_oldPrimary!=0)Ei=Ef_oldPrimary;
      Ef_oldPrimary=Ef;
  }
   if (it != layerNames.end()){
  //if (hitEdep!=0.){
    // Create a new crystal hit (maybe better an object hit
         GateCrystalHit* aHit=new GateCrystalHit();
   //GateCrystalHit aHit;
    //std::shared_ptr<GateCrystalHit> aHit(new GateCrystalHit());
    aHit->SetEdep(hitEdep);
    aHit->SetEnergyFin(Ef);
    aHit->SetEnergyIniTrack(Ei);
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
    //step.GetPreStepPoint()->GetGlobalTime()
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
    aHit->SetPostStepProcess( processPostStep);

    //Try to obtain information of the source (I DO NOT FROM WHERE SD SYS TAKE IT)
    aHit->SetSourcePosition(sourcePos);
    //track creator
    aHit->SetProcess(processName);
    aHit->SetStepLength(step->GetStepLength());
    aHit->SetMomentumDir( aTrack->GetMomentumDirection());
    //Except for this volume related information (volume, touchable, material, ...)
    //the information kept by 'track' is same as one kept by 'PostStepPoint'.

    //Only hits in the layer are saved
    //if(aHit->GetProcess()!="Transportation"){

      //Maybe I need different collection for each layer to apply different digitization chains
     if(hitEdep!=0. ||(parentID==0 && processPostStep!="Transportation")){
         hitsList.push_back(aHit);
        if(mSaveHitsTreeFlag){
            m_hitsBuffer.Fill(aHit,VolNameStep);
            //m_hitsBuffer.Fill(&aHit,VolNameStep);
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


    //  ossSingles << "# EventID" <<"    Energy (MeV) "<<"    layerName"<<std::endl;


    int copyCrys=-1;
    //if(layerN=="absorberBlocks"){
    // copyCrys=aHit->GetVolumeID().GetVolume(4)->GetCopyNo();
    copyCrys=aHit->GetVolumeID().GetBottomCreator()->GetPhysicalVolume()->GetCopyNo();

    //}


    ossHits<<"    evtID="<< aHit->GetEventID()<<"  PDG="<<aHit->GetPDGEncoding()<<" processPost"<<aHit->GetPostStepProcess()<<"  pID="<<aHit->GetParentID()<<"     E="<<aHit->GetEdep()<<"  copyCry="<<copyCrys<<"  "<<aHit->GetGlobalPos().getX()<<"    "<<aHit->GetGlobalPos().getY()<<"    "<<aHit->GetGlobalPos().getZ()<<"    "<<layerN<< '\n';


}

//-----------------------------------------------------------------------------
void GateComptonCameraActor::SaveAsTextSingleEvt(GateSingleDigi *aSin)
{


 //  ossSingles << "# EventID" <<"    Energy (MeV) "<<"    layerName"<<std::endl;


    int copyN=aSin->GetPulse().GetVolumeID().GetBottomVolume()->GetCopyNo();
    //int copyN=aSin->GetPulse().GetVolumeID().GetVolume(2)->GetCopyNo();
    std::string layerName;
    if(copyN==0){
    layerName=aSin->GetPulse().GetVolumeID().GetVolume(2)->GetName();
    }
     else{
       layerName=aSin->GetPulse().GetVolumeID().GetVolume(2)->GetName()+std::to_string(copyN);
    }



     ossSingles<<aSin->GetEventID()<<"    "<<"    "<<std::setprecision(8)<<aSin->GetTime()<<"    "<<aSin->GetEnergy()<<"    "<<aSin->GetGlobalPos().getX()<<"    "<<aSin->GetGlobalPos().getY()<<"    "<<aSin->GetGlobalPos().getZ()<<"    "<<layerName<< '\n';


}
//--------------------------------------------------------------


void GateComptonCameraActor::SaveAsTextCoincEvt(GateCCCoincidenceDigi* aCoin, std::ofstream& ossC)
{


      int copyN=aCoin->GetPulse().GetVolumeID().GetBottomVolume()->GetCopyNo();

    //int copyN=aCoin->GetPulse().GetVolumeID().GetVolume(2)->GetCopyNo();
    std::string layerName;
    if(copyN==0){
        //layerName=aCoin->GetPulse().GetVolumeID().GetVolume(2)->GetName();
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

                //LAYER NAMES MUST BE INTRODUCED AS INPUT IN ACTOR MESSENGER ang HERE some generalization
                //Possible layer. Physical name Not info about copy (0+world, 1-BB, 2--layers, 3-sublayer (4 example segmented crys))
                //G4cout << "vol name nivel 2"<<inputPulse->GetVolumeID().GetVolume(2)->GetName()<<G4endl;
                //G4cout << "PDGsummedPulse="<<inputPulse->GetPDGEncoding()<<G4endl;
                //G4cout << " processsSingles  PosZ="<<aSingleDigi->GetGlobalPos().getZ()<<G4endl;

                //rooG4cout << "PosZ="<<aSingleDigi->GetSourcePosition().getZ()<<G4endl;

                //    ///This identification not good like that. It depends on the arbitrary name that I have given to the different layer.
                //    /// Oblige    to use those names or maybe  better    insert the chosen names for the layers with macro commands so that I can load them using the messenger
                //    if(inputPulse->GetVolumeID().GetVolume(2)->GetName()=="absorber_phys" && nDaughterBB>1){
                //     // slayerID=nDaughterBB-1;
                //        slayerID=0;
                //    }
                //    else if(inputPulse->GetVolumeID().GetVolume(2)->GetName()==("scatterer_phys")){
                //      slayerID=inputPulse->GetVolumeID().GetVolume(2)->GetCopyNo()+1;
                //    }
                //    else{
                //      G4cout << "problems of layer identification"<<G4endl;
                //    }


                if(mSaveSinglesTreeFlag){
                    m_SinglesBuffer.Fill(aSingleDigi);
                    // G4cout<< " unidades del tiempo tras Singlesbuffer="<<m_SinglesBuffer.time<<G4endl;
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
        std::cout<<"No pulse list"<<std::endl;
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
