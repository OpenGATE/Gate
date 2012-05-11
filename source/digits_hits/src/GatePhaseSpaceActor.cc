/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT

/*
  \brief Class GatePhaseSpaceActor
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATESOURCEACTOR_CC
#define GATESOURCEACTOR_CC

#include "G4VProcess.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"

#include "GatePhaseSpaceActor.hh"
#include "GateMiscFunctions.hh"
#include "GateObjectStore.hh"
#include "GateIAEAHeader.h"
#include "GateIAEARecord.h"
#include "GateIAEAUtilities.h"

// --------------------------------------------------------------------
GatePhaseSpaceActor::GatePhaseSpaceActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GatePhaseSpaceActor() -- begin"<<G4endl);

  EnableXPosition = true;
  EnableYPosition = true;
  EnableZPosition = true;
  EnableEkine = true;
  EnableXDirection = true;
  EnableYDirection = true;
  EnableZDirection = true;
  EnablePartName = true;
  EnableProdVol = true;
  EnableProdProcess = true;
  EnableWeight = true;
  EnableTime = false;
  EnableMass = false;
  EnableSec = false;
  mIsFistStep = true;
  mUseVolFrame=false;
  mStoreOutPart=false;

  mFileType = " ";
  mNevent = 0; 
  pIAEARecordType = 0;
  pIAEAheader = 0;
  mFileSize = 0;
  pActorMessenger = new GatePhaseSpaceActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GatePhaseSpaceActor() -- end"<<G4endl);
}
// --------------------------------------------------------------------


// --------------------------------------------------------------------
/// Destructor 
GatePhaseSpaceActor::~GatePhaseSpaceActor() 
{
  GateDebugMessageInc("Actor",4,"~GatePhaseSpaceActor() -- begin"<<G4endl);
  delete pActorMessenger;  
  // if(pIAEAFile) fclose(pIAEAFile);
  //  pIAEAFile = 0;
  free(pIAEAheader);
  free(pIAEARecordType);
  pIAEAheader = 0;
  pIAEARecordType = 0;
  GateDebugMessageDec("Actor",4,"~GatePhaseSpaceActor() -- end"<<G4endl);
}
// --------------------------------------------------------------------

// --------------------------------------------------------------------
/// Construct
void GatePhaseSpaceActor::Construct()
{
  GateVActor::Construct();
  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);

  G4String extension = getExtension(mSaveFilename);

  if (extension == "root") mFileType = "rootFile";
  else if (extension == "IAEAphsp" || extension == "IAEAheader" ) mFileType = "IAEAFile";
  else GateError( "Unknow phase space file extension. Knowns extensions are : "
                  << G4endl << ".IAEAphsp (or IAEAheader), .root" << G4endl);

  if(mFileType == "rootFile"){
    
    pFile = new TFile(mSaveFilename,"RECREATE","ROOT file for phase space",9);
    pListeVar = new TTree("PhaseSpace","Phase space tree");

   if(GetMaxFileSize()!=0) pListeVar->SetMaxTreeSize(GetMaxFileSize());

    if(EnableEkine) pListeVar->Branch("Ekine", &e,"Ekine/F");
    if(EnableWeight) pListeVar->Branch("Weight", &w,"Weight/F");
    if(EnableTime) pListeVar->Branch("Time", &t,"Time/F");
    if(EnableMass) pListeVar->Branch("Mass", &m,"Mass/F"); // in MeV/c2
    if(EnableXPosition) pListeVar->Branch("X", &x,"X/F");
    if(EnableYPosition) pListeVar->Branch("Y", &y,"Y/F");
    if(EnableZPosition) pListeVar->Branch("Z", &z,"Z/F");
    if(EnableXDirection) pListeVar->Branch("dX", &dx,"dX/F");
    if(EnableYDirection) pListeVar->Branch("dY", &dy,"dY/F");
    if(EnableZDirection) pListeVar->Branch("dZ", &dz,"dZ/F");
    if(EnablePartName) pListeVar->Branch("ParticleName", pname ,"ParticleName/C");
    if(EnableProdVol) pListeVar->Branch("ProductionVolume", vol,"ProductionVolume/C");
    if(EnableProdProcess) pListeVar->Branch("ProductionProcess", pro,"ProductionProcess/C");

  }
  else if(mFileType == "IAEAFile"){
    pIAEAheader = (iaea_header_type *) calloc(1, sizeof(iaea_header_type));
    pIAEAheader->initialize_counters();   
    pIAEARecordType = (iaea_record_type *) calloc(1, sizeof(iaea_record_type));

    G4String IAEAFileExt   = ".IAEAphsp";
    G4String IAEAFileName  = " ";
    IAEAFileName = G4String(removeExtension(mSaveFilename));
 
    pIAEARecordType->p_file = open_file(const_cast<char*>(IAEAFileName.c_str()), const_cast<char*>(IAEAFileExt.c_str()),(char*)"wb");

    if(pIAEARecordType->p_file == NULL) GateError("File "<<IAEAFileName<<IAEAFileExt<<" not opened.");
    if(pIAEARecordType->initialize() != OK) GateError("File "<<IAEAFileName<<IAEAFileExt<<" not initialized.");
 
    if(EnableXPosition) pIAEARecordType->ix = 1;
    if(EnableYPosition) pIAEARecordType->iy = 1;
    if(EnableZPosition) pIAEARecordType->iz = 1;
    if(EnableXDirection) pIAEARecordType->iu = 1;
    if(EnableYDirection) pIAEARecordType->iv = 1;
    if(EnableZDirection) pIAEARecordType->iw = 1;
    if(EnableWeight) pIAEARecordType->iweight = 1;
    if(EnableTime){GateWarning("'Time' is not available in IAEA phase space.");}
    if(EnableMass){GateWarning("'Mass' is not available in IAEA phase space.");}
    if( pIAEAheader->set_record_contents(pIAEARecordType) == FAIL) GateError("Record contents not setted.");
  }
}
// --------------------------------------------------------------------


// --------------------------------------------------------------------
void GatePhaseSpaceActor::PreUserTrackingAction(const GateVVolume * /*v*/, const G4Track* /*t*/)
{
  mIsFistStep = true;
}
// --------------------------------------------------------------------


// --------------------------------------------------------------------
//void GatePhaseSpaceActor::BeginOfEventAction(const G4Event * e) { 
//  mNevent++;
//}
// --------------------------------------------------------------------


// --------------------------------------------------------------------
void GatePhaseSpaceActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  if(!mIsFistStep) return;
  if(mIsFistStep && step->GetTrack()->GetTrackID()==1 ) mNevent++;

  G4StepPoint *stepPoint;
  if(mStoreOutPart) stepPoint = step->GetPostStepPoint();
  else stepPoint = step->GetPreStepPoint();


  G4String st = "";
  if(step->GetTrack()->GetLogicalVolumeAtVertex()) 
    st = step->GetTrack()->GetLogicalVolumeAtVertex()->GetName();
  sscanf(st.c_str(), "%s",vol);

  //if(vol!=mVolume->GetLogicalVolumeName() && mStoreOutPart) return;
  if(vol==mVolume->GetLogicalVolumeName() && !EnableSec && !mStoreOutPart) return;
  //if(!( mStoreOutPart && step->IsLastStepInVolume())) return;

  if(mStoreOutPart && step->GetTrack()->GetVolume()==step->GetTrack()->GetNextVolume())return;
  if(mStoreOutPart ){
  GateVVolume* nextVol = GateObjectStore::GetInstance()->FindVolumeCreator(step->GetTrack()->GetNextVolume());
  if(nextVol ==  mVolume)return;
  GateVVolume *parent = nextVol->GetParentVolume();
  while(parent){
    if(parent==mVolume) return;
      parent = parent->GetParentVolume();
  }
  }

  /*if(mStoreOutPart && step->GetTrack()->GetVolume()!=mVolume->GetPhysicalVolume() ){
    GateVVolume *parent = mVolume->GetParentVolume();
    while(parent){
      if(parent==mVolume) return;
      parent = parent->GetParentVolume();
    }
  }
  */

  st = step->GetTrack()->GetDefinition()->GetParticleName();
  sscanf(st.c_str(), "%s",pname); 

  G4ThreeVector localPosition = stepPoint->GetPosition();
 
  if(GetUseVolumeFrame()){
    const G4AffineTransform transformation = step->GetPreStepPoint()->GetTouchable()->GetHistory()->GetTopTransform();
    localPosition = transformation.TransformPoint(localPosition);
  }

  x = localPosition.x();
  y = localPosition.y();
  z = localPosition.z();


  // particle momentum
  // pc = sqrt(Ek^2 + 2*Ek*m_0*c^2)
  // sqrt( p*cos(Ax)^2 + p*cos(Ay)^2 + p*cos(Az)^2 ) = p

  G4ThreeVector localMomentum = stepPoint->GetMomentumDirection();

  if(GetUseVolumeFrame()){
    const G4AffineTransform transformation = step->GetPreStepPoint()->GetTouchable()->GetHistory()->GetTopTransform();
    localMomentum = transformation.TransformAxis(localMomentum);
  }

  dx = localMomentum.x();
  dy = localMomentum.y();
  dz = localMomentum.z();

  w = stepPoint->GetWeight();

  t = stepPoint->GetGlobalTime() ;
  //t = step->GetTrack()->GetProperTime() ; //tibo : which time?????
  GateDebugMessage("Actor", 4, st 
                   << " stepPoint time proper=" << G4BestUnit(stepPoint->GetProperTime(), "Time")
                   << " global=" << G4BestUnit(stepPoint->GetGlobalTime(), "Time") 
                   << " local=" << G4BestUnit(stepPoint->GetLocalTime(), "Time") << G4endl);
  GateDebugMessage("Actor", 4, "trackid=" 
                   << step->GetTrack()->GetParentID() 
                   << " event="<<G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID()
                   << " run="<<G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID() << G4endl);
  GateDebugMessage("Actor", 4, "pos = " << x << " " << y  << " " << z << G4endl);
  GateDebugMessage("Actor", 4, "E = " << G4BestUnit(stepPoint->GetKineticEnergy(), "Energy") << G4endl);

  e = stepPoint->GetKineticEnergy();

  m = step->GetTrack()->GetDefinition()->GetAtomicMass();
  //G4cout << st << " " << step->GetTrack()->GetDefinition()->GetAtomicMass() << " " << step->GetTrack()->GetDefinition()->GetPDGMass() << G4endl;

  st = "";
  if(step->GetTrack()->GetCreatorProcess() )
    st =  step->GetTrack()->GetCreatorProcess()->GetProcessName();
  sscanf(st.c_str(), "%s",pro); 


  if(mFileType == "rootFile"){ 
    if(GetMaxFileSize()!=0) pListeVar->SetMaxTreeSize(GetMaxFileSize());
    pListeVar->Fill();
  }
  else if(mFileType == "IAEAFile"){

    const G4Track* aTrack = step->GetTrack();
    int pdg = aTrack->GetDefinition()->GetPDGEncoding();

    if( pdg == 22) pIAEARecordType->particle = 1; // gamma
    else if( pdg == 11) pIAEARecordType->particle = 2; // electron
    else if( pdg == -11) pIAEARecordType->particle = 3; // positron
    else if( pdg == 2112) pIAEARecordType->particle = 4; // neutron
    else if( pdg == 2122) pIAEARecordType->particle = 5; // proton
    else GateError("Actor phase space: particle not available in IAEA format." );
 
    pIAEARecordType->energy = e;

    if(pIAEARecordType->ix > 0) pIAEARecordType->x = localPosition.x()/cm;
    if(pIAEARecordType->iy > 0) pIAEARecordType->y = localPosition.y()/cm; 
    if(pIAEARecordType->iz > 0) pIAEARecordType->z = localPosition.z()/cm;
  
    if(pIAEARecordType->iu > 0)  pIAEARecordType->u = localMomentum.x(); 
    if(pIAEARecordType->iv > 0)  pIAEARecordType->v = localMomentum.y(); 
    if(pIAEARecordType->iw > 0)  pIAEARecordType->w = fabs(localMomentum.z())/localMomentum.z();
 
    // G4double charge = aTrack->GetDefinition()->GetPDGCharge();

    if(pIAEARecordType->iweight > 0)  pIAEARecordType->weight = w;

    // pIAEARecordType->IsNewHistory = 0;  // not yet used
  
    pIAEARecordType->write_particle();

    pIAEAheader->update_counters(pIAEARecordType);

  }
  mIsFistStep = false;  

}
// --------------------------------------------------------------------


// --------------------------------------------------------------------
/// Save data
void GatePhaseSpaceActor::SaveData()
{
  if(mFileType == "rootFile"){
    pFile = pListeVar->GetCurrentFile();
    pFile->Write();
    pFile->Close();
  }
  else if(mFileType == "IAEAFile"){
    pIAEAheader->orig_histories = mNevent;
    G4String IAEAHeaderExt = ".IAEAheader";

    strcpy(pIAEAheader->title, "Phase space generated by GATE softawre (Geant4)");   
    
    pIAEAheader->iaea_index = 0;   
    
    G4String IAEAFileName  = " ";
    IAEAFileName = G4String(removeExtension(mSaveFilename));
    pIAEAheader->fheader = open_file(const_cast<char*>(IAEAFileName.c_str()), const_cast<char*>(IAEAHeaderExt.c_str()), (char*)"wb");
    
    if( pIAEAheader->write_header() != OK) GateError("Phase space header not writed.");

    fclose(pIAEAheader->fheader);
    fclose(pIAEARecordType->p_file);
  }
}
// --------------------------------------------------------------------


#endif /* end #define GATESOURCEACTOR_CC */
#endif /* end #define G4ANALYSIS_USE_ROOT */
