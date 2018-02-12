



#include "GateCCRootDefs.hh"
#include "GateCrystalHit.hh"
#include "GateSingleDigi.hh"





void GateCCRootHitBuffer::Clear()
{
     //G4cout<<"GATECCRootHitBuffer::clear"<<G4endl;
  PDGEncoding   = 0;
  trackID       = 0;
  parentID      = 0;
  time            = 0./s;
  trackLocalTime  = 0./s;
  edep            = 0./MeV;
  stepLength      = 0./mm;
  trackLength      = 0./mm;
  posX            = 0./mm;
  posY            = 0./mm;
  posZ            = 0./mm;
  localPosX       = 0./mm;
  localPosY       = 0./mm;
  localPosZ       = 0./mm;
  eventID         = -1;
  runID           = -1;
   strcpy (layerName, " ");
  strcpy (processName, " ");
  


}


void GateCCRootHitBuffer::Fill(GateCrystalHit* aHit,std::string layerN)
{
  //G4cout<<"GATECCRootHitBuffer::Fill"<<G4endl;

  PDGEncoding     = aHit->GetPDGEncoding();
  trackID         = aHit->GetTrackID();
  parentID        = aHit->GetParentID();
  SetTime(          aHit->GetTime() );
  SetTrackLocalTime(aHit->GetTrackLocalTime() );
  SetEdep(          aHit->GetEdep() );
  SetStepLength(    aHit->GetStepLength() );
  SetTrackLength(    aHit->GetTrackLength() );
  SetPos(           aHit->GetGlobalPos() );
  SetLocalPos(      aHit->GetLocalPos() );
  parentID        = aHit->GetParentID();
  eventID         = aHit->GetEventID();
  runID           = aHit->GetRunID();
  strcpy (processName, aHit->GetProcess().c_str());
  strcpy (layerName, layerN.c_str());


}


GateCrystalHit* GateCCRootHitBuffer::CreateHit()
{
  // Create a new hit
  GateCrystalHit* aHit = new GateCrystalHit();
// Initialise the hit data from the root-hit data
  aHit->SetPDGEncoding(PDGEncoding);\
  aHit->SetTrackID(trackID);
  aHit->SetParentID(parentID);
  aHit->SetTime(GetTime());
  aHit->SetTrackLocalTime(GetTrackLocalTime());
  aHit->SetEdep(GetEdep());
  aHit->SetStepLength(GetStepLength());
  aHit->SetTrackLength(GetTrackLength());
  aHit->SetGlobalPos(GetPos());
  aHit->SetLocalPos(GetLocalPos());
  aHit->SetEventID(eventID);
  aHit->SetRunID(runID);
  aHit->SetProcess(processName);


  return aHit;
}




void GateCCHitTree::Init(GateCCRootHitBuffer& buffer)
{
      //G4cout<<"GATECCHitTree::init"<<G4endl;
//When large Trees are produced, it is safe to activate the AutoSave
//   procedure. Some branches may have buffers holding many entries.
//   AutoSave is automatically called by TTree::Fill when the number ofbytes
//   generated since the previous AutoSave is greater than fAutoSave bytes.
  SetAutoSave(10000);
  Branch("PDGEncoding",    &buffer.PDGEncoding,"PDGEncoding/I");
  Branch("trackID",        &buffer.trackID,"trackID/I");
  Branch("parentID",       &buffer.parentID,"parentID/I");
  Branch("trackLocalTime", &buffer.trackLocalTime,"trackLocalTime/D");
  Branch("time",           &buffer.time,"time/D");
  Branch("edep",           &buffer.edep,"edep/F");
  Branch("stepLength",     &buffer.stepLength,"stepLength/F");
  Branch("trackLength",    &buffer.trackLength,"trackLength/F");
  Branch("posX",           &buffer.posX,"posX/F");
  Branch("posY",           &buffer.posY,"posY/F");
  Branch("posZ",           &buffer.posZ,"posZ/F");
  Branch("localPosX",      &buffer.localPosX,"localPosX/F");
  Branch("localPosY",      &buffer.localPosY,"localPosY/F");
  Branch("localPosZ",      &buffer.localPosZ,"localPosZ/F");
  Branch("eventID",        &buffer.eventID,"eventID/I");
  Branch("runID",          &buffer.runID,"runID/I");
  Branch("processName",    (void *)buffer.processName,"processName/C");
    Branch("layerName",     (void *)buffer.layerName,"layername/C");
 
}

void GateCCHitTree::SetBranchAddresses(TTree* hitTree,GateCCRootHitBuffer& buffer)
{
   // G4cout<<"GATECCHitTree::seBranchaddres"<<G4endl;
  // Set the addresses of the branch buffers: each buffer is a field of the root-hit structure
  hitTree->SetBranchAddress("PDGEncoding",&buffer.PDGEncoding);
  hitTree->SetBranchAddress("trackID",&buffer.trackID);
  hitTree->SetBranchAddress("parentID",&buffer.parentID);
  hitTree->SetBranchAddress("time",&buffer.time);
  hitTree->SetBranchAddress("edep",&buffer.edep);
  hitTree->SetBranchAddress("stepLength",&buffer.stepLength);
  hitTree->SetBranchAddress("posX",&buffer.posX);
  hitTree->SetBranchAddress("posY",&buffer.posY);
  hitTree->SetBranchAddress("posZ",&buffer.posZ);
  hitTree->SetBranchAddress("localPosX",&buffer.localPosX);
  hitTree->SetBranchAddress("localPosY",&buffer.localPosY);
  hitTree->SetBranchAddress("localPosZ",&buffer.localPosZ);
  hitTree->SetBranchAddress("eventID",&buffer.eventID);
  hitTree->SetBranchAddress("runID",&buffer.runID);
  hitTree->SetBranchAddress("processName",&buffer.processName);
   hitTree->SetBranchAddress("layerName",&buffer.layerName);
}




void GateCCRootSingleBuffer::Clear()
{


  runID            = -1;
  eventID          = -1;
  time             = 0./s;
  energy           = 0./MeV;
  globalPosX       = 0./mm;
  globalPosY       = 0./mm;
  globalPosZ       = 0./mm;
   strcpy (layerName, " ");
   layerID=-1;
   sublayerID=-1;

}



void GateCCRootSingleBuffer::Fill(GateSingleDigi* aDigi, int slayerID)
{


  runID         =  aDigi->GetRunID();
  eventID       =  aDigi->GetEventID();
  time          =  aDigi->GetTime()/s;
  energy        =  aDigi->GetEnergy()/MeV;
  globalPosX    = (aDigi->GetGlobalPos()).x()/mm;
  globalPosY    = (aDigi->GetGlobalPos()).y()/mm;
  globalPosZ    = (aDigi->GetGlobalPos()).z()/mm;
  layerID=slayerID;

  strcpy (layerName, aDigi->GetPulse().GetVolumeID().GetVolume(2)->GetName());
  //Not working think for segmented detectore identifier
 //sublayerID=aDigi->GetPulse().GetVolumeID().GetVolume(3)->GetCopyNo();


}



void GateCCSingleTree::Init(GateCCRootSingleBuffer& buffer)
{
  SetAutoSave(2000);
  if ( GateSingleDigi::GetSingleASCIIMask(0) )
    Branch("runID",          &buffer.runID,"runID/I");
  if ( GateSingleDigi::GetSingleASCIIMask(1) )
    Branch("eventID",        &buffer.eventID,"eventID/I");
  if ( GateSingleDigi::GetSingleASCIIMask(7) )
    Branch("time",           &buffer.time,"time/D");
  if ( GateSingleDigi::GetSingleASCIIMask(8) )
    Branch("energy",         &buffer.energy,"energy/F");
  if ( GateSingleDigi::GetSingleASCIIMask(9) )
    Branch("globalPosX",     &buffer.globalPosX,"globalPosX/F");
  if ( GateSingleDigi::GetSingleASCIIMask(10) )
    Branch("globalPosY",     &buffer.globalPosY,"globalPosY/F");
  if ( GateSingleDigi::GetSingleASCIIMask(11) )
    Branch("globalPosZ",     &buffer.globalPosZ,"globalPosZ/F");

   Branch("layerName",    (void *)buffer.layerName,"layername/C");
   Branch("layerID",     &buffer.layerID,"layerID/I");
    Branch("sublayerID",     &buffer.sublayerID,"sublayerID/I");
}


