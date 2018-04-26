/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCCRootHitBuffer
  \class  GateCCHitTree
  \class  GateCCRootSingleBuffer
  \class  GateCCSingleTree
*/

#include "GateCCRootDefs.hh"
#include "GateCrystalHit.hh"
#include "GateSingleDigi.hh"
#include "GateCCCoincidenceDigi.hh"

//-----------------------------------------------------------------------------
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

  size_t d;
  for ( d = 0 ; d < ROOT_VOLUMEIDSIZE ; ++d )
    volumeID[d] = -1;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
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
   aHit->GetVolumeID().StoreDaughterIDs(volumeID,ROOT_VOLUMEIDSIZE);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateCrystalHit* GateCCRootHitBuffer::CreateHit()
{
    GateVolumeID aVolumeID(volumeID,ROOT_VOLUMEIDSIZE);
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
   aHit->SetVolumeID(	      	aVolumeID);
  return aHit;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
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
  Branch("volumeID",       (void *)buffer.volumeID,"volumeID[10]/I");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
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
   hitTree->SetBranchAddress("volumeID",buffer.volumeID);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCRootSingleBuffer::Clear()
{
  runID            = -1;
  eventID          = -1;
  time             = 0./s;
  energy           = 0./MeV;
  globalPosX       = 0./mm;
  globalPosY       = 0./mm;
  globalPosZ       = 0./mm;
  sourcePosX       = 0./mm;
  sourcePosY       = 0./mm;
  sourcePosZ       = 0./mm;
  strcpy (layerName, " ");
  //layerID=-1;
  sublayerID=-1;
  size_t d;
  for ( d = 0 ; d < ROOT_VOLUMEIDSIZE ; ++d )
    volumeID[d] = -1;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCRootSingleBuffer::Fill(GateSingleDigi* aDigi)
{
    runID         =  aDigi->GetRunID();
    eventID       =  aDigi->GetEventID();
    time          =  aDigi->GetTime()/s;
    energy        =  aDigi->GetEnergy()/MeV;
    globalPosX    = (aDigi->GetGlobalPos()).x()/mm;
    globalPosY    = (aDigi->GetGlobalPos()).y()/mm;
    globalPosZ    = (aDigi->GetGlobalPos()).z()/mm;
    sourcePosX    = (aDigi->GetSourcePosition().getX())/mm;
    sourcePosY    = (aDigi->GetSourcePosition().getY())/mm;
    sourcePosZ    = (aDigi->GetSourcePosition().getZ())/mm;
 // layerID=slayerID;//it was as a second argument of the function.
    aDigi->GetPulse().GetVolumeID().StoreDaughterIDs(volumeID,ROOT_VOLUMEIDSIZE);



    int copyN=aDigi->GetPulse().GetVolumeID().GetVolume(2)->GetCopyNo();
    if(copyN==0){
        strcpy (layerName, aDigi->GetPulse().GetVolumeID().GetVolume(2)->GetName());
    }
    else{

        const G4String name=aDigi->GetPulse().GetVolumeID().GetVolume(2)->GetName()+std::to_string(copyN);
        strcpy (layerName,name);
    }
    //layerName is not a good thing because you are taking with the 2 only the volume ID of daughter of BB
    //With the geom and volID you can recover he layerbame

    //Not working think for segmented detectore identifier
    //sublayerID=aDigi->GetPulse().GetVolumeID().GetVolume(3)->GetCopyNo();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
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

   Branch("sourcePosX",     &buffer.sourcePosX,"sourcePosX/F");
    Branch("sourcePosY",     &buffer.sourcePosY,"sourcePosY/F");
     Branch("sourcePosZ",     &buffer.sourcePosZ,"sourcePosZ/F");

  Branch("layerName",    (void *)buffer.layerName,"layername/C");
  //Branch("layerID",     &buffer.layerID,"layerID/I");
  Branch("sublayerID",     &buffer.sublayerID,"sublayerID/I");
   Branch("volumeID",       (void *)buffer.volumeID,"volumeID[10]/I");
}
//-----------------------------------------------------------------------------



void GateCCSingleTree::SetBranchAddresses(TTree* singlesTree,GateCCRootSingleBuffer& buffer)
{

    singlesTree->SetBranchAddress("runID",&buffer.runID);
    singlesTree->SetBranchAddress("eventID",&buffer.eventID);
    singlesTree->SetBranchAddress("time",&buffer.time);
    singlesTree->SetBranchAddress("energy",&buffer.energy);
    singlesTree->SetBranchAddress("globalPosX",&buffer.globalPosX);
    singlesTree->SetBranchAddress("globalPosY",&buffer.globalPosY);
    singlesTree->SetBranchAddress("globalPosZ",&buffer.globalPosZ);
    singlesTree->SetBranchAddress("sourcePosX",&buffer.sourcePosX);
    singlesTree->SetBranchAddress("sourcePosY",&buffer.sourcePosY);
    singlesTree->SetBranchAddress("sourcePosZ",&buffer.sourcePosZ);



    singlesTree->SetBranchAddress("layerName",&buffer.layerName);
 //   singlesTree->SetBranchAddress("layerID",&buffer.layerID);
    singlesTree->SetBranchAddress("sublayerID",&buffer.sublayerID);
    singlesTree->SetBranchAddress("volumeID",buffer.volumeID);



}



GateSingleDigi* GateCCRootSingleBuffer::CreateSingle()
{


    GateVolumeID aVolumeID(volumeID,ROOT_VOLUMEIDSIZE);
    // G4cout<<"CreateSingle::tras create aVolume"<<G4endl;
  GateSingleDigi* aSingle = new GateSingleDigi();
  // Initialise the hit data from the root-hit data
  aSingle->SetRunID(runID);
  aSingle->SetEventID(eventID);

  aSingle->SetTime(GetTime());
  G4ThreeVector globalPos;
  globalPos.setX(globalPosX);
  globalPos.setY(globalPosY);
  globalPos.setZ(globalPosZ);

  aSingle->SetGlobalPos(globalPos);

  G4ThreeVector sPos;
  sPos.setX(sourcePosX);
  sPos.setY(sourcePosY);
  sPos.setZ(sourcePosZ);
  aSingle->SetSourcePosition(sPos);



  aSingle->SetEnergy(energy);
  //G4cout<<"antes del setvolID to the aSingle"<<G4endl;
  aSingle->SetVolumeID(aVolumeID);

  //Set the paremeters

  return aSingle;
}
//-----------------------------------------------------------------------------
void GateCCRootCoincBuffer::Clear()
{
  coincID          =-1;
  runID            = -1;
  eventID          = -1;
  time             = 0./s;
  energy           = 0./MeV;
  globalPosX       = 0./mm;
  globalPosY       = 0./mm;
  globalPosZ       = 0./mm;
  strcpy (layerName, " ");
  //layerID=-1;
  sublayerID=-1;
  size_t d;
  for ( d = 0 ; d < ROOT_VOLUMEIDSIZE ; ++d )
    volumeID[d] = -1;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCRootCoincBuffer::Fill(GateCCCoincidenceDigi* aDigi)
{
    coincID       = aDigi->GetCoincidenceID();
    runID         =  aDigi->GetRunID();
    eventID       =  aDigi->GetEventID();
    time          =  aDigi->GetTime()/s;
    energy        =  aDigi->GetEnergy()/MeV;
    energyFin     =  aDigi->GetFinalEnergy()/MeV;
     energyIni     =  aDigi->GetIniEnergy()/MeV;
    globalPosX    = (aDigi->GetGlobalPos()).x()/mm;
    globalPosY    = (aDigi->GetGlobalPos()).y()/mm;
    globalPosZ    = (aDigi->GetGlobalPos()).z()/mm;
    //layerID=slayerID;
    aDigi->GetPulse().GetVolumeID().StoreDaughterIDs(volumeID,ROOT_VOLUMEIDSIZE);

    //Tengo  problema cuando uso  el offline(xq no tengo volId e los isngles)
   int copyN=aDigi->GetPulse().GetVolumeID().GetVolume(2)->GetCopyNo();
    if(copyN==0){
        strcpy (layerName, aDigi->GetPulse().GetVolumeID().GetVolume(2)->GetName());
    }
    else{
        const G4String name=aDigi->GetPulse().GetVolumeID().GetVolume(2)->GetName()+std::to_string(copyN);
        strcpy (layerName,name);
        strcpy (layerName, name );
    }


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCCoincTree::Init(GateCCRootCoincBuffer& buffer)
{
    SetAutoSave(2000);
    Branch("coincID",          &buffer.coincID,"coincID/I");
    Branch("runID",          &buffer.runID,"runID/I");
    Branch("eventID",        &buffer.eventID,"eventID/I");
    Branch("time",           &buffer.time,"time/D");
    Branch("energy",         &buffer.energy,"energy/F");
     Branch("energyFinal",         &buffer.energyFin,"energyFinal/F");
      Branch("energyIni",         &buffer.energyIni,"energyIni/F");
    Branch("globalPosX",     &buffer.globalPosX,"globalPosX/F");
    Branch("globalPosY",     &buffer.globalPosY,"globalPosY/F");
    Branch("globalPosZ",     &buffer.globalPosZ,"globalPosZ/F");
     Branch("layerName",    (void *)buffer.layerName,"layername/C");
    //Branch("layerID",     &buffer.layerID,"layerID/I");
    Branch("sublayerID",     &buffer.sublayerID,"sublayerID/I");
     Branch("volumeID",       (void *)buffer.volumeID,"volumeID[10]/I");
}
//-----------------------------------------------------------------------------

void GateCCCoincTree::SetBranchAddresses(TTree* coinTree,GateCCRootCoincBuffer& buffer)
{


     coinTree->SetBranchAddress("coincID",&buffer.coincID);
    coinTree->SetBranchAddress("runID",&buffer.runID);
    coinTree->SetBranchAddress("eventID",&buffer.eventID);
    coinTree->SetBranchAddress("time",&buffer.time);
    coinTree->SetBranchAddress("energy",&buffer.energy);
    coinTree->SetBranchAddress("energyFinal",&buffer.energyFin);
     coinTree->SetBranchAddress("energyIni",&buffer.energyIni);
    coinTree->SetBranchAddress("globalPosX",&buffer.globalPosX);
    coinTree->SetBranchAddress("globalPosY",&buffer.globalPosY);
    coinTree->SetBranchAddress("globalPosZ",&buffer.globalPosZ);



    coinTree->SetBranchAddress("layerName",&buffer.layerName);
    //coinTree->SetBranchAddress("layerID",&buffer.layerID);
    coinTree->SetBranchAddress("sublayerID",&buffer.sublayerID);
    coinTree->SetBranchAddress("volumeID",buffer.volumeID);



}
