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
#include "GateHit.hh"
#include "GateDigi.hh"
#include "GateCCCoincidenceDigi.hh"
#include "GateComptonCameraCones.hh"

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
    energyFin           = 0./MeV;
    energyIniT            = 0./MeV;
    stepLength      = 0./mm;
    trackLength      = 0./mm;
    posX            = 0./mm;
    posY            = 0./mm;
    posZ            = 0./mm;
    localPosX       = 0./mm;
    localPosY       = 0./mm;
    localPosZ       = 0./mm;
    sPosX       = 0./mm;
    sPosY       = 0./mm;
    sPosZ       = 0./mm;
    sourceEnergy      = 0./MeV;
    sourcePDG      = 0;
    nCrystalConv=0;
    nCrystalCompt=0;
    nCrystalRayl=0;

    eventID         = -1;
    runID           = -1;
    strcpy (layerName, " ");
    strcpy (processName, " ");
    strcpy (postStepProcess, " ");

    size_t d;
    for ( d = 0 ; d < ROOT_VOLUMEIDSIZE ; ++d )
        volumeID[d] = -1;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCRootHitBuffer::Fill(GateHit* aHit,std::string layerN)
{

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
    SetsourcePos(      aHit->GetSourcePosition() );
    SetSourceEnergy(aHit->GetSourceEnergy());
    SetSourcePDG(aHit->GetSourcePDG());
    SetNCrystalConv(aHit->GetNCrystalConv());
    SetNCrystalCompton(aHit->GetNCrystalCompton());
    SetNCrystalRayleigh(aHit->GetNCrystalRayleigh());
    parentID        = aHit->GetParentID();
    eventID         = aHit->GetEventID();
    runID           = aHit->GetRunID();
    strcpy (processName, aHit->GetProcess().c_str());
    strcpy (layerName, layerN.c_str());
    aHit->GetVolumeID().StoreDaughterIDs(volumeID,ROOT_VOLUMEIDSIZE);
    strcpy (postStepProcess, aHit->GetPostStepProcess().c_str());
    SetEnergyFin(aHit->GetEnergyFin());
    SetEnergyIniT(aHit->GetEnergyIniTrack());

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateHit* GateCCRootHitBuffer::CreateHit()
{
    GateVolumeID aVolumeID(volumeID,ROOT_VOLUMEIDSIZE);
    // Create a new hit
    GateHit* aHit = new GateHit();
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
    aHit->SetSourcePosition(GetsourcePos());
    aHit->SetSourceEnergy(GetSourceEnergy());
    aHit->SetSourcePDG(GetSourcePDG());
    aHit->SetNCrystalConv(GetNCrystalConv());
    aHit->SetNCrystalCompton(GetNCrystalCompton());
    aHit->SetNCrystalRayleigh(GetNCrystalRayleigh());
    aHit->SetEventID(eventID);
    aHit->SetRunID(runID);
    aHit->SetProcess(processName);
    aHit->SetVolumeID(	      	aVolumeID);
    aHit->SetPostStepProcess( postStepProcess);
    aHit->SetEnergyFin(GetEnergyFin());
    aHit->SetEnergyIniTrack(GetEnergyIniT());

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
    Branch("sourcePosX",     &buffer.sPosX,"sourcePosX/F");
    Branch("sourcePosY",     &buffer.sPosY,"sourcePosY/F");
    Branch("sourcePosZ",     &buffer.sPosZ,"sourcePosZ/F");
    Branch("eventID",        &buffer.eventID,"eventID/I");
    Branch("runID",          &buffer.runID,"runID/I");
    Branch("processName",    (void *)buffer.processName,"processName/C");
    Branch("layerName",      (void *)buffer.layerName,"layername/C");
    Branch("volumeID",       (void *)buffer.volumeID,"volumeID[10]/I");
    Branch("sourceEnergy",    &buffer.sourceEnergy,"sourceEnergy/F");
    Branch("sourcePDG",    &buffer.sourcePDG,"sourcePDG/I");
    Branch("nCrystalConv",    &buffer.nCrystalConv,"nCrystalConv/I");
    Branch("nCrystalCompt",    &buffer.nCrystalCompt,"nCrystalCompt/I");
    Branch("nCrystalRayl",    &buffer.nCrystalRayl,"nCrystalRayl/I");
    Branch("energyFinal",    &buffer.energyFin,"energyFinal/F");
    Branch("energyIniT",     &buffer.energyIniT,"energyIniT/F");
    Branch("postStepProcess", (void *)buffer.postStepProcess,"postStepProcess/C");

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
    hitTree->SetBranchAddress("sourcePosX",&buffer.sPosX);
    hitTree->SetBranchAddress("sourcePosY",&buffer.sPosY);
    hitTree->SetBranchAddress("sourcePosZ",&buffer.sPosZ);
    hitTree->SetBranchAddress("sourceEnergy",&buffer.sourceEnergy);
    hitTree->SetBranchAddress("sourcePDG",&buffer.sourcePDG);
    hitTree->SetBranchAddress("nCrystalConv",&buffer.nCrystalConv);
    hitTree->SetBranchAddress("nCrystalCompt",&buffer.nCrystalCompt);
    hitTree->SetBranchAddress("nCrystalRayl",&buffer.nCrystalRayl);
    hitTree->SetBranchAddress("eventID",&buffer.eventID);
    hitTree->SetBranchAddress("runID",&buffer.runID);
    hitTree->SetBranchAddress("processName",&buffer.processName);
    hitTree->SetBranchAddress("layerName",&buffer.layerName);
    hitTree->SetBranchAddress("volumeID",buffer.volumeID);
    hitTree->SetBranchAddress("energyFinal",&buffer.energyFin);
    hitTree->SetBranchAddress("energyIniT",&buffer.energyIniT);
    hitTree->SetBranchAddress("postStepProcess",&buffer.postStepProcess);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCRootSingleBuffer::Clear()
{
    runID            = -1;
    eventID          = -1;
    time             = 0./s;
    energy           = 0./MeV;
    energyIni           = 0./MeV;
    energyFin           = 0./MeV;
    globalPosX       = 0./mm;
    globalPosY       = 0./mm;
    globalPosZ       = 0./mm;
    localPosX       = 0./mm;
    localPosY       = 0./mm;
    localPosZ       = 0./mm;
    sourcePosX       = 0./mm;
    sourcePosY       = 0./mm;
    sourcePosZ       = 0./mm;
    sourceEnergy      = 0./MeV;
    sourcePDG        =0;
    nCrystalConv=0;
    nCrystalCompt=0;
    nCrystalRayl=0;

    strcpy (layerName, " ");
    //layerID=-1;
    sublayerID=-1;
    size_t d;
    for ( d = 0 ; d < ROOT_VOLUMEIDSIZE ; ++d )
        volumeID[d] = -1;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCRootSingleBuffer::Fill(GateDigi* aDigi)
{
    runID         =  aDigi->GetRunID();
    eventID       =  aDigi->GetEventID();
    time          =  aDigi->GetTime()/s;
    energy        =  aDigi->GetEnergy()/MeV;
    energyIni     =  aDigi->GetEnergyIniTrack()/MeV;
    energyFin     =  aDigi->GetEnergyFin()/MeV;
    localPosX    = (aDigi->GetLocalPos()).x()/mm;
    localPosY    = (aDigi->GetLocalPos()).y()/mm;
    localPosZ    = (aDigi->GetLocalPos()).z()/mm;
    globalPosX    = (aDigi->GetGlobalPos()).x()/mm;
    globalPosY    = (aDigi->GetGlobalPos()).y()/mm;
    globalPosZ    = (aDigi->GetGlobalPos()).z()/mm;
    sourcePosX    = (aDigi->GetSourcePosition().getX())/mm;
    sourcePosY    = (aDigi->GetSourcePosition().getY())/mm;
    sourcePosZ    = (aDigi->GetSourcePosition().getZ())/mm;
    sourceEnergy    =aDigi->GetSourceEnergy()/MeV;
    sourcePDG    =aDigi->GetSourcePDG();
    nCrystalConv    =aDigi->GetNCrystalConv();
    nCrystalRayl    =aDigi->GetNCrystalRayleigh();
    nCrystalCompt =aDigi->GetNCrystalCompton();
    // layerID=slayerID;//it was as a second argument of the function.
    aDigi->GetVolumeID().StoreDaughterIDs(volumeID,ROOT_VOLUMEIDSIZE);



    int copyN=aDigi->GetVolumeID().GetBottomVolume()->GetCopyNo();
    if(copyN==0){
        strcpy (layerName, aDigi->GetVolumeID().GetBottomVolume()->GetName());
    }
    else{
        const G4String name=aDigi->GetVolumeID().GetBottomVolume()->GetName()+std::to_string(copyN);
        strcpy (layerName,name);
    }
    //layerName  is not necessary  with  the geom and volID you can recover he layerbame

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCSingleTree::Init(GateCCRootSingleBuffer& buffer)
{
    SetAutoSave(2000);
    if ( GateDigi::GetSingleASCIIMask(0) )
        Branch("runID",          &buffer.runID,"runID/I");
    if ( GateDigi::GetSingleASCIIMask(1) )
        Branch("eventID",        &buffer.eventID,"eventID/I");
    if ( GateDigi::GetSingleASCIIMask(7) )
        Branch("time",           &buffer.time,"time/D");
    if ( GateDigi::GetSingleASCIIMask(8) )
        Branch("energy",         &buffer.energy,"energy/F");
    if ( GateDigi::GetSingleASCIIMask(9) )
        Branch("globalPosX",     &buffer.globalPosX,"globalPosX/F");
    if ( GateDigi::GetSingleASCIIMask(10) )
        Branch("globalPosY",     &buffer.globalPosY,"globalPosY/F");
    if ( GateDigi::GetSingleASCIIMask(11) )
        Branch("globalPosZ",     &buffer.globalPosZ,"globalPosZ/F");

    Branch("sourcePosX",     &buffer.sourcePosX,"sourcePosX/F");
    Branch("sourcePosY",     &buffer.sourcePosY,"sourcePosY/F");
    Branch("sourcePosZ",     &buffer.sourcePosZ,"sourcePosZ/F");
    Branch("sourceEnergy",     &buffer.sourceEnergy,"sourceEnergy/F");
    Branch("sourcePDG",     &buffer.sourcePDG,"sourcePDG/I");
    Branch("nCrystalConv",     &buffer.nCrystalConv,"nCrystalConv/I");
    Branch("nCrystalCompt",     &buffer.nCrystalCompt,"nCrystalCompt/I");
    Branch("nCrystalRayl",     &buffer.nCrystalRayl,"nCrystalRayl/I");
    Branch("localPosX",      &buffer.localPosX,"localPosX/F");
    Branch("localPosY",      &buffer.localPosY,"localPosY/F");
    Branch("localPosZ",      &buffer.localPosZ,"localPosZ/F");

    Branch("layerName",    (void *)buffer.layerName,"layername/C");
    //Branch("layerID",     &buffer.layerID,"layerID/I");
    Branch("sublayerID",     &buffer.sublayerID,"sublayerID/I");
    Branch("volumeID",       (void *)buffer.volumeID,"volumeID[10]/I");

    Branch("energyFinal",         &buffer.energyFin,"energyFinal/F");
    Branch("energyIni",         &buffer.energyIni,"energyIni/F");
}
//-----------------------------------------------------------------------------



void GateCCSingleTree::SetBranchAddresses(TTree* singlesTree,GateCCRootSingleBuffer& buffer)
{

    singlesTree->SetBranchAddress("runID",&buffer.runID);
    singlesTree->SetBranchAddress("eventID",&buffer.eventID);
    singlesTree->SetBranchAddress("time",&buffer.time);
    singlesTree->SetBranchAddress("energy",&buffer.energy);
    singlesTree->SetBranchAddress("energyFinal",&buffer.energyFin);
    singlesTree->SetBranchAddress("energyIni",&buffer.energyIni);
    singlesTree->SetBranchAddress("globalPosX",&buffer.globalPosX);
    singlesTree->SetBranchAddress("globalPosY",&buffer.globalPosY);
    singlesTree->SetBranchAddress("globalPosZ",&buffer.globalPosZ);
    singlesTree->SetBranchAddress("sourcePosX",&buffer.sourcePosX);
    singlesTree->SetBranchAddress("sourcePosY",&buffer.sourcePosY);
    singlesTree->SetBranchAddress("sourcePosZ",&buffer.sourcePosZ);
    singlesTree->SetBranchAddress("sourceEnergy",&buffer.sourceEnergy);
    singlesTree->SetBranchAddress("sourcePDG",&buffer.sourcePDG);
    singlesTree->SetBranchAddress("nCrystalConv",&buffer.nCrystalConv);
    singlesTree->SetBranchAddress("nCrystalCompt",&buffer.nCrystalCompt);
    singlesTree->SetBranchAddress("nCrystalRayl",&buffer.nCrystalRayl);
    singlesTree->SetBranchAddress("localPosX",&buffer.localPosX);
    singlesTree->SetBranchAddress("localPosY",&buffer.localPosY);
    singlesTree->SetBranchAddress("localPosZ",&buffer.localPosZ);
    singlesTree->SetBranchAddress("layerName",&buffer.layerName);
    //   singlesTree->SetBranchAddress("layerID",&buffer.layerID);
    singlesTree->SetBranchAddress("sublayerID",&buffer.sublayerID);
    singlesTree->SetBranchAddress("volumeID",buffer.volumeID);

}



GateDigi* GateCCRootSingleBuffer::CreateSingle()
{


    GateVolumeID aVolumeID(volumeID,ROOT_VOLUMEIDSIZE);
    // G4cout<<"CreateSingle::tras create aVolume"<<G4endl;
    GateDigi* aSingle = new GateDigi();
    // Initialise the hit data from the root-hit data
    aSingle->SetRunID(runID);
    aSingle->SetEventID(eventID);

    aSingle->SetTime(GetTime());
    G4ThreeVector globalPos;
    globalPos.setX(globalPosX);
    globalPos.setY(globalPosY);
    globalPos.setZ(globalPosZ);

    aSingle->SetGlobalPos(globalPos);

    G4ThreeVector lPos;
    lPos.setX(localPosX);
    lPos.setY(localPosY);
    lPos.setZ(localPosZ);
    aSingle->SetLocalPos(lPos);

    G4ThreeVector sPos;
    sPos.setX(sourcePosX);
    sPos.setY(sourcePosY);
    sPos.setZ(sourcePosZ);
    aSingle->SetSourcePosition(sPos);
    aSingle->SetSourceEnergy(sourceEnergy);
    aSingle->SetSourcePDG(sourcePDG);
    aSingle->SetNCrystalConv(nCrystalConv);
    aSingle->SetNCrystalCompton(nCrystalCompt);
    aSingle->SetNCrystalRayleigh(nCrystalRayl);
    aSingle->SetEnergy(energy);
    aSingle->SetEnergyIniTrack(energyIni);
    aSingle->SetEnergyFin(energyFin);
    aSingle->SetVolumeID(aVolumeID);


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
    energyFin        =  0./MeV;
    energyIni        = 0./MeV;
    globalPosX       = 0./mm;
    globalPosY       = 0./mm;
    globalPosZ       = 0./mm;
    localPosX        = 0./mm;
    localPosY        = 0./mm;
    localPosZ        = 0./mm;
    sourcePosX       = 0./mm;
    sourcePosY       = 0./mm;
    sourcePosZ       = 0./mm;
    sourceEnergy      =  0./MeV;
    sourcePDG        =0;
    nCrystalConv        =0;
    nCrystalCompt        =0;
    nCrystalRayl        =0;
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

    localPosX    = (aDigi->GetLocalPos()).x()/mm;
    localPosY    = (aDigi->GetLocalPos()).y()/mm;
    localPosZ    = (aDigi->GetLocalPos()).z()/mm;

    sourcePosX    = (aDigi->GetSourcePosition().getX())/mm;
    sourcePosY    = (aDigi->GetSourcePosition().getY())/mm;
    sourcePosZ    = (aDigi->GetSourcePosition().getZ())/mm;

    sourceEnergy   =aDigi->GetSourceEnergy()/MeV;
    sourcePDG    =aDigi->GetSourcePDG();
    nCrystalConv    =aDigi->GetNCrystalConv();
    nCrystalCompt    =aDigi->GetNCrystalCompton();
    nCrystalRayl    =aDigi->GetNCrystalRayleigh();
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
    Branch("sourcePosX",     &buffer.sourcePosX,"sourcePosX/F");
    Branch("sourcePosY",     &buffer.sourcePosY,"sourcePosY/F");
    Branch("sourcePosZ",     &buffer.sourcePosZ,"sourcePosZ/F");
    Branch("sourceEnergy",    &buffer.sourceEnergy,"sourceEnergy/F");
    Branch("sourcePDG",     &buffer.sourcePDG,"sourcePDG/I");
    Branch("nCrystalConv",     &buffer.nCrystalConv,"nCrystalConv/I");
    Branch("nCrystalCompt",     &buffer.nCrystalCompt,"nCrystalCompt/I");
    Branch("nCrystalRayl",     &buffer.nCrystalRayl,"nCrystalRayl/I");
    Branch("localPosX",      &buffer.localPosX,"localPosX/F");
    Branch("localPosY",      &buffer.localPosY,"localPosY/F");
    Branch("localPosZ",      &buffer.localPosZ,"localPosZ/F");
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

    coinTree->SetBranchAddress("sourcePosX",&buffer.sourcePosX);
    coinTree->SetBranchAddress("sourcePosY",&buffer.sourcePosY);
    coinTree->SetBranchAddress("sourcePosZ",&buffer.sourcePosZ);
    coinTree->SetBranchAddress("sourceEnergy",&buffer.sourceEnergy);
    coinTree->SetBranchAddress("sourcePDG",&buffer.sourcePDG);
    coinTree->SetBranchAddress("nCrystalConv",&buffer.nCrystalConv);
    coinTree->SetBranchAddress("nCrystalCompt",&buffer.nCrystalCompt);
    coinTree->SetBranchAddress("nCrystalRayl",&buffer.nCrystalRayl);
    coinTree->SetBranchAddress("localPosX",&buffer.localPosX);
    coinTree->SetBranchAddress("localPosY",&buffer.localPosY);
    coinTree->SetBranchAddress("localPosZ",&buffer.localPosZ);

    coinTree->SetBranchAddress("layerName",&buffer.layerName);
    //coinTree->SetBranchAddress("layerID",&buffer.layerID);
    coinTree->SetBranchAddress("sublayerID",&buffer.sublayerID);
    coinTree->SetBranchAddress("volumeID",buffer.volumeID);



}

GateCCCoincidenceDigi* GateCCRootCoincBuffer::CreateCoincidence()
{


    GateVolumeID aVolumeID(volumeID,ROOT_VOLUMEIDSIZE);

    GateCCCoincidenceDigi* aCoin = new GateCCCoincidenceDigi();

    aCoin->SetCoincidenceID(coincID);
    aCoin->SetRunID(runID);
    aCoin->SetEventID(eventID);
    aCoin->SetTime(GetTime());

    G4ThreeVector globalPos;
    globalPos.setX(globalPosX);
    globalPos.setY(globalPosY);
    globalPos.setZ(globalPosZ);

    aCoin->SetGlobalPos(globalPos);

    G4ThreeVector lPos;
    lPos.setX(localPosX);
    lPos.setY(localPosY);
    lPos.setZ(localPosZ);
    aCoin->SetLocalPos(lPos);

    G4ThreeVector sPos;
    sPos.setX(sourcePosX);
    sPos.setY(sourcePosY);
    sPos.setZ(sourcePosZ);
    aCoin->SetSourcePosition(sPos);

    aCoin->SetSourceEnergy(sourceEnergy);
    aCoin->SetSourcePDG(sourcePDG);
    aCoin->SetNCrystalConv(nCrystalConv);
    aCoin->SetNCrystalCompton(nCrystalCompt);
    aCoin->SetNCrystalRayleigh(nCrystalRayl);

    aCoin->SetEnergy(energy);
    aCoin->SetIniEnergy(energyIni);
    aCoin->SetFinalEnergy(energyFin);
    //G4cout<<"antes del setvolID to the aSingle"<<G4endl;
    aCoin->SetVolumeID(aVolumeID);

    //Set the paremeters

    return aCoin;
}



void GateCCRootConesBuffer::Clear()
{
    //coincID          =-1;

    energy1           = 0./MeV;
    energyR           = 0./MeV;
    globalPosX1       = 0./mm;
    globalPosY1       = 0./mm;
    globalPosZ1       = 0./mm;
    globalPosX2       = 0./mm;
    globalPosY2       = 0./mm;
    globalPosZ2       = 0./mm;

    m_nSingles=0;
    m_IsTrueCoind=true;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCRootConesBuffer::Fill(GateComptonCameraCones* aCon)
{
    //I need to include it
    //coincID       =

    energy1        =  aCon->GetEnergy1()/MeV;;
    energyR        =  aCon->GetEnergyR()/MeV;

    globalPosX1    = (aCon->GetPosition1()).x()/mm;
    globalPosY1    = (aCon->GetPosition1()).y()/mm;
    globalPosZ1    = (aCon->GetPosition1()).z()/mm;
    globalPosX2    = (aCon->GetPosition2()).x()/mm;
    globalPosY2    = (aCon->GetPosition2()).y()/mm;
    globalPosZ2    = (aCon->GetPosition2()).z()/mm;

    m_nSingles=aCon->GetNumSingles();
    m_IsTrueCoind=aCon->GetTrueFlag();


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCCConesTree::Init(GateCCRootConesBuffer& buffer)
{
    SetAutoSave(2000);
    //Branch("coincID",          &buffer.coincID,"coincID/I");
    Branch("energy1",         &buffer.energy1,"energy1/F");
    Branch("energyR",         &buffer.energyR,"energyR/F");

    Branch("globalPosX1",     &buffer.globalPosX1,"globalPosX1/F");
    Branch("globalPosY1",     &buffer.globalPosY1,"globalPosY1/F");
    Branch("globalPosZ1",     &buffer.globalPosZ1,"globalPosZ1/F");
    Branch("globalPosX2",     &buffer.globalPosX2,"globalPosX2/F");
    Branch("globalPosY2",     &buffer.globalPosY2,"globalPosY2/F");
    Branch("globalPosZ2",     &buffer.globalPosZ2,"globalPosZ2/F");


    Branch("nSingles",        &buffer.m_nSingles,"m_nSingles/I");
    Branch("IsTrueCoind",     &buffer.m_IsTrueCoind,"m_IsTrueCoind/O");

}
//-----------------------------------------------------------------------------

void GateCCConesTree::SetBranchAddresses(TTree* conTree,GateCCRootConesBuffer& buffer)
{


    //coinTree->SetBranchAddress("coincID",&buffer.coincID);

    conTree->SetBranchAddress("energy1",&buffer.energy1);
    conTree->SetBranchAddress("energyR",&buffer.energyR);

    conTree->SetBranchAddress("globalPosX1",&buffer.globalPosX1);
    conTree->SetBranchAddress("globalPosY1",&buffer.globalPosY1);
    conTree->SetBranchAddress("globalPosZ1",&buffer.globalPosZ1);
    conTree->SetBranchAddress("globalPosX2",&buffer.globalPosX2);
    conTree->SetBranchAddress("globalPosY2",&buffer.globalPosY2);
    conTree->SetBranchAddress("globalPosZ2",&buffer.globalPosZ2);

    conTree->SetBranchAddress("nSingles",&buffer.m_nSingles);
    conTree->SetBranchAddress("IsTrueCoind",&buffer.m_IsTrueCoind);



}

GateComptonCameraCones* GateCCRootConesBuffer::CreateCone()
{
    GateComptonCameraCones* aCon = new GateComptonCameraCones();

    //aCoin->SetCoincidenceID(coincID);



    G4ThreeVector globalPos1;
    globalPos1.setX(globalPosX1);
    globalPos1.setY(globalPosY1);
    globalPos1.setZ(globalPosZ1);
    G4ThreeVector globalPos2;
    globalPos2.setX(globalPosX2);
    globalPos2.setY(globalPosY2);
    globalPos2.setZ(globalPosZ2);


    aCon->SetPosition1(globalPos1);
    aCon->SetPosition2(globalPos2);

    aCon->SetEnergy1(energy1);
    aCon->SetEnergy1(energyR);
    aCon->SetNumSingles(m_nSingles);
    aCon->SetTrueFlag(m_IsTrueCoind);


    return aCon;
}
