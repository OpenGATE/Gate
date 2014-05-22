/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateRootDefs.hh"
#include "GateCrystalHit.hh"
#include "GateSingleDigi.hh"
#include "GateCoincidenceDigi.hh"

#include "GateOutputMgr.hh"
#include "GateAnalysis.hh"

static char *theDefaultOutputIDName[ROOT_OUTPUTIDSIZE] = 
 {(char *)"baseID", 
  (char *)"unused1ID",
  (char *)"unused2ID",
  (char *)"unused3ID",
  (char *)"unused4ID",
  (char *)"unused5ID"
  };

static char outputIDName     [ROOT_OUTPUTIDSIZE][24];
static char outputIDLeafList [ROOT_OUTPUTIDSIZE][24];
static char outputIDName1    [ROOT_OUTPUTIDSIZE][24];
static char outputIDLeafList1[ROOT_OUTPUTIDSIZE][24];
static char outputIDName2    [ROOT_OUTPUTIDSIZE][24];
static char outputIDLeafList2[ROOT_OUTPUTIDSIZE][24];


void GateRootDefs::SetDefaultOutputIDNames()
{
  for (size_t depth=0; depth<ROOT_OUTPUTIDSIZE ; ++depth )
  SetOutputIDName(theDefaultOutputIDName[depth],depth); 
}

void GateRootDefs::SetOutputIDName(char * anOutputIDName, size_t depth)
{  
  sprintf(outputIDName[depth],"%s",anOutputIDName);
  sprintf(outputIDName1[depth],"%s1",anOutputIDName);
  sprintf(outputIDName2[depth],"%s2",anOutputIDName);

  sprintf(outputIDLeafList[depth],"%s/I",anOutputIDName);
  sprintf(outputIDLeafList1[depth],"%s1/I",anOutputIDName);
  sprintf(outputIDLeafList2[depth],"%s2/I",anOutputIDName);
}

/*	HDS : septal penetration
	The aim of this method is to read the boolean value of the RecordSeptalFlag in the output module 
	GateAnalysis. If the flag is true, we add a branch "septalNb" in the Hit tree and in all Single
	Chain trees (not coincidences)."
	If no GateAnalysis output module is currently running, this method will throw a warning.
	If more than one GateAnalysis module is present, only the first one will be taken into account.
*/

G4bool GateRootDefs::GetRecordSeptalFlag()
{
	G4bool ans = false;

	GateOutputMgr* theOutputMgr = GateOutputMgr::GetInstance();
	GateAnalysis* analysis = dynamic_cast<GateAnalysis*>(theOutputMgr->GetModule("analysis"));
	if ( ! analysis ) {
		G4cout << G4endl << "!!! WARNING : No 'analysis' output module found. "
				<< "Septal hits won't be recorded. !!!" << G4endl
				<< "!!! This is just a warning message. The simulation will continue. !!!";
	} else {
		ans = analysis->GetRecordSeptalFlag();
	} 

	return ans;
}


void GateRootHitBuffer::Clear()
{
  PDGEncoding   = 0;
  trackID       = 0;
  parentID      = 0;
  time            = 0./s;
  edep            = 0./MeV;
  stepLength      = 0./mm;
  posX            = 0./mm;
  posY            = 0./mm;
  posZ            = 0./mm;
  localPosX       = 0./mm;
  localPosY       = 0./mm;
  localPosZ       = 0./mm;
  photonID        = -1;
  nPhantomCompton = -1;
  nCrystalCompton = -1;
  nPhantomRayleigh = -1;
  nCrystalRayleigh = -1;
  primaryID       = -1;
  sourcePosX      = 0./mm;
  sourcePosY      = 0./mm;
  sourcePosZ      = 0./mm;
  sourceID        = -1;
  eventID         = -1;
  runID           = -1;
  axialPos        = 0.;
  rotationAngle   = 0.;

  strcpy (processName, " ");
  strcpy (comptonVolumeName," ");
  strcpy (RayleighVolumeName," ");

  size_t d;
  for ( d = 0 ; d < ROOT_OUTPUTIDSIZE ; ++d )
    outputID[d] = -1;
  for ( d = 0 ; d < ROOT_VOLUMEIDSIZE ; ++d )
    volumeID[d] = -1;
    
// HDS : septal
  septalNb = 0;
}


void GateRootHitBuffer::Fill(GateCrystalHit* aHit)
{
  size_t d;

  PDGEncoding     = aHit->GetPDGEncoding();
  trackID         = aHit->GetTrackID();
  parentID        = aHit->GetParentID();
  SetTime(          aHit->GetTime() );
  SetEdep(          aHit->GetEdep() );
  SetStepLength(    aHit->GetStepLength() );
  SetPos(           aHit->GetGlobalPos() );
  SetLocalPos(      aHit->GetLocalPos() );
  parentID        = aHit->GetParentID();
  for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
    outputID[d]   = aHit->GetComponentID(d);
  photonID        = aHit->GetPhotonID();
  nPhantomCompton = aHit->GetNPhantomCompton();
  nCrystalCompton = aHit->GetNCrystalCompton();
  nPhantomRayleigh = aHit->GetNPhantomRayleigh();
  nCrystalRayleigh = aHit->GetNCrystalRayleigh();
  primaryID       = aHit->GetPrimaryID();
  SetSourcePos(     aHit->GetSourcePosition() );
  sourceID        = aHit->GetSourceID();
  eventID         = aHit->GetEventID();
  runID           = aHit->GetRunID();
  SetAxialPos(      aHit->GetScannerPos().z() );
  SetRotationAngle( aHit->GetScannerRotAngle() );
	  
// HDS : septal
	septalNb = aHit->GetNSeptal();
	
  strcpy (processName, aHit->GetProcess().c_str());

  strcpy (comptonVolumeName,aHit->GetComptonVolumeName().c_str());
  if (aHit->GetComptonVolumeName().length()>=40)
    G4cout << "GateToRoot::RecordEndOfEvent : length of volume name exceeding 40: " << 
	      aHit->GetComptonVolumeName().length()+1 << G4endl;

  strcpy (RayleighVolumeName,aHit->GetRayleighVolumeName().c_str());
  if (aHit->GetRayleighVolumeName().length()>=40)
    G4cout << "GateToRoot::RecordEndOfEvent : length of volume name exceeding 40: " << 
	      aHit->GetRayleighVolumeName().length()+1 << G4endl;

  aHit->GetVolumeID().StoreDaughterIDs(volumeID,ROOT_VOLUMEIDSIZE);


//G4cout << "RootDefs : runID = " << runID << G4endl;


}

GateCrystalHit* GateRootHitBuffer::CreateHit()
{	
    // Create a volumeID from the root-hit data
    GateVolumeID aVolumeID(volumeID,ROOT_VOLUMEIDSIZE);

    // Create an output-volumeID from the root-hit data
    GateOutputVolumeID anOutputVolumeID;
    size_t d;
    for ( d=0 ; d<ROOT_OUTPUTIDSIZE ; ++d)
      anOutputVolumeID[d] = outputID[d];

    // Create a new hit
    GateCrystalHit* aHit = new GateCrystalHit();

    // Initialise the hit data from the root-hit data
    aHit->SetEdep(    	      	GetEdep() );  
    aHit->SetStepLength(      	GetStepLength() );  
    aHit->SetTime(    	      	GetTime() );
    aHit->SetGlobalPos(       	GetPos() );
    aHit->SetLocalPos(        	GetLocalPos() );
    aHit->SetProcess( 	      	processName );
    aHit->SetPDGEncoding(     	PDGEncoding );
    aHit->SetTrackID( 	      	trackID );
    aHit->SetParentID(        	parentID );
    aHit->SetSourceID(        	sourceID );
    aHit->SetSourcePosition(  	GetSourcePos() );
    aHit->SetPhotonID(        	photonID );
    aHit->SetNPhantomCompton( 	nPhantomCompton );
    aHit->SetNCrystalCompton( 	nCrystalCompton );
    aHit->SetNPhantomRayleigh( 	nPhantomRayleigh);
    aHit->SetNCrystalRayleigh( 	nCrystalRayleigh );
    aHit->SetComptonVolumeName( comptonVolumeName );
    aHit->SetRayleighVolumeName( RayleighVolumeName );
    aHit->SetPrimaryID(       	primaryID );
    aHit->SetEventID( 	      	eventID );
    aHit->SetRunID(   	      	runID );
    aHit->SetScannerPos(      	G4ThreeVector(0., 0., GetAxialPos() ) );
    aHit->SetScannerRotAngle( 	GetRotationAngle() );
    aHit->SetVolumeID(	      	aVolumeID);
    aHit->SetOutputVolumeID(  	anOutputVolumeID );
	aHit->SetNSeptal( septalNb );  // HDS : septal penetration
    return aHit;
}

void GateHitTree::Init(GateRootHitBuffer& buffer)
{
    SetAutoSave(1000);
    Branch("PDGEncoding",    &buffer.PDGEncoding,"PDGEncoding/I");
    Branch("trackID",        &buffer.trackID,"trackID/I");
    Branch("parentID",       &buffer.parentID,"parentID/I");
    Branch("time",           &buffer.time,"time/D");
    Branch("edep",           &buffer.edep,"edep/F");
    Branch("stepLength",     &buffer.stepLength,"stepLength/F");
    Branch("posX",           &buffer.posX,"posX/F");
    Branch("posY",           &buffer.posY,"posY/F");
    Branch("posZ",           &buffer.posZ,"posZ/F");
    Branch("localPosX",      &buffer.localPosX,"localPosX/F");
    Branch("localPosY",      &buffer.localPosY,"localPosY/F");
    Branch("localPosZ",      &buffer.localPosZ,"localPosZ/F");
    for (size_t d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
      Branch(outputIDName[d],(void *)(buffer.outputID+d),outputIDLeafList[d]);
    Branch("photonID",       &buffer.photonID,"photonID/I");
    Branch("nPhantomCompton",&buffer.nPhantomCompton,"nPhantomCompton/I");
    Branch("nCrystalCompton",&buffer.nCrystalCompton,"nCrystalCompton/I");
    Branch("nPhantomRayleigh",&buffer.nPhantomRayleigh,"nPhantomRayleigh/I");
    Branch("nCrystalRayleigh",&buffer.nCrystalRayleigh,"nCrystalRayleigh/I");
    Branch("primaryID",      &buffer.primaryID,"primaryID/I");
    Branch("sourcePosX",     &buffer.sourcePosX,"sourcePosX/F");
    Branch("sourcePosY",     &buffer.sourcePosY,"sourcePosY/F");
    Branch("sourcePosZ",     &buffer.sourcePosZ,"sourcePosZ/F");
    Branch("sourceID",       &buffer.sourceID,"sourceID/I");
    Branch("eventID",        &buffer.eventID,"eventID/I");
    Branch("runID",          &buffer.runID,"runID/I");
    Branch("axialPos",       &buffer.axialPos,"axialPos/F");
    Branch("rotationAngle",  &buffer.rotationAngle,"rotationAngle/F");
    Branch("volumeID",       (void *)buffer.volumeID,"volumeID[10]/I");
    Branch("processName",    (void *)buffer.processName,"processName/C");
    Branch("comptVolName",   (void *)buffer.comptonVolumeName,"comptVolName/C");
    Branch("RayleighVolName",   (void *)buffer.RayleighVolumeName,"RayleighVolName/C");
    // HDS : record septal penetration
    if (GateRootDefs::GetRecordSeptalFlag())	Branch("septalNb",   &buffer.septalNb,"septalNb/I");
}

void GateHitTree::SetBranchAddresses(TTree* hitTree,GateRootHitBuffer& buffer)
{
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
  for (size_t d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
      hitTree->SetBranchAddress(outputIDName[d],(void *)(buffer.outputID+d));
  hitTree->SetBranchAddress("photonID",&buffer.photonID);
  hitTree->SetBranchAddress("nPhantomCompton",&buffer.nPhantomCompton);
  hitTree->SetBranchAddress("nCrystalCompton",&buffer.nCrystalCompton);
  hitTree->SetBranchAddress("nPhantomRayleigh",&buffer.nPhantomRayleigh);
  hitTree->SetBranchAddress("nCrystalRayleigh",&buffer.nCrystalRayleigh);
  hitTree->SetBranchAddress("primaryID",&buffer.primaryID);
  hitTree->SetBranchAddress("sourcePosX",&buffer.sourcePosX);
  hitTree->SetBranchAddress("sourcePosY",&buffer.sourcePosY);
  hitTree->SetBranchAddress("sourcePosZ",&buffer.sourcePosZ);
  hitTree->SetBranchAddress("sourceID",&buffer.sourceID);
  hitTree->SetBranchAddress("eventID",&buffer.eventID);
  hitTree->SetBranchAddress("runID",&buffer.runID);
  hitTree->SetBranchAddress("axialPos",&buffer.axialPos);
  hitTree->SetBranchAddress("rotationAngle",&buffer.rotationAngle);
  hitTree->SetBranchAddress("processName",&buffer.processName);
  hitTree->SetBranchAddress("comptVolName",&buffer.comptonVolumeName);
  hitTree->SetBranchAddress("RayleighVolName",&buffer.RayleighVolumeName);
  hitTree->SetBranchAddress("volumeID",buffer.volumeID);
}


void GateRootSingleBuffer::Clear()
{
  size_t d;

  runID            = -1;
  eventID          = -1;
  sourceID         = -1;
  sourcePosX       = 0./mm;
  sourcePosY       = 0./mm;
  sourcePosZ       = 0./mm;
  time             = 0./s;
  energy           = 0./MeV;
  globalPosX       = 0./mm;
  globalPosY       = 0./mm;
  globalPosZ       = 0./mm;
  for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
    outputID[d]    = -1;
  axialPos         = 0.;
  rotationAngle    = 0.;
  comptonPhantom   = -1;
  comptonCrystal   = -1;
  RayleighPhantom   = -1;
  RayleighCrystal   = -1;
  strcpy (comptonVolumeName," ");
  strcpy (RayleighVolumeName," ");
  // HDS : septal
  septalNb = 0;
}



void GateRootSingleBuffer::Fill(GateSingleDigi* aDigi)
{
      size_t d;

      runID         =  aDigi->GetRunID();
      eventID       =  aDigi->GetEventID();
      sourceID      =  aDigi->GetSourceID();
      sourcePosX    = (aDigi->GetSourcePosition()).x()/mm;
      sourcePosY    = (aDigi->GetSourcePosition()).y()/mm;
      sourcePosZ    = (aDigi->GetSourcePosition()).z()/mm;
      time          =  aDigi->GetTime()/s;
      energy        =  aDigi->GetEnergy()/MeV;
      globalPosX    = (aDigi->GetGlobalPos()).x()/mm;
      globalPosY    = (aDigi->GetGlobalPos()).y()/mm;
      globalPosZ    = (aDigi->GetGlobalPos()).z()/mm;
      for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
    	outputID[d] =  aDigi->GetComponentID(d);
      comptonPhantom=  aDigi->GetNPhantomCompton();
      comptonCrystal=  aDigi->GetNCrystalCompton();
      RayleighPhantom=  aDigi->GetNPhantomRayleigh();
      RayleighCrystal=  aDigi->GetNCrystalRayleigh();
      axialPos      = (aDigi->GetScannerPos()).z()/mm;
      rotationAngle = aDigi->GetScannerRotAngle()/deg;
      strcpy (comptonVolumeName,(aDigi->GetComptonVolumeName()).c_str());
      strcpy (RayleighVolumeName,(aDigi->GetRayleighVolumeName()).c_str());

      // HDS : septal penetration
      septalNb = aDigi->GetNSeptal();
}


void GateSingleTree::Init(GateRootSingleBuffer& buffer)
{
    SetAutoSave(1000);
    Branch("runID",          &buffer.runID,"runID/I");
    Branch("eventID",        &buffer.eventID,"eventID/I");
    Branch("sourceID",       &buffer.sourceID,"sourceID/I");
    Branch("sourcePosX",     &buffer.sourcePosX,"sourcePosX/F");
    Branch("sourcePosY",     &buffer.sourcePosY,"sourcePosY/F");
    Branch("sourcePosZ",     &buffer.sourcePosZ,"sourcePosZ/F");
    Branch("time",           &buffer.time,"time/D");
    Branch("energy",         &buffer.energy,"energy/F");
    Branch("globalPosX",     &buffer.globalPosX,"globalPosX/F");
    Branch("globalPosY",     &buffer.globalPosY,"globalPosY/F");
    Branch("globalPosZ",     &buffer.globalPosZ,"globalPosZ/F");
    for (size_t d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
      Branch(outputIDName[d],(void *)(buffer.outputID+d),outputIDLeafList[d]);
    Branch("comptonPhantom", &buffer.comptonPhantom,"comptonPhantom/I");
    Branch("comptonCrystal", &buffer.comptonCrystal,"comptonCrystal/I");
    Branch("RayleighPhantom", &buffer.RayleighPhantom,"RayleighPhantom/I");
    Branch("RayleighCrystal", &buffer.RayleighCrystal,"RayleighCrystal/I");
    Branch("axialPos",       &buffer.axialPos,"axialPos/F");
    Branch("rotationAngle",  &buffer.rotationAngle,"rotationAngle/F");
    Branch("comptVolName",   (void *)buffer.comptonVolumeName,"comptVolName/C");
    Branch("RayleighVolName",   (void *)buffer.RayleighVolumeName,"RayleighVolName/C");
    // HDS : record septal penetration
    if (GateRootDefs::GetRecordSeptalFlag())	Branch("septalNb",   &buffer.septalNb,"septalNb/I");
}


void GateRootCoincBuffer::Clear()
{
  size_t d;

  runID           = -1;
  axialPos        =  0.;
  rotationAngle   =  0.;

  eventID1        = -1;
  sourceID1       = -1;
  sourcePosX1     = 0./mm;
  sourcePosY1     = 0./mm;
  sourcePosZ1     = 0./mm;
  time1           = 0./s;
  energy1         = 0./MeV;
  globalPosX1     = 0./mm;
  globalPosY1     = 0./mm;
  globalPosZ1     = 0./mm;
  for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
    outputID1[d]  = -1;
  comptonPhantom1 = -1;
  comptonCrystal1 = -1;
  RayleighPhantom1 = -1;
  RayleighCrystal1 = -1;
  strcpy (comptonVolumeName1," ");
  strcpy (RayleighVolumeName1," ");

  eventID2        = -1;
  sourceID2       = -1;
  sourcePosX2     = 0./mm;
  sourcePosY2     = 0./mm;
  sourcePosZ2     = 0./mm;
  time2           = 0./s;
  energy2         = 0./MeV;
  globalPosX2     = 0./mm;
  globalPosY2     = 0./mm;
  globalPosZ2     = 0./mm;
  for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
    outputID2[d]  = -1;
  comptonPhantom2 = -1;
  comptonCrystal2 = -1;
  RayleighPhantom2 = -1;
  RayleighCrystal2 = -1;
  strcpy (comptonVolumeName2," ");
  strcpy (RayleighVolumeName2," ");
}



void GateRootCoincBuffer::Fill(GateCoincidenceDigi* aDigi)
{      
      size_t d;

      runID          = (aDigi->GetPulse(0)).GetRunID();
      axialPos       = (aDigi->GetPulse(0)).GetScannerPos().z()/mm;
      rotationAngle  = (aDigi->GetPulse(0)).GetScannerRotAngle()/deg;

      eventID1       = (aDigi->GetPulse(0)).GetEventID();
      sourceID1      = (aDigi->GetPulse(0)).GetSourceID();      
      sourcePosX1    = (aDigi->GetPulse(0)).GetSourcePosition().x()/mm;
      sourcePosY1    = (aDigi->GetPulse(0)).GetSourcePosition().y()/mm;
      sourcePosZ1    = (aDigi->GetPulse(0)).GetSourcePosition().z()/mm;
      time1          = (aDigi->GetPulse(0)).GetTime()/s;
      energy1        = (aDigi->GetPulse(0)).GetEnergy()/MeV;
      globalPosX1    = (aDigi->GetPulse(0)).GetGlobalPos().x()/mm;
      globalPosY1    = (aDigi->GetPulse(0)).GetGlobalPos().y()/mm;
      globalPosZ1    = (aDigi->GetPulse(0)).GetGlobalPos().z()/mm;
      for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
    	outputID1[d] = (aDigi->GetPulse(0)).GetComponentID(d);
      comptonPhantom1       = (aDigi->GetPulse(0)).GetNPhantomCompton();
      comptonCrystal1       = (aDigi->GetPulse(0)).GetNCrystalCompton();
      RayleighPhantom1       = (aDigi->GetPulse(0)).GetNPhantomRayleigh();
      RayleighCrystal1       = (aDigi->GetPulse(0)).GetNCrystalRayleigh();
     
      strcpy (comptonVolumeName1,((aDigi->GetPulse(0)).GetComptonVolumeName()).c_str());
      strcpy (RayleighVolumeName1,((aDigi->GetPulse(0)).GetRayleighVolumeName()).c_str());

      eventID2       = (aDigi->GetPulse(1)).GetEventID();
      sourceID2      = (aDigi->GetPulse(1)).GetSourceID();      
      sourcePosX2    = (aDigi->GetPulse(1)).GetSourcePosition().x()/mm;
      sourcePosY2    = (aDigi->GetPulse(1)).GetSourcePosition().y()/mm;
      sourcePosZ2    = (aDigi->GetPulse(1)).GetSourcePosition().z()/mm;
      time2          = (aDigi->GetPulse(1)).GetTime()/s;
      energy2        = (aDigi->GetPulse(1)).GetEnergy()/MeV;
      globalPosX2    = (aDigi->GetPulse(1)).GetGlobalPos().x()/mm;
      globalPosY2    = (aDigi->GetPulse(1)).GetGlobalPos().y()/mm;
      globalPosZ2    = (aDigi->GetPulse(1)).GetGlobalPos().z()/mm;
      for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
    	outputID2[d] = (aDigi->GetPulse(1)).GetComponentID(d);
      comptonPhantom2       = (aDigi->GetPulse(1)).GetNPhantomCompton();
      comptonCrystal2       = (aDigi->GetPulse(1)).GetNCrystalCompton();
      RayleighPhantom2       = (aDigi->GetPulse(1)).GetNPhantomRayleigh();
      RayleighCrystal2       = (aDigi->GetPulse(1)).GetNCrystalRayleigh();

      strcpy (comptonVolumeName2,((aDigi->GetPulse(1)).GetComptonVolumeName()).c_str());
      strcpy (RayleighVolumeName2,((aDigi->GetPulse(1)).GetRayleighVolumeName()).c_str());

      sinogramTheta  = ComputeSinogramTheta();
      sinogramS      = ComputeSinogramS();
}



G4double GateRootCoincBuffer::ComputeSinogramTheta()
{
  G4double theta;
  
  if ((globalPosX1-globalPosX2) != 0.) {
    theta = atan( (globalPosX1 - globalPosX2) / (globalPosY1 - globalPosY2) );
    if (theta < 0.) theta = theta+3.1416;
  } else {
    theta=3.1416/2.;
  }

  return theta;
}




G4double GateRootCoincBuffer::ComputeSinogramS()
{
  G4double s;

  G4double denom = (globalPosY1-globalPosY2) * (globalPosY1-globalPosY2) +
                   (globalPosX2-globalPosX1) * (globalPosX2-globalPosX1);
  
  if (denom!=0.) {
    denom = sqrt(denom);

    s = ( globalPosX1 * (globalPosY2-globalPosY1) +
	  globalPosY1 * (globalPosX1-globalPosX2)  ) 
      	/ denom; 
  } else {
    s = 0.;
  }

  G4double theta;
  if ((globalPosX1-globalPosX2)!=0.) {
    theta=atan((globalPosX1-globalPosX2) /
	       (globalPosY1-globalPosY2));
  } else {
    theta=3.1416/2.;
  }
  if ((theta > 0.) && ((globalPosX1-globalPosX2) > 0.)) s = -s;
  if ((theta < 0.) && ((globalPosX1-globalPosX2) < 0.)) s = -s;
  if ( theta < 0.) {
    theta = theta+3.1416;
    s = -s;
  }

  return s;
}



void GateCoincTree::Init(GateRootCoincBuffer& buffer)
{
    SetAutoSave(1000);
    Branch("runID",          &buffer.runID,"runID/I");
    Branch("axialPos",       &buffer.axialPos,"axialPos/F");
    Branch("rotationAngle",  &buffer.rotationAngle,"rotationAngle/F");
    
    Branch("eventID1",       &buffer.eventID1,"eventID1/I");
    Branch("sourceID1",      &buffer.sourceID1,"sourceID1/I");
    Branch("sourcePosX1",    &buffer.sourcePosX1,"sourcePosX1/F");
    Branch("sourcePosY1",    &buffer.sourcePosY1,"sourcePosY1/F");
    Branch("sourcePosZ1",    &buffer.sourcePosZ1,"sourcePosZ1/F");
    Branch("time1",          &buffer.time1,"time1/D");
    Branch("energy1",        &buffer.energy1,"energy1/F");
    Branch("globalPosX1",    &buffer.globalPosX1,"globalPosX1/F");
    Branch("globalPosY1",    &buffer.globalPosY1,"globalPosY1/F");
    Branch("globalPosZ1",    &buffer.globalPosZ1,"globalPosZ1/F");
    size_t d;
    for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
      Branch(outputIDName1[d],(void*)( buffer.outputID1 +d ),outputIDLeafList1[d]);
    Branch("comptonPhantom1",&buffer.comptonPhantom1,"comptonPhantom1/I");
    Branch("comptonCrystal1",&buffer.comptonCrystal1,"comptonCrystal1/I");
    Branch("RayleighPhantom1",&buffer.RayleighPhantom1,"RayleighPhantom1/I");
    Branch("RayleighCrystal1",&buffer.RayleighCrystal1,"RayleighCrystal1/I");

    Branch("eventID2",       &buffer.eventID2,"eventID2/I");
    Branch("sourceID2",      &buffer.sourceID2,"sourceID2/I");
    Branch("sourcePosX2",    &buffer.sourcePosX2,"sourcePosX2/F");
    Branch("sourcePosY2",    &buffer.sourcePosY2,"sourcePosY2/F");
    Branch("sourcePosZ2",    &buffer.sourcePosZ2,"sourcePosZ2/F");
    Branch("time2",          &buffer.time2,"time2/D");
    Branch("energy2",        &buffer.energy2,"energy2/F");
    Branch("globalPosX2",    &buffer.globalPosX2,"globalPosX2/F");
    Branch("globalPosY2",    &buffer.globalPosY2,"globalPosY2/F");
    Branch("globalPosZ2",    &buffer.globalPosZ2,"globalPosZ2/F");
    for (d=0; d<ROOT_OUTPUTIDSIZE ; ++d)
      Branch(outputIDName2[d],(void*)( buffer.outputID2 + d),outputIDLeafList2[d]);
    Branch("comptonPhantom2",&buffer.comptonPhantom2,"comptonPhantom2/I");
    Branch("comptonCrystal2",&buffer.comptonCrystal2,"comptonCrystal2/I");
    Branch("RayleighPhantom2",&buffer.RayleighPhantom2,"RayleighPhantom2/I");
    Branch("RayleighCrystal2",&buffer.RayleighCrystal2,"RayleighCrystal2/I");
    
    Branch("sinogramTheta",  &buffer.sinogramTheta,"sinogramTheta/F");
    Branch("sinogramS",      &buffer.sinogramS,"sinogramS/F");

    Branch("comptVolName1",  (void *)buffer.comptonVolumeName1,"comptVolName1/C");
    Branch("comptVolName2",  (void *)buffer.comptonVolumeName2,"comptVolName2/C");

    Branch("RayleighVolName1",  (void *)buffer.RayleighVolumeName1,"RayleighVolName1/C");
    Branch("RayleighVolName2",  (void *)buffer.RayleighVolumeName2,"RayleighVolName2/C");
}


#endif




