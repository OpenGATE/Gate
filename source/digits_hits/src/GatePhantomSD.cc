/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateMessageManager.hh"
#include "GatePhantomSD.hh"
#include "GatePhantomHit.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4ios.hh"
#include "G4VProcess.hh"

// Name of the hit collection
const G4String GatePhantomSD::thePhantomCollectionName = "phantomCollection";



//------------------------------------------------------------------------------
// Constructor
GatePhantomSD::GatePhantomSD(const G4String& name)
:G4VSensitiveDetector(name)
{
  collectionName.insert(thePhantomCollectionName);
}
//------------------------------------------------------------------------------


/*GatePhantomSD::GatePhantomSD(G4String name)
:G4VSensitiveDetector(name)
{
  G4String HCname;
  collectionName.insert(HCname="PhantomCollection");

}
*/
GatePhantomSD::~GatePhantomSD(){;}

GatePhantomSD *GatePhantomSD::Clone() const {
    auto clone = new GatePhantomSD(SensitiveDetectorName);
    clone->thePathName = thePathName;
    clone->fullPathName = fullPathName;
    clone->verboseLevel = verboseLevel;
    clone->active = active;
    clone->ROgeometry = ROgeometry;
    clone->filter = filter;

    clone->phantomCollection = phantomCollection;

    return clone;
}

void GatePhantomSD::Initialize(G4HCofThisEvent*HCE)
{
  static int HCID = -1;
  // Not thread safe but moving to local variable doesn't work
  phantomCollection = new GatePhantomHitsCollection
                   (SensitiveDetectorName,thePhantomCollectionName);
  if(HCID<0)
  { HCID = GetCollectionID(0); }
  HCE->AddHitsCollection(HCID,phantomCollection);

  //G4cout << "GatePhantomSD::Initialize - Full collection name: "  << SensitiveDetectorName+"/"+collectionName[0] << Gateendl << std::flush;

}

G4bool GatePhantomSD::ProcessHits(G4Step* aStep,G4TouchableHistory* /*ROhist*/) {
  G4Track* aTrack       = aStep->GetTrack();
  G4int    trackID      = aTrack->GetTrackID();
  G4int    parentID     = aTrack->GetParentID();

  G4String partName     = aTrack->GetDefinition()->GetParticleName();
  G4int    PDGEncoding  = aTrack->GetDefinition()->GetPDGEncoding();

  G4StepPoint* newStepPoint = aStep->GetPostStepPoint();
  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();

  G4int voxCoord(0);
  const G4VTouchable* t(preStepPoint->GetTouchable());

  G4String pvName;
  if (t) {
    voxCoord=t->GetReplicaNumber(0);
    pvName  =t->GetVolume()->GetName();
    //   G4cout << "GatePhantomSD::ProcessHits - voxelcoord is "<< voxCoord << ", pvname "<< pvName << Gateendl;
  }


//    ID : crystal where the hit takes place
//    G4TouchableHistory* theTouchable =
//    	(G4TouchableHistory*)(newStepPoint->GetTouchable() );
//    G4VPhysicalVolume* physVol = theTouchable->GetVolume();
//    G4int crystalID = physVol->GetCopyNo();

  // moduleID : module where the crystal is placed
//    physVol = physVol->GetMother();
//    G4int moduleID  = physVol->GetCopyNo();

  // process in the current step
  const G4VProcess* process;
  G4String processName;

  process = newStepPoint->GetProcessDefinedStep();
  if (process != NULL)
      processName = process->GetProcessName();
  else
      processName = "";

  //Note: if the energy is deposited by an electron hit by the gamma it doesn't work...

  // deposit energy in the current step
  G4double edep = aStep->GetTotalEnergyDeposit();

  //if(edep==0.) return true;

  // stepLength of the current step
  G4double stepLength = aStep->GetStepLength();
  // time of the current step
  G4double aTime = newStepPoint->GetGlobalTime();

  // hit position
  G4ThreeVector position = newStepPoint->GetPosition();

  GatePhantomHit* aHit = new GatePhantomHit();
  aHit->SetPDGEncoding( PDGEncoding );
  aHit->SetEdep( edep );
  aHit->SetStepLength( stepLength );
  aHit->SetTime( aTime );
  aHit->SetPos( position );
  aHit->SetProcess( processName );
  aHit->SetTrackID( trackID );
  aHit->SetParentID( parentID );
  aHit->SetVoxelCoordinates( voxCoord );
  aHit->SetPhysVolName( pvName );

  /*
  G4cout << "PARTICLE:" << partName
	 << "TRACK ID:" << trackID
  	 << "; PARENT ID:" << parentID
	 << "; PROCESS:" << processName
  	 << "; EDEP:" << edep/keV
	 << ", position " <<position
	 << "; Coord " << voxCoord
	 << Gateendl;
  */

  phantomCollection->insert( aHit );

  return true;
}

void GatePhantomSD::EndOfEvent(G4HCofThisEvent*)
{;}

void GatePhantomSD::clear()
{
}

void GatePhantomSD::DrawAll()
{
}

void GatePhantomSD::PrintAll()
{
}
