/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateCrystalSD.hh"
#include "GateCrystalHit.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4ios.hh"
#include "G4VProcess.hh"

#include "G4TransportationManager.hh"

#include "GateVSystem.hh"
#include "GateRotationMove.hh"
#include "GateOrbitingMove.hh"
#include "GateEccentRotMove.hh"
#include "GateSystemListManager.hh"
#include "GateVVolume.hh"
#include "GateDigitizer.hh"


#include "GateObjectStore.hh"

// Name of the hit collection
const G4String GateCrystalSD::theCrystalCollectionName = "crystalCollection";



//------------------------------------------------------------------------------
// Constructor
GateCrystalSD::GateCrystalSD(const G4String& name)
:G4VSensitiveDetector(name),m_system(0)
{
  collectionName.insert(theCrystalCollectionName);
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// Destructor
GateCrystalSD::~GateCrystalSD()
{
   if(m_systemList) delete m_systemList;
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// Method overloading the virtual method Initialize() of G4VSensitiveDetector
void GateCrystalSD::Initialize(G4HCofThisEvent*HCE)
{
  static int HCID = -1; // Static variable storing the hit collection ID

  // Creation of a new hit collection
  crystalCollection = new GateCrystalHitsCollection
                   (SensitiveDetectorName,theCrystalCollectionName);

  // We store the hit collection ID into the static variable HCID
  if(HCID<0)
  { HCID = GetCollectionID(0); }

  // Add the hit collection to the G4HCofThisEvent
  HCE->AddHitsCollection(HCID,crystalCollection);
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// Implementation of the pure virtual method ProcessHits().
// This methods generates a GateCrystalHit and stores it into the SD's hit collection
//G4bool GateCrystalSD::ProcessHits(G4Step*aStep,G4TouchableHistory*ROhist)
G4bool GateCrystalSD::ProcessHits(G4Step*aStep, G4TouchableHistory*)
{

  // Get the track information
  G4Track* aTrack       = aStep->GetTrack();
  G4int    trackID      = aTrack->GetTrackID();
  G4int    parentID     = aTrack->GetParentID();

  G4String partName     = aTrack->GetDefinition()->GetParticleName();
  G4int    PDGEncoding  = aTrack->GetDefinition()->GetPDGEncoding();

  // Get the step-points
  G4StepPoint  *oldStepPoint = aStep->GetPreStepPoint(),
      	       *newStepPoint = aStep->GetPostStepPoint();


  //  Get the process name
  const G4VProcess* process = newStepPoint->GetProcessDefinedStep();
 G4String processName = ( (process != NULL) ? process->GetProcessName() : G4String() ) ;

  //  For all processes except transportation, we select the PostStepPoint volume
  //  For the transportation, we select the PreStepPoint volume
  const G4TouchableHistory* touchable;
  if ( processName == "Transportation" )
      touchable = (const G4TouchableHistory*)(oldStepPoint->GetTouchable() );
  else
      touchable = (const G4TouchableHistory*)(newStepPoint->GetTouchable() );


  GateVolumeID volumeID(touchable);


  if (volumeID.IsInvalid())
    G4Exception( "GateCrystalSD::ProcessHits", "ProcessHits", FatalException, "could not get the volume ID! Aborting!\n");

  // Get the hit global position
  //Modifs Seb 22-06-2011
  //G4ThreeVector position = oldStepPoint->GetPosition();
  G4ThreeVector position = newStepPoint->GetPosition();

  // Compute the hit local position
  // (It will be in the reference frame of the PreStepPoint volume for a transportation hit)
  G4ThreeVector localPosition = volumeID.MoveToBottomVolumeFrame(position);


  // Get the scanner position and rotation angle
/*  GateSystemComponent* baseComponent = GetSystem()->GetBaseComponent();*/
  GateVSystem* system = FindSystem(volumeID);
  GateSystemComponent* baseComponent = system->GetBaseComponent();
  G4ThreeVector scannerPos = baseComponent->GetCurrentTranslation();
  G4double scannerRotAngle = 0;


  if ( baseComponent->FindRotationMove() )
    scannerRotAngle = baseComponent->FindRotationMove()->GetCurrentAngle();
  else if ( baseComponent->FindOrbitingMove() )
    scannerRotAngle = baseComponent->FindOrbitingMove()->GetCurrentAngle();
  else if ( baseComponent->FindEccentRotMove() )
    scannerRotAngle = baseComponent->FindEccentRotMove()->GetCurrentAngle();


  // deposit energy in the current step
  G4double edep = aStep->GetTotalEnergyDeposit();

 //  G4double E;

  // stepLength of the current step
  G4double stepLength = aStep->GetStepLength();
  // time of the current step
  G4double aTime = newStepPoint->GetGlobalTime();
  // Create a new crystal hit
  GateCrystalHit* aHit = new GateCrystalHit();

  // Store the data already obtained into the hit
  aHit->SetPDGEncoding( PDGEncoding );
  aHit->SetEdep( edep );
  aHit->SetStepLength( stepLength );
  aHit->SetTime( aTime );
  aHit->SetGlobalPos( position );
  aHit->SetLocalPos( localPosition );
  aHit->SetProcess( processName );
  aHit->SetTrackID( trackID );
  aHit->SetParentID( parentID );
  aHit->SetVolumeID( volumeID );
  aHit->SetScannerPos( scannerPos );
  aHit->SetScannerRotAngle( scannerRotAngle );
  aHit->SetSystemID(system->GetItsNumber());

  // Ask the system to compute the output volume ID and store it into the hit

//Seb Modif 24/02/2009
/*  GateOutputVolumeID outputVolumeID = GetSystem()->ComputeOutputVolumeID(aHit->GetVolumeID());*/
  GateOutputVolumeID outputVolumeID = system->ComputeOutputVolumeID(aHit->GetVolumeID());
  aHit->SetOutputVolumeID(outputVolumeID);

  // Insert the new hit into the hit collection
  crystalCollection->insert( aHit );

  return true;
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//! Next method underwent an important modification to be compatible with the multi-system approach
G4int GateCrystalSD::PrepareCreatorAttachment(GateVVolume* aCreator)
{

   GateVSystem* creatorSystem = GateSystemListManager::GetInstance()->FindSystemOfCreator(aCreator->GetCreator());
   if (!creatorSystem) {
      G4cout  << G4endl << G4endl << "[GateCrystalSD::PrepareCreatorAttachment]:" << G4endl
            << "Volume '" << aCreator->GetObjectName() << "' does not belong to any system." << G4endl
            << "Your volume must belong to a system to be used with a crystalSD." << G4endl
            << "Attachment request ignored --> you won't have any hit output from this volume!!!" << G4endl << G4endl;
      return -1;
   }

   if (!m_systemList)
   {
      m_systemList = new GateSystemList;
      AddSystem(creatorSystem);
   }
   else
   {
      if((m_systemList->back()) != creatorSystem)
         AddSystem(creatorSystem);
   }

   return 0;
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
//mhadi_obso[ //! This method is no longer used.
// Set the system to which the SD is attached
void GateCrystalSD::SetSystem(GateVSystem* aSystem)
{
  m_system=aSystem;
//Seb Modif 24/02/2009
  GateDigitizer::GetInstance()->SetSystem(aSystem);
}
//mhadi_obso]
//------------------------------------------------------------------------------
// The next three methods were added for the multi-system approach

// Set the system to which the SD is attached
void GateCrystalSD::AddSystem(GateVSystem* aSystem)
{
   m_systemList->push_back(aSystem);
   GateDigitizer::GetInstance()->AddSystem(aSystem);
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
GateVSystem* GateCrystalSD::FindSystem(GateVolumeID volumeID)
{
   // MP Garcia (24/03/2014) Modif to handle imbricated boxes between the SPECThead volume and the world
    //size_t m = volumeID.size();
    //G4String hitSystemName = volumeID.GetVolume(m - 2)->GetName();
    G4String hitSystemName = volumeID.GetVolume(1)->GetName();

   size_t n = hitSystemName.size();
   hitSystemName.erase(n-5,5);

   GateVSystem* system = FindSystem(hitSystemName);

   return system;
}
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
GateVSystem* GateCrystalSD::FindSystem(G4String& systemName)
{
   G4int index = -1;
   for(size_t i=0; i<m_systemList->size(); i++)
   {
      if(systemName.compare(m_systemList->at(i)->GetOwnName()) == 0)
         index = i;
   }

   return m_systemList->at(index);
}
