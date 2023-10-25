/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateEmptySD.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4Step.hh"
#include "G4SDManager.hh"


//------------------------------------------------------------------------------
// Constructor
GateEmptySD::GateEmptySD(const G4String& name)
:G4VSensitiveDetector(name)
{
	G4String collName=name+"Collection";
	collectionName.insert(collName);


	HCID = G4SDManager::GetSDMpointer()->GetCollectionCapacity() ;

}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Method overloading the virtual method Initialize() of G4VSensitiveDetector
// Called at the beginning of each event
void GateEmptySD::Initialize(G4HCofThisEvent*HCE)
{
	HCE->AddHitsCollection(HCID, nullptr);
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Implementation of the pure virtual method ProcessHits().
G4bool GateEmptySD::ProcessHits(G4Step*aStep, G4TouchableHistory*)
{
  return true;
}