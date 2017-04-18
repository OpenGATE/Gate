/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateParticleFilter.hh"
#include "GateUserActions.hh"
#include "GateTrajectory.hh"

//---------------------------------------------------------------------------
GateParticleFilter::GateParticleFilter(G4String name)
  : GateVFilter(name)
{
  thePdef.clear();
  pPartMessenger = new GateParticleFilterMessenger(this);
  nFilteredParticles = 0;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
GateParticleFilter::~GateParticleFilter()
{
  if (nFilteredParticles == 0) GateWarning("No particle has been selected by filter: " << GetObjectName());
  delete pPartMessenger ;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
G4bool GateParticleFilter::Accept(const G4Track *aTrack)
{
  G4bool accept = true;

  // Test the particle type
  if (!thePdef.empty()) {
    accept = false;
    for (size_t i = 0; i < thePdef.size(); i++) {
      if (thePdef[i] == aTrack->GetDefinition()->GetParticleName() ||
          (aTrack->GetDefinition()->GetParticleSubType() == "generic" && thePdef[i] == "GenericIon") ) {
        nFilteredParticles++;
        accept = true;
        break;
      }
    }
  } // end thePdef !empty
  if (!accept) return false;

  // Test the particle Z
  if (!thePdefZ.empty()) {
    accept = false;
    for (size_t i = 0; i < thePdefZ.size(); i++) {
      if (thePdefZ[i] == aTrack->GetDefinition()->GetAtomicNumber()) {
        nFilteredParticles++;
        accept = true;
        break;
      }
    }
  } // end thePdefZ !empty
  if (!accept) return false;

  // Test the parent
  if (!theParentPdef.empty()) {
    accept = false;
    GateTrackIDInfo * trackInfo =
      GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    while (trackInfo) {
      for (size_t i = 0; i < theParentPdef.size(); i++) {
        if (theParentPdef[i] == trackInfo->GetParticleName()) {
          nFilteredParticles++;
          accept = true;
          break;
        }
      }
      if (accept == true) break;
      int id = trackInfo->GetParentID();
      trackInfo = GateUserActions::GetUserActions()->GetTrackIDInfo(id);
    }
  } // end theParentPdef !empty
  if (!accept) return false;


  // Test the directParent
  if (!theDirectParentPdef.empty()) {
    accept = false;
    GateTrackIDInfo * trackInfo =
      GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    if (trackInfo) {
      for (size_t i = 0; i < theDirectParentPdef.size(); i++) {
        if (theDirectParentPdef[i] == trackInfo->GetParticleName()) {
          nFilteredParticles++;
          accept = true;
          break;
        }
      }
    }
  } // end theDirectParentPdef !empty

  return accept;
}

//---------------------------------------------------------------------------
void GateParticleFilter::Add(const G4String &particleName)
{
  for (size_t i = 0; i < thePdef.size(); i++) {
    if (thePdef[i] == particleName ) return;
  }
  thePdef.push_back(particleName);
}

//---------------------------------------------------------------------------
void GateParticleFilter::AddZ(const G4int &particleZ)
{
  for (size_t i = 0; i < thePdefZ.size(); i++) {
    if (thePdefZ[i] == particleZ ) return;
  }
  thePdefZ.push_back(particleZ);
}
//---------------------------------------------------------------------------

void GateParticleFilter::AddParent(const G4String &particleName)
{
  for (size_t i = 0; i < theParentPdef.size(); i++) {
    if (theParentPdef[i] == particleName ) return;
  }
  theParentPdef.push_back(particleName);
}
//---------------------------------------------------------------------------

void GateParticleFilter::AddDirectParent(const G4String &particleName)
{
  for (size_t i = 0; i < theDirectParentPdef.size(); i++) {
    if (theDirectParentPdef[i] == particleName ) return;
  }
  theDirectParentPdef.push_back(particleName);
}

//---------------------------------------------------------------------------
void GateParticleFilter::show() {
  G4cout << "------ Filter: " << GetObjectName() << " ------" << G4endl;
  G4cout << "     particle list:" << G4endl;

  for (size_t i = 0; i < thePdef.size(); i++) {
    G4cout << thePdef[i] << G4endl;
  }
  G4cout << "     parent particle list:" << G4endl;
  for (size_t i = 0; i < theParentPdef.size(); i++) {
    G4cout << theParentPdef[i] << G4endl;
  }
  G4cout << "-------------------------------------------" << G4endl;
}
//---------------------------------------------------------------------------
