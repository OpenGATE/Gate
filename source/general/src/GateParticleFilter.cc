/*----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
  std::vector<bool> acceptTemp;

  // Test the particle name, keep the particle if the name is in the list
  acceptTemp.push_back(true);
  if (!thePdef.empty()) {
    acceptTemp.back()=false;
    for (size_t i = 0; i < thePdef.size(); i++) {
      if (thePdef[i] == aTrack->GetDefinition()->GetParticleName() ||
          (aTrack->GetDefinition()->GetParticleSubType() == "generic" && thePdef[i] == "GenericIon") ) {
        nFilteredParticles++;
        acceptTemp.back()=true;
        break;
      }
    }
  } // end thePdef !empty
  if (!acceptTemp.back()) _FILTER_RETURN_WITH_INVERSION false;

  // Test the particle Z, keep the particle if Z is in the list
  acceptTemp.push_back(true);
  if (!thePdefZ.empty()) {
    acceptTemp.back()=false;
    for (size_t i = 0; i < thePdefZ.size(); i++) {
      if (thePdefZ[i] == aTrack->GetDefinition()->GetAtomicNumber()) {
        nFilteredParticles++;
        acceptTemp.back()=true;
        break;
      }
    }
  } // end thePdefZ !empty
  if (!acceptTemp.back()) _FILTER_RETURN_WITH_INVERSION false;

  //// Test the particle A
  acceptTemp.push_back(true);
  if (!thePdefA.empty()) {
    acceptTemp.back()=false;
    for (size_t i = 0; i < thePdefA.size(); i++) {
      if (thePdefA[i] == aTrack->GetDefinition()->GetAtomicMass()) {
        nFilteredParticles++;
        acceptTemp.back()=true;
        break;
      }
    }
  } // end thePdefA !empty
  if (!acceptTemp.back()) _FILTER_RETURN_WITH_INVERSION false;

  // Test the particle PDG
  acceptTemp.push_back(true);
  if (!thePdefPDG.empty()) {
    acceptTemp.back()=false;
    for (size_t i = 0; i < thePdefPDG.size(); i++) {
      if (thePdefPDG[i] == aTrack->GetDefinition()->GetPDGEncoding()) {
        nFilteredParticles++;
        acceptTemp.back()=true;
        break;
      }
    }
  } // end thePdefPFG !empty
  if (!acceptTemp.back()) _FILTER_RETURN_WITH_INVERSION false;

  // Test the parent
  acceptTemp.push_back(true);
  if (!theParentPdef.empty()) {
    acceptTemp.back()=false;
    GateTrackIDInfo * trackInfo =
      GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    while (trackInfo) {
      for (size_t i = 0; i < theParentPdef.size(); i++) {
        if (theParentPdef[i] == trackInfo->GetParticleName()) {
          nFilteredParticles++;
          acceptTemp.back()=true;
          break;
        }
      }
      if (acceptTemp.back() == true) break;
      int id = trackInfo->GetParentID();
      trackInfo = GateUserActions::GetUserActions()->GetTrackIDInfo(id);
    }
  } // end theParentPdef !empty
  if (!acceptTemp.back()) _FILTER_RETURN_WITH_INVERSION false;

  // Test the directParent
  acceptTemp.push_back(true);
  if (!theDirectParentPdef.empty()) {
    acceptTemp.back()=false;
    GateTrackIDInfo * trackInfo =
      GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    if (trackInfo) {
      for (size_t i = 0; i < theDirectParentPdef.size(); i++) {
        if (theDirectParentPdef[i] == trackInfo->GetParticleName()) {
          nFilteredParticles++;
          acceptTemp.back()=true;
          break;
        }
      }
    }
  } // end theDirectParentPdef !empty
  if (!acceptTemp.back()) _FILTER_RETURN_WITH_INVERSION false;

  // Keep the track !
  _FILTER_RETURN_WITH_INVERSION true;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateParticleFilter::Add(const G4String &particleName)
{
  for (size_t i = 0; i < thePdef.size(); i++) {
    if (thePdef[i] == particleName ) return;
  }
  thePdef.push_back(particleName);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateParticleFilter::AddZ(const G4int &particleZ)
{
  for (size_t i = 0; i < thePdefZ.size(); i++) {
    if (thePdefZ[i] == particleZ ) return;
  }
  thePdefZ.push_back(particleZ);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateParticleFilter::AddA(const G4int &particleA)
{
  for (size_t i = 0; i < thePdefA.size(); i++) {
    if (thePdefA[i] == particleA ) return;
  }
  thePdefA.push_back(particleA);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateParticleFilter::AddPDG(const G4int &particlePDG)
{
  for (size_t i = 0; i < thePdefPDG.size(); i++) {
    if (thePdefPDG[i] == particlePDG ) return;
  }
  thePdefPDG.push_back(particlePDG);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateParticleFilter::AddParent(const G4String &particleName)
{
  for (size_t i = 0; i < theParentPdef.size(); i++) {
    if (theParentPdef[i] == particleName ) return;
  }
  theParentPdef.push_back(particleName);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateParticleFilter::AddDirectParent(const G4String &particleName)
{
  for (size_t i = 0; i < theDirectParentPdef.size(); i++) {
    if (theDirectParentPdef[i] == particleName ) return;
  }
  theDirectParentPdef.push_back(particleName);
}
//---------------------------------------------------------------------------


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
