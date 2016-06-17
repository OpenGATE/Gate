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
  G4bool accept = false;
  G4bool acceptparent = false;
  G4bool acceptdirectparent = false;
  GateTrackIDInfo *trackInfo;

  if (thePdef.empty()) {
    accept = true; //if no particles given, setting to true will disable filtering on particle
  } else {
    for ( size_t i = 0; i < thePdef.size(); i++) {
      if ( thePdef[i] == aTrack->GetDefinition()->GetParticleName() ||
           (aTrack->GetDefinition()->GetParticleSubType() == "generic" && thePdef[i] == "GenericIon") )
      {
        nFilteredParticles++;
        accept = true;
        break;
      }
    }
  }
  
  if (thePdefZ.empty()) {
    accept = true; //if no particles given, setting to true will disable filtering on particle
    //G4cout<< "List empty. "<<G4endl;
  } else {
    for ( size_t i = 0; i < thePdefZ.size(); i++) { 
	  //if ( thePdefZ[i] == aTrack->GetDefinition()->GetAtomicNumber())
	  //{ 
		  //G4cout << "Z particle equal to defined Atomic Number " << thePdefZ[i]<< G4endl; 
	  //}
      if ( thePdefZ[i] == aTrack->GetDefinition()->GetAtomicNumber() )
      {
		  //G4cout << "Enters and is simulated: "<< aTrack->GetDefinition()->GetAtomicNumber() << G4endl;
        nFilteredParticles++;
        accept = true;
        break;
      }
      else
      {
		  accept = false;
		  //G4cout << "Particle is rejected: " << aTrack->GetDefinition()->GetAtomicNumber() << G4endl;
	  }
       //G4cout << "Z particle equal to defined Atomic Number " << aTrack->GetDefinition()->GetAtomicNumber()<< G4endl; 
    }
  }
  if (theParentPdef.empty()) {
    acceptparent = true; //if no parents given, setting to true will disable filtering on parent
  } else {
    trackInfo = GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    while (trackInfo)
    {
      for ( size_t i = 0; i < theParentPdef.size(); i++) {
        if ( theParentPdef[i] == trackInfo->GetParticleName())
        {
          nFilteredParticles++;
          acceptparent = true;
          break;
        }
      }
      if (acceptparent == true) break;
      int id = trackInfo->GetParentID();
      trackInfo = GateUserActions::GetUserActions()->GetTrackIDInfo(id);
    }
  }

  if (theDirectParentPdef.empty()) {
    acceptdirectparent = true; //if no directparents given, setting to true will disable filtering on parent
  } else {
    trackInfo = GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    if (trackInfo) {
      for ( size_t i = 0; i < theDirectParentPdef.size(); i++) {
        if ( theDirectParentPdef[i] == trackInfo->GetParticleName())
        {
          nFilteredParticles++;
          acceptdirectparent = true;
          break;
        }
      }
    }
  }

  return accept && acceptparent && acceptdirectparent;
}

//---------------------------------------------------------------------------
void GateParticleFilter::Add(const G4String &particleName)
{
  for ( size_t i = 0; i < thePdef.size(); i++) {
    if ( thePdef[i] == particleName ) return;
  }
  thePdef.push_back(particleName);
}

//---------------------------------------------------------------------------
void GateParticleFilter::AddZ(const G4int &particleZ)
{ 
  for ( size_t i = 0; i < thePdefZ.size(); i++) {
    if ( thePdefZ[i] == particleZ ) return;
  }
  thePdefZ.push_back(particleZ);
}
//---------------------------------------------------------------------------

void GateParticleFilter::AddParent(const G4String &particleName)
{
  for ( size_t i = 0; i < theParentPdef.size(); i++) {
    if ( theParentPdef[i] == particleName ) return;
  }
  theParentPdef.push_back(particleName);
}
//---------------------------------------------------------------------------

void GateParticleFilter::AddDirectParent(const G4String &particleName)
{
  for ( size_t i = 0; i < theDirectParentPdef.size(); i++) {
    if ( theDirectParentPdef[i] == particleName ) return;
  }
  theDirectParentPdef.push_back(particleName);
}

//---------------------------------------------------------------------------
void GateParticleFilter::show() {
  G4cout << "------ Filter: " << GetObjectName() << " ------" << G4endl;
  G4cout << "     particle list:" << G4endl;

  for ( size_t i = 0; i < thePdef.size(); i++) {
    G4cout << thePdef[i] << G4endl;
  }
  G4cout << "     parent particle list:" << G4endl;
  for ( size_t i = 0; i < theParentPdef.size(); i++) {
    G4cout << theParentPdef[i] << G4endl;
  }
  G4cout << "-------------------------------------------" << G4endl;
}
//---------------------------------------------------------------------------
