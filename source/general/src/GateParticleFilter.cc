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
  :GateVFilter(name)
{
  thePdef.clear();
  pPartMessenger = new GateParticleFilterMessenger(this);
  nFilteredParticles = 0;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
GateParticleFilter::~GateParticleFilter()
{
  if(nFilteredParticles==0) GateWarning("No particle has been selected by filter: "<<GetObjectName()); 
  delete pPartMessenger ;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
G4bool GateParticleFilter::Accept(const G4Track* aTrack) 
{
 // if(theParentPdef.size()==0)
 // {
 
    for ( size_t i = 0; i < thePdef.size(); i++){
      if ( thePdef[i] == aTrack->GetDefinition()->GetParticleName() || 
	   (aTrack->GetDefinition()->GetParticleSubType()=="generic" && thePdef[i] == "GenericIon") ) 
      {
         nFilteredParticles++;
         return true;
      }
   }
 // }

 // if(thePdef.size()==0)
//  {    
    GateTrackIDInfo  *trackInfo = GateUserActions::GetUserActions()->GetTrackIDInfo(aTrack->GetParentID());
    while(trackInfo)
    {
      for ( size_t i = 0; i < theParentPdef.size(); i++){
        if ( theParentPdef[i] == trackInfo->GetParticleName()) 
        {
           nFilteredParticles++;
           return true;
        }
      }
      int id = trackInfo->GetParentID();
      trackInfo = GateUserActions::GetUserActions()->GetTrackIDInfo(id);
    }
//  }
 
  return false;
}

//---------------------------------------------------------------------------
void GateParticleFilter::Add(const G4String& particleName)
{
  for ( size_t i = 0; i < thePdef.size(); i++){
    if ( thePdef[i] == particleName ) return;
  }
  thePdef.push_back(particleName);
}

void GateParticleFilter::AddParent(const G4String& particleName)
{
  for ( size_t i = 0; i < theParentPdef.size(); i++){
    if ( theParentPdef[i] == particleName ) return;
  }
  theParentPdef.push_back(particleName);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateParticleFilter::show(){
  G4cout << "------ Filter: "<<GetObjectName()<<" ------"<<G4endl;
  G4cout <<"     particle list:"<<G4endl;

  for ( size_t i = 0; i < thePdef.size(); i++){
    G4cout << thePdef[i] << G4endl;
  }
  G4cout <<"     parent particle list:"<<G4endl;
  for ( size_t i = 0; i < theParentPdef.size(); i++){
    G4cout << theParentPdef[i] << G4endl;
  }
  G4cout << "-------------------------------------------"<<G4endl;

}
//---------------------------------------------------------------------------
