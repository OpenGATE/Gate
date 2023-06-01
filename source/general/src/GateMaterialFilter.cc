/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \brief Class GateMaterialFilter
*/

#include "GateMaterialFilter.hh"

#include "GateUserActions.hh"
#include "GateTrajectory.hh"


//---------------------------------------------------------------------------
GateMaterialFilter::GateMaterialFilter(G4String name)
  :GateVFilter(name)
{
  theMdef.clear();
  pMatMessenger = new GateMaterialFilterMessenger(this);
  nFilteredParticles = 0;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
GateMaterialFilter::~GateMaterialFilter()
{
  if(nFilteredParticles==0) GateWarning("No particle has been selected by filter: " << GetObjectName()); 
  delete pMatMessenger ;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
G4bool GateMaterialFilter::Accept(const G4Step* aStep) 
{
    for ( size_t i = 0; i < theMdef.size(); i++){
      if ( theMdef[i] == aStep->GetPreStepPoint()->GetMaterial()->GetName() ) {
        nFilteredParticles++;
        _FILTER_RETURN_WITH_INVERSION true;
      } 
    }
  _FILTER_RETURN_WITH_INVERSION false;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
G4bool GateMaterialFilter::Accept(const G4Track* aTrack) 
{
    for ( size_t i = 0; i < theMdef.size(); i++){ 	
      if ( theMdef[i] == aTrack->GetMaterial()->GetName() ) {
        nFilteredParticles++;
        _FILTER_RETURN_WITH_INVERSION true;
      } 
    }
  _FILTER_RETURN_WITH_INVERSION false;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateMaterialFilter::Add(const G4String& materialName)
{
  for ( size_t i = 0; i < theMdef.size(); i++ ){
    if ( theMdef[i] == materialName ) return;
  }
  theMdef.push_back(materialName);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateMaterialFilter::show()
{
  G4cout << "------ Filter: " << GetObjectName() << " ------\n";
  G4cout << "       Material list: \n";

  for ( size_t i = 0; i < theMdef.size(); i++ ){
    G4cout << theMdef[i] << Gateendl;
  }
  G4cout << "-------------------------------------------\n";
}
//---------------------------------------------------------------------------
