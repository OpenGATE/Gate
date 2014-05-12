/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateIDFilter.hh"


//---------------------------------------------------------------------------
GateIDFilter::GateIDFilter(G4String name)
  :GateVFilter(name)
{
  mID = 0;
  mParentID = 0; 

  pIDMessenger = new GateIDFilterMessenger(this);
}
//---------------------------------------------------------------------------




//---------------------------------------------------------------------------
G4bool GateIDFilter::Accept(const G4Track* aTrack) 
{

  if(mID!=0) if(aTrack->GetTrackID()!=mID) return false;


  if(mParentID!=0)
       if(aTrack->GetParentID()!=mParentID) return false;

  return true;
}
//---------------------------------------------------------------------------



//---------------------------------------------------------------------------
void GateIDFilter::addParentID(G4int id)
{
  mParentID = id;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateIDFilter::addID(G4int id)
{
  mID = id;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateIDFilter::show(){
  G4cout << "------Filter: "<<GetObjectName()<<" ------"<<G4endl;

  if(mID!=0) G4cout<<"Particle ID= "<<mID<<G4endl;
  if(mParentID!=0) G4cout<<"Parent particle ID= "<<mParentID<<G4endl;

  G4cout << "-------------------------------------------"<<G4endl;

}
//---------------------------------------------------------------------------
