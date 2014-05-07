/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateAngleFilter.hh"



//---------------------------------------------------------------------------
GateAngleFilter::GateAngleFilter(G4String name)
  :GateVFilter(name)
{
  //mDirection;
  mAngle = 360.;
  pAngleMessenger = new GateAngleFilterMessenger(this);

}
//---------------------------------------------------------------------------



//---------------------------------------------------------------------------
G4bool GateAngleFilter::Accept(const G4Track* aTrack) 
{

  G4ThreeVector stepdirection = aTrack->GetMomentumDirection();
  if(stepdirection.x()*mDirection.x() 
     + stepdirection.y()*mDirection.y() 
     + stepdirection.z()*mDirection.z() < std::cos(mAngle) ) return false;

  return true;
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateAngleFilter::show(){
  G4cout << "------Filter: "<<GetObjectName()<<" particle list------"<<G4endl;
  G4cout << " Direction   "<< mDirection  <<G4endl;
  G4cout << " Angle   "<< mAngle  <<G4endl;
  G4cout << "-------------------------------------------"<<G4endl;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void GateAngleFilter::SetMomentum(G4ThreeVector direction){
  G4double norm = std::sqrt(direction.x()*direction.x() + direction.y()*direction.y() + direction.z()*direction.z());

  mDirection.setX( direction.x()/norm );
  mDirection.setY( direction.y()/norm );
  mDirection.setZ( direction.z()/norm );
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateAngleFilter::SetAngle(G4double angle){
  mAngle = angle;
}
//---------------------------------------------------------------------------
