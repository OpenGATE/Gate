/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateVFilter.hh"
#include "GateMessageManager.hh"

//-------------------------------------------------------------
GateVFilter::GateVFilter(G4String name)
  :GateNamedObject(name)
{
 
}
//-------------------------------------------------------------

//-------------------------------------------------------------
G4bool GateVFilter::Accept(const G4Step* aStep) 
{

  G4Track* aTrack = aStep->GetTrack();

  if(!Accept(aTrack) ) return false;

  return true;

}
//-------------------------------------------------------------

//-------------------------------------------------------------
G4bool GateVFilter::Accept(const G4Track* /*aTrack*/) 
{
  return true;
}
//-------------------------------------------------------------

//-------------------------------------------------------------
void GateVFilter::show(){
  G4cout << "------Filter: "<<GetObjectName()<<" particle list------\n";

 

  G4cout << "-------------------------------------------\n";

}
//-------------------------------------------------------------


