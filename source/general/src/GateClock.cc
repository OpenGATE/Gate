/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateClock.hh"
#include "GateClockMessenger.hh"
#include "GateDetectorConstruction.hh"

GateClock* GateClock::pInstance = 0; 

//------------------------------------------------------------------------------------
GateClock::GateClock() : mTime(0.)
{
  nVerboseLevel = 0;
  if(pInstance != 0) { G4Exception( "GateClock::GateClock", "GateClock", FatalException, "constructed twice."); }
  pClockMessenger = new GateClockMessenger();
}
//------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------
GateClock::~GateClock() 
{
  delete pClockMessenger;
}
//------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------
void GateClock::SetTime(G4double aTime) 
{
  mTime = aTime;
  if (nVerboseLevel>0) GateMessage("Time", 1, "Time set to (s) " << mTime/s << G4endl);

  GateDetectorConstruction::GetGateDetectorConstruction()->ClockHasChanged();
}
//------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------
void GateClock::SetTimeNoGeoUpdate(G4double aTime) 
{
  mTime = aTime;
  if (nVerboseLevel>0) GateMessage("Time", 1, "Time set to (s) without Geometry update " << mTime/s << G4endl);
}
//------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------
G4double GateClock::GetTime() 
{
  return mTime;
}
//------------------------------------------------------------------------------------


