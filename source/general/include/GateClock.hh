/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateClock_h
#define GateClock_h 1

#include "globals.hh"
#include "G4ios.hh"
#include "GateClockMessenger.hh"

class GateClock
{
public:

  static GateClock* GetInstance() {
    if (pInstance == 0)
      pInstance = new GateClock();
    return pInstance;
  }

  ~GateClock();

  G4double GetTime();
  void     SetTime(G4double aTime);
//dk cluster
  void     SetTimeNoGeoUpdate(G4double aTime);
//dk cluster end

  void SetVerboseLevel(G4int value) { nVerboseLevel = value; };

private:
  
  GateClock();


private:

  static GateClock* pInstance;
 
  G4double mTime;
  GateClockMessenger* pClockMessenger;

  G4int nVerboseLevel;
};

#endif
