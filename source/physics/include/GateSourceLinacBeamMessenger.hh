/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSourceLinacBeamMessenger_h
#define GateSourceLinacBeamMessenger_h 1

#include "globals.hh"
#include "G4UImessenger.hh"
#include "GateVSourceMessenger.hh"
#include "GateSourceLinacBeam.hh"
#include "GateUIcmdWithAVector.hh"

//-------------------------------------------------------------------------------------------------
class GateSourceLinacBeamMessenger: public GateVSourceMessenger
{
public:
  GateSourceLinacBeamMessenger(GateSourceLinacBeam * source);
  ~GateSourceLinacBeamMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);
    
private:
  GateSourceLinacBeam *  mSource;
  G4UIcmdWith3VectorAndUnit * mRefPosCmd;
  G4UIcmdWithAString * mSourceFromPhSCmd;
  G4UIcmdWithAString * mRmaxCmd;
};

#endif

