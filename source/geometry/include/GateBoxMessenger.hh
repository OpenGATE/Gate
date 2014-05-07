/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBoxMessenger_h
#define GateBoxMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateBox;

//---------------------------------------------------------------------------
class GateBoxMessenger: public GateVolumeMessenger
{
public:
  GateBoxMessenger(GateBox*);
  ~GateBoxMessenger();
    
  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateBox* GetBoxCreator() { return (GateBox*)GetVolumeCreator(); }

protected:
  G4UIcmdWithADoubleAndUnit* pBoxXLengthCmd;
  G4UIcmdWithADoubleAndUnit* pBoxYLengthCmd;
  G4UIcmdWithADoubleAndUnit* pBoxZLengthCmd;
};
//---------------------------------------------------------------------------

#endif
