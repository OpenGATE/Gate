/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateVirtualSegmentationSD.cc for more detals
  */


/*! \class  GateVirtualSegmentationSDMessenger
    \brief  Messenger for the GateVirtualSegmentationSD

    - GateVirtualSegmentationSD - by marc.granado@universite-paris-saclay.fr

    \sa GateVirtualSegmentationSD, GateVirtualSegmentationSDMessenger
*/


#ifndef GateVirtualSegmentationSDMessenger_h
#define GateVirtualSegmentationSDMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateVirtualSegmentationSD;
class G4UIcmdWithAString;

class GateVirtualSegmentationSDMessenger : public GateClockDependentMessenger
{
public:
  
  GateVirtualSegmentationSDMessenger(GateVirtualSegmentationSD*);
  ~GateVirtualSegmentationSDMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateVirtualSegmentationSD* m_VirtualSegmentationSD;

  G4UIcmdWithAString*	nameAxisCmd;

  G4UIcmdWithABool*		useMacroGeneratorCmd;
  G4UIcmdWithADoubleAndUnit*  	pitchCmd;
  G4UIcmdWithADoubleAndUnit*   pitchXCmd;
  G4UIcmdWithADoubleAndUnit*   pitchYCmd;
  G4UIcmdWithADoubleAndUnit*   pitchZCmd;
};

#endif








