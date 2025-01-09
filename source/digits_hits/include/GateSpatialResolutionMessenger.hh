/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022


/*! \class  GateSpatialResolutionMessenger
    \brief  Messenger for the GateSpatialResolution

    - GateSpatialResolution - by name.surname@email.com

    \sa GateSpatialResolution, GateSpatialResolutionMessenger
*/


#ifndef GateSpatialResolutionMessenger_h
#define GateSpatialResolutionMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateSpatialResolution;
class G4UIcmdWithAString;
class GateSpatialResolutionMessenger : public GateClockDependentMessenger
{
public:
  
  GateSpatialResolutionMessenger(GateSpatialResolution*);
  ~GateSpatialResolutionMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateSpatialResolution* m_SpatialResolution;

  G4UIcmdWithADoubleAndUnit*  	spresolutionCmd;
  G4UIcmdWithADoubleAndUnit*   spresolutionXCmd;
  G4UIcmdWithADoubleAndUnit*   spresolutionYCmd;
  G4UIcmdWithADoubleAndUnit*   spresolutionZCmd;
  G4UIcmdWithAString	*spresolutionXdistribCmd;// Command declaration for 1D X-resolution distribution
  G4UIcmdWithAString	*spresolutionYdistribCmd;// Command declaration for 1D Y-resolution distribution
  G4UIcmdWithAString	*spresolutionXYdistrib2DCmd; // Command declaration for 2D XY-resolution distribution
  G4UIcmdWithAString	*nameAxisCmd;
  G4UIcmdWithAString	*spresolutionDistrib2DCmd; // Command declaration for 2D resolution distribution will be used with nameAxisCmd
  G4UIcmdWithABool* 	confineCmd;


};

#endif








