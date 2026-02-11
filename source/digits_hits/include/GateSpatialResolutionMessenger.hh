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
  G4UIcmdWithAString	*spresolutionZdistribCmd;// Command declaration for 1D Y-resolution distribution
  G4UIcmdWithAString	*nameAxisCmd;
  G4UIcmdWithAString	*spresolutionXDistrib2DCmd; // per-axis 2D distribution commands will be used nameAxis to determine which axes to apply the distribution to
  G4UIcmdWithAString	*spresolutionYDistrib2DCmd;
  G4UIcmdWithAString	*spresolutionZDistrib2DCmd;
  G4UIcmdWithABool* 	confineCmd;
  G4UIcmdWithABool*		useTruncatedGaussianCmd;


};

#endif








