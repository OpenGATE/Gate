/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceGeometrySelectorMessenger_h
#define GateCoincidenceGeometrySelectorMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;

class GateCoincidenceGeometrySelector;

class GateCoincidenceGeometrySelectorMessenger: public GateClockDependentMessenger
{
public:
  GateCoincidenceGeometrySelectorMessenger(GateCoincidenceGeometrySelector* itsGeometrySelector);
  virtual ~GateCoincidenceGeometrySelectorMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  inline GateCoincidenceGeometrySelector* GetGeometrySelector(){ return (GateCoincidenceGeometrySelector*) GetClockDependent(); }

private:
  G4UIcmdWithADoubleAndUnit *maxSCmd; //!< set the max time window
  G4UIcmdWithADoubleAndUnit *maxDeltaZCmd; //!< set the max time window
};

#endif
