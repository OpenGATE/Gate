/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidenceBufferMessenger_h
#define GateCoincidenceBufferMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
//class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class GateCoincidenceBuffer;

class GateCoincidenceBufferMessenger: public GateClockDependentMessenger
{
public:
  GateCoincidenceBufferMessenger(GateCoincidenceBuffer* itsBuffer);
  virtual ~GateCoincidenceBufferMessenger();

  inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  inline GateCoincidenceBuffer* GetBuffer(){ return (GateCoincidenceBuffer*) GetClockDependent(); }

private:
  G4UIcmdWithADoubleAndUnit *m_bufferSizeCmd; //!< set the dead time value
  G4UIcmdWithADoubleAndUnit *m_readFrequencyCmd;   //!< set the dead time mode
//  G4UIcmdWithABool          *m_modifyTimeCmd;   //!< does buffer modify the time of pulses
  G4UIcmdWithAnInteger      *m_setModeCmd;   //!<  buffer readout mode

};

#endif
