/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBufferMessenger_h
#define GateBufferMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;

class GateBuffer;

class GateBufferMessenger: public GatePulseProcessorMessenger
{
  public:
    GateBufferMessenger(GateBuffer* itsPulseProcessor);
    virtual ~GateBufferMessenger();

    inline void SetNewValue(G4UIcommand* aCommand, G4String aString);

  private:
    G4UIcmdWithADoubleAndUnit *m_bufferSizeCmd; //!< set the buffer size
    G4UIcmdWithADoubleAndUnit *m_readFrequencyCmd;   //!< set the read frequency
    G4UIcmdWithABool          *m_modifyTimeCmd;   //!< does buffer modify the time of pulses
    G4UIcmdWithAnInteger      *m_setDepthCmd;   //!< the depth of each individual buffer
    G4UIcmdWithAnInteger      *m_setModeCmd;   //!< buffer readout mode
};

#endif
