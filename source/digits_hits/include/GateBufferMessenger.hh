/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateBufferMessenger
    \brief  Messenger for the GateBuffer


    \sa GateBuffer, GateBufferMessenger
*/


#ifndef GateBufferMessenger_h
#define GateBufferMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateBuffer;
class G4UIcmdWithAString;
class G4UIdirectory;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithABool;

class GateBufferMessenger : public GateClockDependentMessenger
{
public:
  
  GateBufferMessenger(GateBuffer*);
  ~GateBufferMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateBuffer* m_Buffer;

  G4UIcmdWithADoubleAndUnit *m_BufferSizeCmd; //!< set the Buffer size
  G4UIcmdWithADoubleAndUnit *m_readFrequencyCmd;   //!< set the read frequency
  G4UIcmdWithABool          *m_modifyTimeCmd;   //!< does Buffer modify the time of pulses
  G4UIcmdWithAnInteger      *m_setDepthCmd;   //!< the depth of each individual Buffer
  G4UIcmdWithAnInteger      *m_setModeCmd;   //!< Buffer readout mode


};

#endif








