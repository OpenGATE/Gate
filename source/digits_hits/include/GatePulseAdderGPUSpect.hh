/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePulseAdderGpuSpect_h
#define GatePulseAdderGpuSpect_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GatePulseAdderGPUSpectMessenger;


class GatePulseAdderGPUSpect : public GateVPulseProcessor
{
  public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GatePulseAdderGPUSpect(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GatePulseAdderGPUSpect();

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the pulse adder
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    GatePulseAdderGPUSpectMessenger *m_messenger;     //!< Messenger
};


#endif
