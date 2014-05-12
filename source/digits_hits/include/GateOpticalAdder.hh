/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_OPTICAL

#ifndef GateOpticalAdder_h
#define GateOpticalAdder_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GateOpticalAdderMessenger;

/*! \class  GateOpticalAdder
    \brief  Pulse-processor for adding/grouping pulses per volume, only pules caused by optical photons are added.

    - GateOpticalAdder - by d.j.vanderlaan@iri.tudelft.nl

    - For each volume where there was one or more input pulse, we get exactly
      one output pulse, whose energy is the total number of input-pulses (not the total energy like in the
      GatePulseAdder, but the total number!), and whose position is the centroid of the input-pulse positions.
*/
class GateOpticalAdder : public GateVPulseProcessor
{
  public:

    //! Constructs a new optical-adder attached to a GateDigitizer
    GateOpticalAdder(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GateOpticalAdder();

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
    GateOpticalAdderMessenger *m_messenger;     //!< Messenger
};


#endif

#endif
