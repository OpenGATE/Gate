/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePulseAdder_h
#define GatePulseAdder_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GatePulseAdderMessenger;

/*! \class  GatePulseAdder
    \brief  Pulse-processor for adding/grouping pulses per volume.

    - GatePulseAdder - by Daniel.Strul@iphe.unil.ch

    - For each volume where there was one or more input pulse, we get exactly
      one output pulse, whose energy is the sum of all the input-pulse energies,
      and whose position is the centroid of the input-pulse positions.

      \sa GateVPulseProcessor
*/
class GatePulseAdder : public GateVPulseProcessor
{
  public:

    //! Constructs a new pulse-adder attached to a GateDigitizer
    GatePulseAdder(GatePulseProcessorChain* itsChain,const G4String& itsName);

    //! Destructor
    virtual ~GatePulseAdder();

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
    GatePulseAdderMessenger *m_messenger;     //!< Messenger
};


#endif
