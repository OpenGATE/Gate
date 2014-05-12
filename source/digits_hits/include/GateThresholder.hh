/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateThresholder_h
#define GateThresholder_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GateThresholderMessenger;


/*! \class  GateThresholder
    \brief  Pulse-processor modelling a simple threshold discriminator.

    - GateThresholder - by Daniel.Strul@iphe.unil.ch

    - The method ProcessOnePulse of this class models a simple
      threshold discriminator: any input pulse whose energy is above
      the energy threshold is copied into the output pulse-list.
      On the contrary, any input pulse whose energy is beyond this
      threshold is discarded.

      \sa GateVPulseProcessor
*/
class GateThresholder : public GateVPulseProcessor
{
  public:

    //! Constructs a new thresholder attached to a GateDigitizer
    GateThresholder(GatePulseProcessorChain* itsChain,
			       const G4String& itsName, G4double itsThreshold=0) ;
    //! Destructor
    virtual ~GateThresholder() ;

    //! Returns the threshold
    G4double GetThreshold()   	      { return m_threshold; }

    //! Set the threshold
    void SetThreshold(G4double val)   { m_threshold = val;  }

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the thresholder
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList&  outputPulseList);

  private:
    G4double m_threshold;     	      	      //!< Threshold value
    GateThresholderMessenger *m_messenger;    //!< Messenger
};


#endif
