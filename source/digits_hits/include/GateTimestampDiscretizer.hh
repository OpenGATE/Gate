/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTimestampDiscretizer_h
#define GateTimestampDiscretizer_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

class GateTimestampDiscretizerMessenger;


/*! \class  GateTimestampDiscretizer
    \brief  Pulse-processor modelling discrete time sampling.

    - GateTimestampDiscretizer - by gergely.patay@gmail.com
    - The method ProcessOnePulse of this class converts a
    	(quasi)continuous time signal into a discrete time sampled
    	pulse chain.
      \sa GateVPulseProcessor
*/
class GateTimestampDiscretizer : public GateVPulseProcessor
{
  public:

    //! Constructs a new time discretizer attached to a GateDigitizer
    GateTimestampDiscretizer(GatePulseProcessorChain* itsChain,
			       const G4String& itsName, G4double itsSamplingtime=0) ;
    //! Destructor
    virtual ~GateTimestampDiscretizer() ;

    //! Returns the sampling frequency
    G4double GetSamplingFrequency()   	      { return 1.0/m_samplingtime; }

    //! Returns the sampling time
    G4double GetSamplingTime()   	      { return m_samplingtime; }

    //! Set the sampling frequency
    void SetSamplingFrequency(G4double val)   { m_samplingtime = 1.0/val;  }

    //! Set the sampling time
    void SetSamplingTime(G4double val)   { m_samplingtime = val;  }


    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the time sampler
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList&  outputPulseList);

  private:
    G4double m_samplingtime;     	      	      //!< Sampling time value
    GateTimestampDiscretizerMessenger *m_messenger;    //!< Messenger
};


#endif
