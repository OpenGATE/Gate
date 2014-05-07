/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateTemporalResolution_h
#define GateTemporalResolution_h 1

#include "globals.hh"
#include <iostream>
#include <fstream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"


class GateTemporalResolutionMessenger;


/*! \class  GateTemporalResolution
    \brief  Pulse-processor modeling a Gaussian blurring on the time of the pulse.

    - GateTemporalResolution - by Martin.Rey@epfl.ch (July 2003)

    - The operator chooses the time resolution of the detection chain. This resolution is equivalent
    to the FWHM of the Gaussian distribution.

    \sa GateVPulseProcessor
*/
class GateTemporalResolution : public GateVPulseProcessor
{
  public:

    //! Constructs a new temporal resolution attached to a GateDigitizer
    GateTemporalResolution(GatePulseProcessorChain* itsChain,
			       const G4String& itsName=theTypeName,
			       G4double itsTimeResolution=0.) ;
    //! Destructor
    virtual ~GateTemporalResolution() ;

    //! Returns the time resolution
    G4double GetTimeResolution()   	      { return m_timeResolution; }

    //! Set the time resolution
    void SetTimeResolution(G4double val)   { m_timeResolution = val;  }

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the timeResolutioner
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList&  outputPulseList);

  private:
    G4double m_timeResolution;     	      	      //!< TimeResolution value
    GateTemporalResolutionMessenger *m_messenger;    //!< Messenger
   static const G4String& theTypeName;   //!< Default type-name for all blurrings
};


#endif
