/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateDiscretizer_h
#define GateDiscretizer_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GateDiscretizerMessenger;
class GateOutputVolumeID;

/*! \class  GateDiscretizer
    \brief  Pulse-processor modelling a the lake of knowledge of "where" a hit occurs inside a crystal

    - GateDiscretizer - by dguez@cea.fr

      \sa GateVPulseProcessor
*/
class GateDiscretizer : public GateVPulseProcessor
{
  public:

    //! Constructs a new Discretizer attached to a GateDigitizer
    GateDiscretizer(GatePulseProcessorChain* itsChain,const G4String& itsName) ;

    //! Destructor
    virtual inline ~GateDiscretizer() ;

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the Discretizer
    virtual void DescribeMyself(size_t indent);


  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:

    GateDiscretizerMessenger *m_messenger;	  //!< Messenger for this Discretizer
};


#endif
