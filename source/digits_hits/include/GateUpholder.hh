/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateUpholder_h
#define GateUpholder_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GateUpholderMessenger;


/*! \class  GateUpholder
    \brief  Pulse-processor modelling a simple uphold limit.

    - GateUpholder - by Steven.Staelens@rug.ac.be

    - The method ProcessOnePulse of this class models a simple
      uphold limit: any input pulse whose energy is below
      the energy limit is copied into the output pulse-list.
      On the contrary, any input pulse whose energy is above this
      limit is discarded.

      \sa GateVPulseProcessor
*/
class GateUpholder : public GateVPulseProcessor
{
  public:

    //! Constructs a new upholder attached to a GateDigitizer
    GateUpholder(GatePulseProcessorChain* itsChain,
			       const G4String& itsName, G4double itsUphold=0) ;
    //! Destructor
    virtual ~GateUpholder() ;

    //! Returns the uphold
    G4double GetUphold()   	      { return m_uphold; }

    //! Set the uphold
    void SetUphold(G4double val)   { m_uphold = val;  }

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the upholder
    virtual void DescribeMyself(size_t indent=0);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    G4double m_uphold;     	      	      //!< Uphold value
    GateUpholderMessenger *m_messenger;       //!< Messenger
};


#endif
