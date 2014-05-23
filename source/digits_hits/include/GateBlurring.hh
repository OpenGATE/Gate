/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBlurring_h
#define GateBlurring_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

#include "GateVBlurringLaw.hh"
#include "GateInverseSquareBlurringLaw.hh"

class GateBlurringMessenger;

/*! \class  GateBlurring
    \brief  Pulse-processor for simulating a Gaussian blurring on the energy spectrum.

    - GateBlurring - by Martin.Rey@epfl.ch

      \sa GateVPulseProcessor
*/
class GateBlurring : public GateVPulseProcessor
{
  public:

    //! \name constructors and destructors
    //@{

    //! Constructs a new blurring attached to a GateDigitizer
    GateBlurring(GatePulseProcessorChain* itsChain,
                 const G4String& itsName) ;

    //! Destructor
    virtual ~GateBlurring() ;
    //@}

    //! \name getters and setters
    //@{
    //! This function returns the blurring law in use.
    inline GateVBlurringLaw* GetBlurringLaw()           { return m_blurringLaw; }

    //! This function sets the blurring law for the resolution.
    /*!
      Choose between "linear" and "inverseSquare" blurring law
    */
    inline void SetBlurringLaw(GateVBlurringLaw* law)   { m_blurringLaw = law; }
    //@}


    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the blurring
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    GateVBlurringLaw* m_blurringLaw;
    GateBlurringMessenger *m_messenger;   //!< Messenger

};


#endif
