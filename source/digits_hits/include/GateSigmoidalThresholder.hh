/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSigmoidalThresholder_h
#define GateSigmoidalThresholder_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GateSigmoidalThresholderMessenger;


/*! \class  GateSigmoidalThresholder
    \brief  Pulse-processor modelling a threshold discriminator based on a sigmoidal function.

    - GateSigmoidalThresholder - by Martin.Rey@epfl.ch (mars 2003)

    - The method ProcessOnePulse of this class models a
      threshold discriminator based on a sigmoidal function
      (sigmoidal is a symmetrical function included between 0 and 1):
      \f[\sigma(E)=\frac{1}{1+exp\big(\alpha;\frac{E-E_0}{E_0}\big)}\f]
      The operator choose the threshold, the percentage of acceptance for this threshold
      and the alpha parameter (which is proportionnel of the slope @ symmetrical point Eo);
      with these parameters and the input pulse energy
      the function is calculated and if it's bigger than a random number included between 0 and 1,
      the pulse is copied into the output pulse-list. On the contrary, the input pulse is discarded.

      \sa GateVPulseProcessor
*/
class GateSigmoidalThresholder : public GateVPulseProcessor
{
  public:

    //! Constructs a new sigmoide thresholder attached to a GateDigitizer
  GateSigmoidalThresholder(GatePulseProcessorChain* itsChain,
			   const G4String& itsName, G4double itsThreshold = 0.,
			   G4double itsAlpha = 1., G4double itsAcceptance = 0.5) ;
    //! Destructor
    virtual ~GateSigmoidalThresholder() ;

    //! Returns the threshold params
    G4double GetThreshold()   	      { return m_threshold; }
    G4double GetThresholdAlpha()   { return m_alpha; }
    G4double GetThresholdPerCent()   { return m_perCent; }

    //! Set the threshold param
    void SetThreshold(G4double val)   { m_threshold = val;  }
    void SetThresholdAlpha(G4double val)   { m_alpha = val;  }
    void SetThresholdPerCent(G4double val)   { m_perCent = val;  }

    //! Sigmoide function
    G4double SigmoideFct(G4double nu) { return 1./(1+exp(-nu)); }

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
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
    G4double m_perCent;                       //!< Threshold per cent
    G4double m_centSigm;                      //!< Number of photoelectrons corresponding of the center of the sigmoidal function
    G4double m_alpha;                         //!< Parameter of the sigmoidal function
    G4bool m_check;
    GateSigmoidalThresholderMessenger *m_messenger;    //!< Messenger
};


#endif
