/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCrystalBlurring_h
#define GateCrystalBlurring_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"

class GateCrystalBlurringMessenger;
class GateCrystalBlurring : public GateVPulseProcessor
{
  public:

    GateCrystalBlurring(GatePulseProcessorChain* itsChain,
		 const G4String& itsName=theTypeName,
		 G4double itsCrystalblurringmin=0,
		 G4double itsCrystalblurringmax=0,
		 G4double itsCrystalQE=1,
		 G4double itsCrystalEnergyRef=-1) ;

    //! Destructor
    virtual ~GateCrystalBlurring() ;

    void SetCrystalResolutionMin(G4double val)   { m_crystalresolutionmin = val;  }
    void SetCrystalResolutionMax(G4double val)   { m_crystalresolutionmax = val;  }
    void SetCrystalQE(G4double val)              { m_crystalQE = val;  }


    void SetCrystalRefEnergy(G4double eval)   { m_crystaleref = eval; }

    //! Implementation of the pure virtual method declared by the base class GateDigitizerComponent
    //! print-out the attributes specific of the blurring
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    G4double m_crystalresolution;         //!< Simulated energy resolution
    G4double m_crystalresolutionmin;      //!< Simulated min energy resolution
    G4double m_crystalresolutionmax;      //!< Simulated max energy resolution
    G4double m_crystalQE;                 //!< Simulated "crystal" QE
    G4double m_crystaleref;                      //!< Simulated energy of reference
    G4double m_crystalcoeff;                     //!< Coefficient which connects energy to the resolution

    GateCrystalBlurringMessenger *m_messenger;   //!< Messenger

    static const G4String& theTypeName;   //!< Default type-name for all blurrings

};


#endif
