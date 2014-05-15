/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateLocalBlurring_h
#define GateLocalBlurring_h 1

#include "globals.hh"
#include <iostream>
#include <vector>

#include "GateVPulseProcessor.hh"

#include "GateMaps.hh"

class GateLocalBlurringMessenger;

/*! \class  GateLocalBlurring
    \brief  Pulse-processor for simulating a Gaussian local blurring on the energy spectrum.

    - GateLocalBlurring - by Martin.Rey@epfl.ch (nov 2002)

    - The user can choose a specific blurring for each crystals.
    Each time, he must choose the resolution and the energy of reference.


      \sa GateVPulseProcessor
*/
class GateLocalBlurring : public GateVPulseProcessor
{
  public:

    //! \name constructors and destructors
    //@{

    //! Constructs a new blurring attached to a GateDigitizer
    GateLocalBlurring(GatePulseProcessorChain* itsChain,
			       const G4String& itsName) ;

    //! Destructor
    virtual ~GateLocalBlurring() ;
    //@}

    //! Adds volume to the hashmap and returns 1 if it exists. If it does not exist, returns 0.

    G4int ChooseVolume(G4String val);

    //! \name setters
    //@{
    //! These functions set the resolution of a gaussian blurring for a volume called 'name'.
    /*!
      If you want a resolution of 15% to 511 keV for the volume "LSO", SetResolution(LSO,0.15), SetRefEnergy(LSO,511. keV)
      \param val is a real number
    */
    void SetResolution(G4String name, G4double val)   { m_table[name].resolution = val;  };

    void SetRefEnergy(G4String name, G4double val)   {m_table[name].eref = val; };
    //@}

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
    /*!
      Structure /param which contains the resolution and the energy of reference.
    */
    struct param {
      G4double resolution;
      G4double eref;
    };

    param m_param;                                 //!< Simulated energy resolution and energy of reference
    G4String m_name;                               //! Name of the volume
    GateMap<G4String,param> m_table ;  //! Table which contains the names of volume with their characteristics
    GateMap<G4String,param> ::iterator im;  //! iterator of the gatemap
    GateLocalBlurringMessenger *m_messenger;       //!< Messenger
};


#endif
