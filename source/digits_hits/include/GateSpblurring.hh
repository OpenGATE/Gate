/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateSpblurring_h
#define GateSpblurring_h 1

#include "globals.hh"
#include <iostream>
#include <vector>
#include "G4ThreeVector.hh"

#include "GateVPulseProcessor.hh"
#include "G4VoxelLimits.hh"


class GateSpblurringMessenger;

/*! \class  GateSpblurring
    \brief  Pulse-processor for simulating a Gaussian blurring on the position.

    - GatePulseAdder - by By Steven.Staelens@rug.ac.be

      \sa GateVPulseProcessor
*/
class GateSpblurring : public GateVPulseProcessor
{
  public:

    //! \name constructors and destructors
    //@{

    //! Constructs a new spblurring attached to a GateDigitizer
    GateSpblurring(GatePulseProcessorChain* itsChain,
			       const G4String& itsName,
		    G4double itsSpblurring=0) ;

    //! Destructor
    virtual ~GateSpblurring() ;
    //@}

     //! \name getters and setters
    //@{
    //! This function returns the resolution in use.
    G4double GetSpresolution()   	       { return m_spresolution; }

    //! This function sets the spresolution of a gaussian spblurring.
    /*!
      If you want a resolution of 10%, SetSpresolution(0.1)
      \param val is a real number
    */
    void SetSpresolution(G4double val)   { m_spresolution = val;  }
    //@}

    //! Implementation of the pure virtual method declared by the base class GateClockDependent
    //! print-out the attributes specific of the blurring
    virtual void DescribeMyself(size_t indent);

  protected:
    //! Implementation of the pure virtual method declared by the base class GateVPulseProcessor
    //! This methods processes one input-pulse
    //! It is is called by ProcessPulseList() for each of the input pulses
    //! The result of the pulse-processing is incorporated into the output pulse-list
    void ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList);

  private:
    G4double m_spresolution;   	      	  //!< Simulated spatial resolution

    GateSpblurringMessenger *m_messenger;  //!< Messenger

    G4VoxelLimits limits;
    G4double Xmin, Xmax, Ymin, Ymax, Zmin, Zmax;
    G4AffineTransform at;

};


#endif
