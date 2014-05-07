/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*! \file GateOpticalSystem.hh
   Created on   2012/07/09  by vesna.cuplov@gmail.com
   Implemented new class GateOpticalSystem for Optical photons: very similar to SPECT. 
    - 3 components: the base, the crystal and the pixel
    - The base is the OPTICAL camera head itself. 
    - The level below is the crystal, which can be monoblock or pixellated.
    - The level below is optional, and is meant to be used for pixellated cameras.
*/  

#ifndef GateOpticalSystem_h
#define GateOpticalSystem_h 1

#include "globals.hh"
#include "GateVSystem.hh"

class GateClockDependentMessenger;
class GateToProjectionSet;
class GateToOpticalRaw;

class GateOpticalSystem : public GateVSystem
{
  public:
    GateOpticalSystem(const G4String& itsName);   //!< Constructor
    virtual ~GateOpticalSystem();     	    //!< Destructor

  //! Return the projection set-maker
  GateToProjectionSet*  GetProjectionSetMaker() const
    { return m_gateToProjectionSet; }

  //! Return the OpticalRaw writer
  GateToOpticalRaw*  GetOpticalRawWriter() const
    { return m_gateToOpticalRaw; }

  //! Return the crystal component
  GateSystemComponent*  GetCrystalComponent() const
    { return m_crystalComponent; }

  //! Return the pixel component
  GateSystemComponent*  GetPixelComponent() const
    { return m_pixelComponent; }

   private:
    GateClockDependentMessenger    	*m_messenger; 	  //!< Messenger

    GateSystemComponent       	*m_crystalComponent;
    GateSystemComponent       	*m_pixelComponent;

    GateToProjectionSet       	*m_gateToProjectionSet;
    GateToOpticalRaw   	      	*m_gateToOpticalRaw;
};

#endif

