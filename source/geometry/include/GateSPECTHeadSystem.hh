/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
#ifndef GateSPECTHeadSystem_h
#define GateSPECTHeadSystem_h 1

#include "globals.hh"
#include "GateVSystem.hh"

class GateClockDependentMessenger;
class GateToProjectionSet;
class GateToInterfile;

/*! \class  GateSPECTHeadSystem
    \brief  The GateSPECTHeadSystem is a model for SPECT scanners and gamma cameras
    
    - GateSPECTHeadSystem - by Daniel.Strul@iphe.unil.ch
    
    - A GateSPECTHeadSystem is a model for SPECT scanners and gamma cameras. The component tree
      is a linear hierarchy of 3 components, the base, the crystal and the pixel

    - The base is the SPECT head itself (including casing, colli, shielding...). You may
      have multiple heads by repeating the base with a 'ring-repeater'. If you do so,
      the heads can be at 180, 90, 120 degrees (be careful when you choose the angle: it
      should be a multiple of the angular step of the scanner). You can also have the heads
      rotating using a 'rotation-move'. Note that, if you have multiple heads, the rotation
      axis and the repetition axis (i.e. the scanner axis) must be the same.
      
    - The level below is the crystal, which can be monoblock or pixellated. It must be square
      in shape (round crystals not handled)
    
    - The level below is optional, and is meant to be used for pixellated gamma-cameras.
    
    - Note that this component tree does not include a component for the collimator, as I
      had the feeling reconstruction did not have an urgent need of collimator information. 
      If needed, such a component could be added to this model.

    - When this system is used, it will automatically insert into the vector of output modules
      2 modules dedicated to SPECT, GateToProjectionSet and GateToInterfile, to build
      projections and save them respectively.
      
      \sa GateProjectionSet, GateToProjectionSet, GateToInterfile
*/      
class GateSPECTHeadSystem : public GateVSystem
{
  public:
    GateSPECTHeadSystem(const G4String& itsName);   //!< Constructor
    virtual ~GateSPECTHeadSystem();     	    //!< Destructor

  //! Return the projection set-maker
  GateToProjectionSet*  GetProjectionSetMaker() const
    { return m_gateToProjectionSet; }

  //! Return the Interfile writer
  GateToInterfile*  GetInterfileWriter() const
    { return m_gateToInterfile; }

  //! Return the crystal component
  GateSystemComponent*  GetCrystalComponent() const
    { return m_crystalComponent; }

  //! Return the pixel component
  GateSystemComponent*  GetPixelComponent() const
    { return m_pixelComponent; }

/* PY Descourt 08/09/2009 */  
  void setARFStage(G4String);
  G4int GetARFStage(){ return m_ARFStage; };
/* PY Descourt 08/09/2009 */  

   private:
    GateClockDependentMessenger    	*m_messenger; 	  //!< Messenger

    GateSystemComponent       	*m_crystalComponent;
    GateSystemComponent       	*m_pixelComponent;

    GateToProjectionSet       	*m_gateToProjectionSet;
    GateToInterfile   	      	*m_gateToInterfile;

	G4int m_ARFStage;/* PY Descourt 08/09/2009 */ 
};

#endif

