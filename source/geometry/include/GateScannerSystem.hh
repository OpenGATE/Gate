/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateScannerSystem_h
#define GateScannerSystem_h 1

#include "globals.hh"

#include "GateVSystem.hh"

class GateClockDependentMessenger;

/*! \class  GateScannerSystem
    \brief  The GateScannerSystem is a generic-purpose model for PET scanners
    
    - GateScannerSystem - by Daniel.Strul@iphe.unil.ch
    
    - A GateScannerSystem is a very generalistic model of PET scanners. The component tree
      is a linear hierarchy of 6 components (base, level1, level2, level3, level4, level5).
      These components don't have any specific properties, so that this system may be use for
      everything... and nothing!
*/      
class GateScannerSystem : public GateVSystem
{
  public:
    GateScannerSystem(const G4String& itsName);   //!< Constructor
    virtual ~GateScannerSystem();     	      	  //!< Destructor

   private:
    GateClockDependentMessenger    	*m_messenger; 	  //!< Messenger
};

#endif

