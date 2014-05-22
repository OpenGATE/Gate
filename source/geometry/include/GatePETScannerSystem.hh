/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePETScannerSystem_h
#define GatePETScannerSystem_h 1

#include "globals.hh"

#include "GateScannerSystem.hh"

class GateClockDependentMessenger;

/*! \class  GatePETScannerSystem
    \brief  The GatePETScannerSystem is a generic-purpose model for PET scanners
    
    - GatePETScannerSystem - by Daniel.Strul@iphe.unil.ch
    
   it just derives from GateScannerSystem and add Coincidences
   which can not be defined for GateScannerSystem, in order
   to be coherent with SPECT systems
*/      
class GatePETScannerSystem : public GateScannerSystem
{
  public:
    GatePETScannerSystem(const G4String& itsName);   //!< Constructor
    virtual ~GatePETScannerSystem(){}     	      	  //!< Destructor

};

#endif

