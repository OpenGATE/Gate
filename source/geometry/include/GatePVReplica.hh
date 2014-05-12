/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePVReplica_h
#define GatePVReplica_h 1

#include "globals.hh"
#include "G4PVReplica.hh"
#include "GateConfiguration.h"

/*! \class  GatePVReplica
    \brief  Base-class for replicas used in the GATE geometry
    
    - GatePVReplica - by Daniel.Strul@iphe.unil.ch
    
    - This class is an extension of the regular class G4PVReplica.
    
    - Its major feature is that it provides a public method Update()
      so that the replica can be updated rather than rebuilt.
      
    - Only replication along the cartesian axes is allowed    
*/      
class GatePVReplica : public G4PVReplica
{ 
  public:
  
    //! Constructor. 
    GatePVReplica(const G4String& pName,
		  G4LogicalVolume* pLogical,
		  G4LogicalVolume* pMother,
                  const EAxis pAxis,
                  const G4int nReplicas,
		  const G4double width,
                  const G4double offset=0);
    //! Destructor
    virtual ~GatePVReplica();

    //! Method to update the replica's parameters
    virtual void Update(const EAxis pAxis,
                        const G4int nReplicas,
		      	const G4double width,
                        const G4double offset=0);
};

#endif


