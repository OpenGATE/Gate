/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBoxReplica_h
#define GateBoxReplica_h 1

#include "globals.hh"
#include "GatePVReplica.hh"

class GateBoxReplicaPlacement;

class GateBoxReplica : public GatePVReplica
{
   public:

     GateBoxReplica(GateBoxReplicaPlacement* itsInserter,
                      const G4String& pName,
                      G4LogicalVolume* pLogical,
                      G4LogicalVolume* pMother,
                      const EAxis pAxis,
                      const G4int nReplicas,
                      const G4double width,
		      const G4double offset);
     virtual ~GateBoxReplica();


   protected:
     GateBoxReplicaPlacement* m_Inserter;
};

#endif
