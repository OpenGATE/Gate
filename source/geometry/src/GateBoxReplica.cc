/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBoxReplica.hh"

#include "GateBoxReplicaPlacement.hh"


GateBoxReplica::GateBoxReplica(GateBoxReplicaPlacement* itsInserter, const G4String& pName,
   G4LogicalVolume* pLogical, G4LogicalVolume* pMother, const EAxis pAxis, const G4int nReplicas,
   const G4double width, const G4double offset)
   : GatePVReplica(pName,pLogical,pMother,pAxis,nReplicas,width,offset), m_Inserter(itsInserter)
{}

GateBoxReplica::~GateBoxReplica()
{}
