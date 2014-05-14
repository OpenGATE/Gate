/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePVReplica.hh"

GatePVReplica::GatePVReplica(const G4String& pName,
		      	     G4LogicalVolume* pLogical,
		      	     G4LogicalVolume* pMother,
                      	     const EAxis pAxis,
                      	     const G4int nReplicas,
		      	     const G4double width,
                      	     const G4double offset)
  : G4PVReplica(pName,pLogical,pMother,pAxis,nReplicas ,width,offset) 
{
  if ((pAxis!=kXAxis) && (pAxis!=kYAxis) && (pAxis!=kZAxis) ) 
    G4Exception( "GatePVReplica::GatePVReplica", "GatePVReplica", FatalException, "Sorry, an object tried to create a replicate along a non-Caretsian axis, but this kind of replication is not handled by GATE." );
}
			     



GatePVReplica::~GatePVReplica()
{}



void GatePVReplica::Update(const EAxis pAxis,const G4int nReplicas,
		      	   const G4double width,const G4double offset)
{
  if ((pAxis!=kXAxis) && (pAxis!=kYAxis) && (pAxis!=kZAxis) ) 
    G4Exception( "GatePVReplica::Update", "Update", FatalException, "Sorry, an object tried to create a replicate along a non-Caretsian axis, but this kind of replication is not handled by GATE" );
  faxis=pAxis;

  if (nReplicas<1)
    G4Exception( "GatePVReplica::Update", "Update", FatalException, "Sorry, an object tried to create a replicate with less than one copy!" );
  fnReplicas=nReplicas;

  if (width<0)
    G4Exception("GatePVReplica::Update", "Update", FatalException, "Sorry, an object tried to create a replicate with a negative width!" );
  fwidth=width;

  foffset=offset;
}


