/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateBoxReplicaPlacement.hh"

#include "GateBox.hh"
#include "GateParallelBeam.hh"

GateBoxReplicaPlacement::GateBoxReplicaPlacement(GateParallelBeam* itsParallelBeamInserter,
					       const G4String& itsName,const G4String& itsMaterialName,
					       G4double itsLength,G4double itsWidth,G4double itsHeight,
			      		       G4double itsDelta,EAxis itsAxis,G4int itsReplicaNb)
  : GateBox(itsName,itsMaterialName,itsLength,itsWidth,itsHeight,false,false),
    m_ParallelBeamInserter(itsParallelBeamInserter),
    m_Delta(itsDelta),m_Length(itsLength),m_Height(itsHeight),
    m_Width(itsWidth),m_Axis(itsAxis),m_ReplicaNb(itsReplicaNb),m_Replica(0)
{
}

GateBoxReplicaPlacement::~GateBoxReplicaPlacement()
{
}

void GateBoxReplicaPlacement::ConstructOwnPhysicalVolume(G4bool flagUpdateOnly)
{
  if (!flagUpdateOnly) {
    
    m_Replica = new GateBoxReplica( this,
      	      	      	      	      mPhysicalVolumeName,
                              	      GetCreator()->GetLogicalVolume(),
                              	      pMotherLogicalVolume,
				      m_Axis,
                              	      m_ReplicaNb,
                              	      m_Delta,
				      0.0 );
    PushPhysicalVolume(m_Replica);
  }
  else {
    m_Replica->Update(m_Axis,m_ReplicaNb,m_Delta,0.0) ;
  }
}


void GateBoxReplicaPlacement::Update(const G4String& itsMaterialName,G4double itsLength,G4double itsWidth,G4double itsHeight,
	G4double itsDelta,G4int itsReplicaNb)
{
  m_materialName = itsMaterialName;
  m_Length = itsLength;
  m_Width = itsWidth;
  m_Height = itsHeight;
  m_Delta = itsDelta;
  m_ReplicaNb = itsReplicaNb;

  GetBoxCreator()->SetBoxXLength(m_Length);
  GetBoxCreator()->SetBoxYLength(m_Width);
  GetBoxCreator()->SetBoxZLength(m_Height);
  GetBoxCreator()->SetMaterialName(m_materialName);
}
