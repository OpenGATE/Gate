/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateBoxReplicaInserter_h
#define GateBoxReplicaInserter_h 1

#include "globals.hh"

#include "GateBox.hh"

#include "GateBoxReplica.hh"


class GateParallelBeam;

class GateBoxReplicaPlacement : public GateBox
{
  public:

     GateBoxReplicaPlacement(GateParallelBeam* itsParallelBeamInserter,
      	      	      	      const G4String& itsName,const G4String& itsMaterialName,
      	      	      	      G4double itsLength,G4double itsWidth,G4double itsHeight,
			      G4double itsDelta,EAxis itsAxis,G4int itsReplicaNb);

     virtual ~GateBoxReplicaPlacement();

     virtual void ConstructOwnPhysicalVolume(G4bool flagUpdateOnly);

     virtual void Update(const G4String& itsMaterialName,
        	         G4double itslength,G4double itsHeight,G4double itsWidth,
		         G4double itsDelta,G4int itsReplicaNb);

    inline GateBox* GetBoxCreator() const
      { return (GateBox*) GetCreator(); }

  protected:
    GateParallelBeam *m_ParallelBeamInserter;

    G4String	      m_materialName;
    G4double  	      m_Delta,m_Length,m_Height,m_Width;
    EAxis     	      m_Axis;
    G4int     	      m_ReplicaNb;

    GateBoxReplica*   m_Replica;
};

#endif
