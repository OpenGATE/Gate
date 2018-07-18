/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#ifndef GATESINGLEFIXEDFORCEDDECTECTIONACTOR_HH
#define GATESINGLEFIXEDFORCEDDECTECTIONACTOR_HH

// Gate
#include "GateSingleFixedForcedDetectionActorMessenger.hh"
#include "GateFixedForcedDetectionActor.hh"


class GateSingleFixedForcedDetectionActorMessenger;
class GateSingleFixedForcedDetectionActor : public GateFixedForcedDetectionActor
{
public:

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateSingleFixedForcedDetectionActor)

  GateSingleFixedForcedDetectionActor(G4String name, G4int depth=0);
  virtual ~GateSingleFixedForcedDetectionActor();

  // Constructs the actor
  virtual void Construct();

  // Callbacks
  virtual void BeginOfRunAction(const G4Run*);

  void SetSingleInteractionFilename(G4String name) { mSingleInteractionFilename = name; }
  void SetSingleInteractionType(G4String type) { mSingleInteractionType = type; }
  void SetSingleInteractionPosition(G4ThreeVector pos) { mSingleInteractionPosition = pos; }
  void SetSingleInteractionDirection(G4ThreeVector dir) { mSingleInteractionDirection = dir; }
  void SetSingleInteractionEnergy(G4double e) { mSingleInteractionEnergy = e; }
  void SetSingleInteractionZ(G4int z) { mSingleInteractionZ = z; }

protected:
  GateSingleFixedForcedDetectionActorMessenger * pActorMessenger;

  InputImageType::Pointer mSingleInteractionImage;
  G4String                mSingleInteractionFilename;
  G4String                mSingleInteractionType;
  G4ThreeVector           mSingleInteractionPosition;
  G4ThreeVector           mSingleInteractionDirection;
  G4double                mSingleInteractionEnergy;
  G4int                   mSingleInteractionZ;
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(SingleFixedForcedDetectionActor, GateSingleFixedForcedDetectionActor)


#endif /* end #define GATESINGLEFIXEDFORCEDDECTECTIONACTOR_HH */

#endif // GATE_USE_RTK
