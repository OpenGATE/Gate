/*!
  \class  GateProductionActor
  \author pierre.gueth@creatis.insa-lyon.fr
*/

#ifndef GATEPRODUCTIONACTOR_HH
#define GATEPRODUCTIONACTOR_HH


#include "GateProductionActorMessenger.hh"
#include "GateVImageActor.hh"

///----------------------------------------------------------------------------
/// \brief Actor displaying nb events/tracks/step
class GateProductionActor : public GateVImageActor
{
public:
  virtual ~GateProductionActor();

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateProductionActor)

  //-----------------------------------------------------------------------------
  // Contruct sensor
  virtual void Construct();

  //-----------------------------------------------------------------------------
  // Save production image
  virtual void SaveData();
  virtual void ResetData();

  //-----------------------------------------------------------------------------
  // Callbacks
  virtual void UserPreTrackActionInVoxel(const int index, const G4Track* track);
  virtual void UserSteppingActionInVoxel(int, const G4Step*) { assert(false); }
  virtual void UserPostTrackActionInVoxel(int, const G4Track*) { assert(false); }

protected:
  GateProductionActor(G4String name, G4int depth=0);
  GateProductionActorMessenger *pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(ProductionActor,GateProductionActor)


#endif /* end #define GATEPRODUCTIONACTOR_HH */
