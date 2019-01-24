/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class GateMuMapActor
  \author gsizeg@gmail.com
  \brief Class GateMuMapActor : This actor produces voxelised images of the Mu Map.

*/

#ifndef GATEMUMAPACTOR_HH
#define GATEMUMAPACTOR_HH

#include "GateVImageActor.hh"

class GateMuMapActorMessenger;
class GateMuMapActor : public GateVImageActor
{
public:

  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateMuMapActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateMuMapActor)

  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run*r);
  virtual void EndOfRunAction(const G4Run*); // default action (save)
  virtual void BeginOfEventAction(const G4Event * event);
  virtual void UserSteppingActionInVoxel(const int /*index*/, const G4Step* /*step*/);
  virtual void UserPreTrackActionInVoxel(const int /*index*/, const G4Track* /*track*/);
  virtual void UserPostTrackActionInVoxel(const int /*index*/, const G4Track* /*t*/) {}

  //  Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  void SetEnergy(G4double energy);
  void SetMuUnit(G4double unit);

protected:

  GateMuMapActor(G4String name, G4int depth=0);
  GateMuMapActorMessenger * pMessenger;

  GateImage mMuMapImage;
  GateImageInt mSourceMapImage;

  G4double  mEnergy;
  G4double  mMuUnit;
  G4String mMuMapFilename;
  G4String mSourceMapFilename;
};

MAKE_AUTO_CREATOR_ACTOR(MuMapActor,GateMuMapActor)

#endif /* end #define GATEMUMAPACTOR_HH*/
