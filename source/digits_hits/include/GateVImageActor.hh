/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class GateVImageActor
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEVIMAGEACTOR_HH
#define GATEVIMAGEACTOR_HH

#include "GateVActor.hh"
#include "GateImage.hh"
#include "GateVVolume.hh"
#include "GateImageWithStatistic.hh"
#include "Randomize.hh"

//-----------------------------------------------------------------------------
/// \brief Base (virtual) class for sensor storing data in a 3D matrix
/// (GateImage)
class GateVImageActor: public GateVActor
{
public :
  //-----------------------------------------------------------------------------
  enum StepHitType {PreStepHitType, PostStepHitType, MiddleStepHitType, RandomStepHitType, RandomStepHitTypeCylindricalCS, PostStepHitTypeCylindricalCS};

  //-----------------------------------------------------------------------------
  /// Constructs the class
  GateVImageActor(G4String name, G4int depth=0);

  /// Destructor
  virtual ~GateVImageActor();

  /// Type name of the sensor

  /// Constructs the sensor
  virtual void Construct();

  // When a image is managed by the actor, you must initialize the
  // coordinate system with this function
  void SetOriginTransformAndFlagToImage(GateImageWithStatistic & image);
  void SetOriginTransformAndFlagToImage(GateVImage & image);
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Sets the resolution/voxelsize/halfsize/position of the image
  void SetResolution(G4ThreeVector v);
  void SetVoxelSize(G4ThreeVector v);
  void SetHalfSize(G4ThreeVector v);
  void SetSize(G4ThreeVector v);
  void SetPosition(G4ThreeVector v);
  //void SetPosition(GateVVolume * v);
  /// Sets the type of the hit
  void SetStepHitType(G4String t);
  //-----------------------------------------------------------------------------

  double GetDoselVolume(){return mVoxelSize.x()*mVoxelSize.y()*mVoxelSize.z();}

  // Retreive the image voxel size (vc)
  G4ThreeVector GetVoxelSize();

  //-----------------------------------------------------------------------------
  /// Computes the voxel in which to store the data and invokes
  /// UserHitAction which is responsible for voxel data updating
  virtual void PreUserTrackingAction(const GateVVolume * v, const G4Track*t);
  virtual void PostUserTrackingAction(const GateVVolume * v, const G4Track*t);
  virtual void UserSteppingAction(const GateVVolume * v, const G4Step*);
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Callbacks called when a hits should be add to the image
  virtual void UserSteppingActionInVoxel(const int index, const G4Step* step) = 0;
  virtual void UserPreTrackActionInVoxel(const int index, const G4Track* t) = 0;
  virtual void UserPostTrackActionInVoxel(const int index, const G4Track* t) = 0;
  //-----------------------------------------------------------------------------

  virtual void ResetData();

  static int GetIndexFromStepPosition2(const GateVVolume *,
                                       const G4Step  * step,
                                       const GateImage & image,
                                       const bool mPositionIsSet,
                                       const G4ThreeVector mPosition,
                                       const StepHitType mStepHitType);

protected:

  //-----------------------------------------------------------------------------
  // The messenger
  //GateImageActorMessenger * pMessenger;

  // image information
  G4ThreeVector  mVoxelSize;
  G4ThreeVector  mResolution;
  G4ThreeVector  mHalfSize;
  G4ThreeVector  mPosition;
  G4ThreeVector  mOrigin;
  StepHitType    mStepHitType;
  G4String       mStepHitTypeName;
  GateImage      mImage;
  bool           mVoxelSizeIsSet;
  bool           mResolutionIsSet;
  bool           mHalfSizeIsSet;
  bool           mPositionIsSet;

  int GetIndexFromTrackPosition(const GateVVolume *, const G4Track * track);
  int GetIndexFromStepPosition(const GateVVolume *, const G4Step  * step);

}; // end class GateVImageActor

//-----------------------------------------------------------------------------

#endif /* end #define GATEVIMAGEACTOR_HH */
