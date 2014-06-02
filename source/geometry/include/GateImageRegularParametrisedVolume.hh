/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateImageRegularParametrisedVolume :
  \brief Descendent of GateVImageVolume which represent the image
  using a G4VPVParametrisation (GateImageParametrisation)
*/

#ifndef __GateImageRegularParametrisedVolume__hh__
#define __GateImageRegularParametrisedVolume__hh__

#include "GateVImageVolume.hh"
#include "G4PVParameterised.hh"
#include "G4PhantomParameterisation.hh"

class GateMultiSensitiveDetector;
class GateImageRegularParametrisedVolumeMessenger;

//-----------------------------------------------------------------------------
///  \brief Descendent of GateVImageVolume which represent the image
///  using a G4VPVParametrisation
class GateImageRegularParametrisedVolume : public GateVImageVolume
{
public:

  //-----------------------------------------------------------------------------
  /// The type of label
  typedef GateVImageVolume::LabelType LabelType;
  /// The type of label images
  typedef GateVImageVolume::ImageType ImageType;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Constructor with :
  /// the path to the volume to create (for commands)
  /// the name of the volume to create
  /// Creates the messenger associated to the volume
  GateImageRegularParametrisedVolume(const G4String& name,G4bool acceptsChildren,G4int depth);
  /// Destructor
  virtual ~GateImageRegularParametrisedVolume();
  //-----------------------------------------------------------------------------
  FCT_FOR_AUTO_CREATOR_VOLUME(GateImageRegularParametrisedVolume)

  //-----------------------------------------------------------------------------
  /// Returns a string describing the type of volume and which is used
  /// for commands
  virtual G4String GetTypeName() { return "ImageRegularParametrised"; }

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
  //virtual void DestroyOwnSolidAndLogicalVolume(){};

  //-----------------------------------------------------------------------------
  /// Method which is called after the image file name and the label
  /// to material file name have been set (callback from
  /// GateVImageVolume)
  virtual void ImageAndTableFilenamesOK() {}
  //-----------------------------------------------------------------------------
  /// Constructs the solid
  virtual void ConstructSolid() {}
  /// Constructs the volume and sub-volumes
  //  virtual void Construct();
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  // IO
  void PrintInfo();
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  void PropagateGlobalSensitiveDetector();
  void PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector * msd);

  //-----------------------------------------------------------------------------
  void SetSkipEqualMaterialsFlag(bool b);
  bool GetSkipEqualMaterialsFlag();

protected:
  // The messenger
  GateImageRegularParametrisedVolumeMessenger* pMessenger;

  G4PVParameterised * mImagePhysVol;
  G4Box             * mVoxelSolid;
  G4LogicalVolume   * mVoxelLog;
  std::vector<G4Material*> mVectorLabel2Material;
  size_t * mImageData;
  bool mSkipEqualMaterialsFlag;

};
// EO class GateImageRegularParametrisedVolume
//-----------------------------------------------------------------------------
MAKE_AUTO_CREATOR_VOLUME(ImageRegularParametrisedVolume,GateImageRegularParametrisedVolume)

#endif
