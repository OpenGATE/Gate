/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
   \class GateImageRegionalizedVolume
   \author thibault.frisson@creatis.insa-lyon.fr
           laurent.guigues@creatis.insa-lyon.fr
	   david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GateImageRegionalizedVolume__hh__
#define __GateImageRegionalizedVolume__hh__

#include "GateVImageVolume.hh"
#include "GateImageRegionalizedVolumeSolid.hh"

#define TODO int

class GateImageRegionalizedVolumeMessenger;
class GateImageRegionalizedSubVolume;
class GateMultiSensitiveDetector;

//====================================================================
///  \brief A GateVImageVolume which represents the regions provided
///  by a 3D image of labels as sub-volumes in which particle
///  transport is optimized
///
/// Note that Gate is a "meta" Volume which creates other sub-volumes
/// (of type GateImageRegionalizedSubVolume; one for each different
/// label in the image). However, it does not create them in
/// CreateInstance because it must have parsed the image in order to
/// know which volumes to create. Moreover, it creates sub-volumes
/// whose name is of the form :
/// <ImageRegionalizedVolumeName>-<MaterialName>. Hence it must also
/// know the LabelToMaterial file to do it. As a conclusion the work
/// is done in the method CreateSubVolumes which is called once
/// SetImageFilename and SetLabelToMaterialTableFilename have been
/// called, using the callback ImageAndTableFilenamesOK() of
/// GateVImageVolume. Gate method loads the image, loads the
/// LabelToMaterial file, parses the image and adds to the scene a
/// GateImageRegionalizedSubVolume for each different label and sets
/// its properties (in particular Material).
///
/// The methode PostConstruct is defined to register the physical volumes
/// of all the sub-volume to the optimized navigator.
class GateImageRegionalizedVolume : public GateVImageVolume
{
public:

  //====================================================================
  /// The type of label
  typedef GateVImageVolume::LabelType LabelType;
  /// The type of label images
  typedef GateVImageVolume::ImageType ImageType;
  /// The type of label to material name correspondence table
  typedef GateVImageVolume::LabelToMaterialNameType LabelToMaterialNameType;
  /// The type of distance maps
  typedef GateImage DistanceMapType;
  /// The type of normal vectors images
  //typedef TODO NormalsImageType;
  //====================================================================

  //====================================================================
  /// Constructor with :
  /// the path to the volume to create (for commands)
  /// the name of the volume to create
  /// Creates the messenger associated to the volume
  GateImageRegionalizedVolume(const G4String& name,G4bool acceptsChildren,G4int depth);
  /// Destructor
  ~GateImageRegionalizedVolume();
  //====================================================================
  FCT_FOR_AUTO_CREATOR_VOLUME(GateImageRegionalizedVolume)

  //====================================================================
  /// Returns a string describing the type of volume and which is used
  /// for commands
  virtual G4String GetTypeName() { return "ImageRegionalized"; }

  /// Allocates and returns a new volume
   virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);


  //====================================================================
  /// Method which is called after the image file name and the label
  /// to material file name have been set (callback from
  /// GateVImageVolume)
  virtual void ImageAndTableFilenamesOK();
  //====================================================================


  //====================================================================
  // Methods used by SubVolumeSolids (through SubVolumes)
  EInside Inside(const G4ThreeVector& p, LabelType l);
  G4double DistanceToIn(const G4ThreeVector& p, const G4ThreeVector& v, LabelType l);
  G4double DistanceToIn(const G4ThreeVector& p, LabelType l);
  G4double DistanceToOut(const G4ThreeVector& p, const G4ThreeVector& v,
						 const G4bool calcNorm, G4bool *validNorm, G4ThreeVector *n,
						 LabelType l);
  G4double DistanceToOut(const G4ThreeVector& p, LabelType l);
  G4ThreeVector SurfaceNormal(const G4ThreeVector& p, LabelType l);
  //====================================================================


  //====================================================================
  // Methods used by GateImageRegionalizedVolumeNavigation
  G4double ComputeSafety(const G4ThreeVector& p);
  G4double ComputeStep(const G4ThreeVector& position,
		       const G4ThreeVector& direction,
		       const G4double currentProposedStepLength,
		       //G4double& newSafety,
		       //              G4NavigationHistory& history,
		       G4bool& validExitNormal,
		       G4ThreeVector& exitNormal,
		       G4bool& exiting,
		       G4bool& entering,
		       G4VPhysicalVolume *(*pBlockedPhysical));

  //====================================================================



  //====================================================================
  // Used by navigation to know the next entered subvolume
  // returns 0 if the particle exits the volume
  G4VPhysicalVolume* GetEnteredPhysicalVolume( const G4ThreeVector& globalPoint,
					       const G4ThreeVector* globalDirection );

  //====================================================================
  // See GateVVolume for explanation
  virtual void PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector *);
  virtual void AttachPhantomSD();

  //====================================================================
  /// Sets the name of the distance map file
  void SetDistanceMapFilename(const G4String& name) { mDistanceMapFilename = name; }
  //====================================================================


  //====================================================================
private:

  //====================================================================
  /// its messenger
  GateImageRegionalizedVolumeMessenger* pMessenger;
  //====================================================================

  //====================================================================
  /// Distance map
  DistanceMapType* pDistanceMap;
  //====================================================================

  //====================================================================
  // Tolerance to surface
  G4double kCarTolerance;
  //====================================================================

  //====================================================================
  // Data to compute efficiently Inside and DistanceToIn (SubVolumeSolid)
  /// Last point for which Inside has been computed
  G4ThreeVector mLastInsidePoint;
  /// Is mLastInsidePoint valid ?
  G4bool mLastInsidePointIsValid;
  /// Is the inside vector has been computed by the last DistanceToOut call ?
  G4bool mInsideComputedByLastDTO;
  /// ID of the Track when the last computation has been made
  G4int mLastTrackID;

  /// Last point for which DistanceToIn has been computed
  G4ThreeVector mLastDTIPoint;
  /// Last direction for which DistanceToIn has been computed
  G4ThreeVector mLastDTIDirection;
  /// Is mLastDTIPoint valid ?
  G4bool mLastDTIPointIsValid;

  /// Vector of "Insideness" for SubVolumes
  std::vector<EInside> mInside;
  /// Vector of DistanceToIn for SubVolumes
  std::vector<G4double> mDistanceToIn;
  //====================================================================

  //====================================================================
  /// Creates the sub volumes : called by ImageAndTableFilenamesOK()
  void CreateSubVolumes();
  //====================================================================

  //====================================================================
  /// The vector of sub volumes
  std::vector<GateImageRegionalizedSubVolume*> mSubVolume;
  std::map<LabelType,GateImageRegionalizedSubVolume*> mLabelToSubVolume;
  //====================================================================

  //====================================================================
  /// STILL VOID
  /// Computes the normal vectors to the interfaces (surfaces between regions)
 // void ComputeInterfacesNormals();
  //====================================================================

  //====================================================================
  /// Loads the distance map
  void LoadDistanceMap();
  //====================================================================
  /// The name of the distance map file
  G4String mDistanceMapFilename;
  //====================================================================
};
// EO class GateImageRegionalizedVolume
//====================================================================
MAKE_AUTO_CREATOR_VOLUME(ImageRegionalizedVolume,GateImageRegionalizedVolume)

#endif
