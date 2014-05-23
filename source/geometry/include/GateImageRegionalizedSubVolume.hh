/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateImageRegionalizedSubVolume :
  \brief  A volume which represents a region of a 3D image of labels (sub-volume of a GateImageRegionalizedVolume)
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef __GateImageRegionalizedSubVolume__hh__
#define __GateImageRegionalizedSubVolume__hh__

#include "GateImageRegionalizedVolume.hh"
#include "GateVVolume.hh"

class GateImageRegionalizedSubVolumeSolid;
class GateImageRegionalizedSubVolumeMessenger;

//====================================================================
/// \brief A volume which represents a region of a 3D image of labels
//(sub-volume of a GateImageRegionalizedVolume)
class GateImageRegionalizedSubVolume : public GateVVolume
{
public:

  //====================================================================
  /// The type of label
  typedef GateImageRegionalizedVolume::LabelType LabelType;
  /// The type of label images
  //  typedef GateImageRegionalizedVolume::ImageType ImageType;
  //====================================================================


  //====================================================================
  /// Constructor with :
  /// the path to the volume to create (for commands)
  /// the name of the volume to create
  /// Creates the messenger associated to the volume
  GateImageRegionalizedSubVolume(const G4String& name,G4bool acceptsChildren,G4int depth);
  /// Destructor
  ~GateImageRegionalizedSubVolume();
  //====================================================================

  //====================================================================
  /// Returns a string describing the type of volume and which is used
  /// for commands
  virtual G4double GetHalfDimension(size_t /*axis*/){return 0.;}
  virtual G4String GetTypeName() { return "ImageRegionalizedRegion"; }

    FCT_FOR_AUTO_CREATOR_VOLUME(GateImageRegionalizedSubVolume)

   virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool);
   virtual void DestroyOwnSolidAndLogicalVolume(){}
  /// Constructs the solid
  //virtual void ConstructSolid();
  //====================================================================

  //====================================================================
  /// Sets the GateImageRegionalizedVolume of which it is a subvolume
  void SetVolume(GateImageRegionalizedVolume* v) { pVolume = v; }
  /// Sets the Image
  //  void SetImage(ImageType* image) { pImage = image; }
  /// Sets the label of the region it represents
  void SetLabel(LabelType label) { mLabel = label; }
  /// Sets the half size of the volume
  //  void SetHalfSize(G4ThreeVector s) { mHalfSize = s; }
  //====================================================================

  //====================================================================
  /// Gets the GateImageRegionalizedVolume of which it is a subvolume
  GateImageRegionalizedVolume* GetVolume() { return pVolume; }
  /// Gets the Image
  //  ImageType* GetImage() const { return pImage; }
  /// Gets the label of the region
  LabelType GetLabel() const { return mLabel; }
  /// Gets the half size of the volume which the half size of its containing volume
  G4ThreeVector GetHalfSize() const { return pVolume->GetHalfSize(); }
  //====================================================================


  //====================================================================
  // Methods used by SubVolumeSolids
  inline EInside Inside(const G4ThreeVector& p) const {
    return pVolume->Inside(p,mLabel);
  }

  inline G4double DistanceToIn(const G4ThreeVector& p, const G4ThreeVector& v) const {
    return pVolume->DistanceToIn(p,v,mLabel);
  }

  inline G4double DistanceToIn(const G4ThreeVector& p) const {
    return pVolume->DistanceToIn(p,mLabel);
  }

  inline G4double DistanceToOut(const G4ThreeVector& p, const G4ThreeVector& v,
								const G4bool calcNorm=false,
								G4bool *validNorm=0, G4ThreeVector *n=0) const {
    return pVolume->DistanceToOut(p,v,calcNorm,validNorm,n,mLabel);
  }

  inline G4double DistanceToOut(const G4ThreeVector& p) const {
    return pVolume->DistanceToOut(p,mLabel);
  }

  inline G4ThreeVector SurfaceNormal(const G4ThreeVector& p) const {
	return pVolume->SurfaceNormal(p, mLabel);
  }

  //====================================================================
  virtual void PropagateGlobalSensitiveDetector();

  //====================================================================
  // IO
 // void PrintInfo();

protected:
  //====================================================================
  /// its messenger
  GateImageRegionalizedSubVolumeMessenger* pMessenger;
  //====================================================================

  /// the regionalized volume in which it is contained
  GateImageRegionalizedVolume* pVolume;

  //====================================================================
  /// The image
  //  ImageType* pImage;
  /// The label of the region
  LabelType mLabel;
  /// The half size of the volume
  //  G4ThreeVector mHalfSize;
  //====================================================================
  GateImageRegionalizedSubVolumeSolid  *pBoxSolid;
  G4LogicalVolume* pBoxLog;
};
// EO class GateImageRegionalizedSubVolume
//====================================================================
MAKE_AUTO_CREATOR_VOLUME(ImageRegionalizedSubVolume,GateImageRegionalizedSubVolume)

#endif
