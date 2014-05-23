/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*!
  \class  GateImageNestedParametrisedVolume :
  \brief  Descendent of GateVImageVolume which represent the image using a G4VPVParametrisation (GateImageParametrisation)
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GateImageNestedParametrisedVolume__hh__
#define __GateImageNestedParametrisedVolume__hh__

#include "GateVImageVolume.hh"
#include "GateImageNestedParametrisation.hh"
class GateMultiSensitiveDetector;

class GateImageNestedParametrisedVolumeMessenger;

//-----------------------------------------------------------------------------
///  \brief Descendent of GateVImageVolume which represent the image using a G4VPVParametrisation (GateImageParametrisation)
class GateImageNestedParametrisedVolume : public GateVImageVolume
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
  GateImageNestedParametrisedVolume(const G4String& name,G4bool acceptsChildren,G4int depth);
  /// Destructor
  virtual ~GateImageNestedParametrisedVolume();
  //-----------------------------------------------------------------------------
  FCT_FOR_AUTO_CREATOR_VOLUME(GateImageNestedParametrisedVolume)

  //-----------------------------------------------------------------------------
  /// Returns a string describing the type of volume and which is used
  /// for commands
  virtual G4String GetTypeName() { return "ImageNestedParametrised"; }

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
  virtual void GetPhysVolForAVoxel(const G4int index,
				   const G4VTouchable & pTouchable,
				   G4VPhysicalVolume ** pPhysVol,
				   G4NavigationHistory & history) const;
  //-----------------------------------------------------------------------------

  // This function add multi-SD to the sub logical-volume
  virtual void PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector *);

  // This function add 'global' SD (Root output, PhantomSD) to the sub
  // logical-volume (see GateVVolume.hh)
  virtual void PropagateGlobalSensitiveDetector();


protected:
  //-----------------------------------------------------------------------------
  /// its messenger
  GateImageNestedParametrisedVolumeMessenger* pMessenger;
  //-----------------------------------------------------------------------------

  GateImageNestedParametrisation * mVoxelParametrisation;
  G4VPhysicalVolume * mPhysVolX;
  G4VPhysicalVolume * mPhysVolY;
  G4VPhysicalVolume * mPhysVolZ;

  G4LogicalVolume * logXRep;
  G4LogicalVolume * logYRep;
  G4LogicalVolume * logZRep;
};
// EO class GateImageNestedParametrisedVolume
//-----------------------------------------------------------------------------
MAKE_AUTO_CREATOR_VOLUME(ImageNestedParametrisedVolume,GateImageNestedParametrisedVolume)

#endif
