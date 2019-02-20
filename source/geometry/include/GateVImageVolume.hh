/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*! \file
  \class GateVImageVolume :
  \brief Base (abstract) class for volumes which represent the data provided by a 3D image of labels and a label to material correspondence table
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef __GateVImageVolume__hh__
#define __GateVImageVolume__hh__

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "GateObjectChildList.hh"
#include "G4VTouchable.hh"
#include "GateImage.hh"
#include <map>
#include <vector>

#include "G4VSolid.hh"
#include "G4Box.hh"
#include "GateHounsfieldMaterialTable.hh"
#include "GateRangeMaterialTable.hh"

class GateVImageVolumeMessenger;

//-----------------------------------------------------------------------------
///  \brief Base (abstract) class for volumes which represent the data provided by a 3D image of labels and a label to material correspondence table
class GateVImageVolume : public GateVVolume
{
public:

  //-----------------------------------------------------------------------------
  /// The type of label
  typedef int LabelType;
  /// The type of label images
  typedef GateImage ImageType;

  /// The type of label to material name correspondence table
  typedef std::map<LabelType,G4String> LabelToMaterialNameType;

  virtual G4double GetHalfDimension(size_t axis);

  virtual G4LogicalVolume* ConstructOwnSolidAndLogicalVolume(G4Material*, G4bool)=0;
  virtual void DestroyOwnSolidAndLogicalVolume();

  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Constructor with :
  /// the path to the volume to create (for commands)
  /// the name of the volume to create
  /// Creates the messenger associated to the volume
  GateVImageVolume(const G4String& name,G4bool acceptsChildren,G4int depth);

  /// Destructor
  virtual ~GateVImageVolume();
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Method which is called after the image file name and the label
  /// to material file name have been set
  virtual void ImageAndTableFilenamesOK() = 0;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// If user set the IsoCenter, update the Position
  void UpdatePositionWithIsoCenter();
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Sets the name of the Image file
  void SetImageFilename(const G4String& name);
  /// Sets the name of the LabelToMaterial file
  void SetLabelToMaterialTableFilename(const G4String& name);
  void SetHUToMaterialTableFilename(const G4String& name);
  void SetRangeMaterialTableFilename(const G4String& name);
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Gets the name of the Image file
  G4String GetImageFilename() const { return mImageFilename; }
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  void SetIsoCenter(const G4ThreeVector & i);
  G4ThreeVector GetIsoCenter() const { return mIsoCenter; }
  void SetIsoCenterRotationFlag(G4bool b) { mIsoCenterRotationFlag = b; }
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Returns the volume's half size
  inline G4ThreeVector GetHalfSize() const { return mHalfSize; }
  /// Gets the Image
  inline ImageType* GetImage() { return pImage; }
  inline const ImageType* GetImage() const { return pImage; }

  /// Returns the volume's transform matrix
  inline G4RotationMatrix GetTransformMatrix() const { return mTransformMatrix; }

  /// Returns the label at point p
  //  inline LabelType GetLabel( G4ThreeVector p ) { return (LabelType)pImage->GetValue(p); }
  /// Returns the label at voxel of index
  //  inline LabelType GetLabel( int index ) { return (LabelType)pImage->GetValue(index); }
  /// Returns the label at voxel of coordinates i,j,k
  //inline LabelType GetLabel( int i, int j, int k ) { return (LabelType)pImage->GetValue(i,j,k); }
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Returns the material name corresponding to a label
  G4String GetMaterialNameFromLabel(LabelType l) {
    LabelToMaterialNameType::iterator mi = mLabelToMaterialName.find(l);
    if (mi == mLabelToMaterialName.end()) {
      G4cerr << "GateVImageVolume<"<<GetObjectName()<<">::GetMaterialNameFromLabel : Could not find material of label "<<l<<" in correspondence table\n";
      exit(0);
    }
    return (*mi).second ;
  }
  //-----------------------------------------------------------------------------
  inline G4VisAttributes* GetMaterialAttributes(G4Material* m){
    return m_voxelAttributesTranslation[m];
  }

  typedef std::pair<std::pair<G4double,G4double>,G4String> GateVoxelMaterialTranslationRange;
  typedef std::vector<GateVoxelMaterialTranslationRange>   GateVoxelMaterialTranslationRangeVector;
  GateVoxelMaterialTranslationRangeVector                  m_voxelMaterialTranslation;

  typedef std::map<G4Material*, G4VisAttributes*>          GateVoxelAttributesTranslationMap;
  GateVoxelAttributesTranslationMap                        m_voxelAttributesTranslation;

  //-----------------------------------------------------------------------------
  /// Builds a label to material map
  void BuildLabelToG4MaterialVector( std::vector<G4Material*>& );
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  // IO
  void PrintInfo();
  //-----------------------------------------------------------------------------

  /// Returns the material of the voxel of coordinates i,j,k
  //return mLabelToG4Material[ pImage->GetValue(i,j,k) ]; }

  //-----------------------------------------------------------------------------
  // Used by the navigator
  int GetNextVoxel(const G4ThreeVector& position,
                   const G4ThreeVector& direction);

  virtual void GetPhysVolForAVoxel(const G4int, const G4VTouchable &, G4VPhysicalVolume **, G4NavigationHistory &) const {}
  //-----------------------------------------------------------------------------

  G4VSolid * GetSolid(){return pBoxSolid; }

  void SetBuildDistanceTransfoFilename(G4String filename);
  void SetLabeledImageFilename(G4String filename);
  void SetDensityImageFilename(G4String filename);
  void SetMassImageFilename   (G4String filename) {mMassImageFilename = filename;}
  void EnableBoundingBoxOnly(bool b);
  void SetMaxOutOfRangeFraction(double f);

protected:

  //-----------------------------------------------------------------------------
  /// Loads the image
  /// If add1VoxelMargin is true then a margin of one voxel in each direction is added to the image (the margin voxels have the value -1).
  void LoadImage(bool add1VoxelMargin);
  /// Loads the LabelToMaterial file
  void LoadImageMaterialsTable();
  void LoadImageMaterialsFromHounsfieldTable();
  void LoadImageMaterialsFromLabelTable();
  void LoadImageMaterialsFromRangeTable();
  /// The name of the LabelToMaterial file
  G4String mLabelToImageMaterialTableFilename;
  G4String mHounsfieldToImageMaterialTableFilename;
  G4String mRangeToImageMaterialTableFilename;
  bool mLoadImageMaterialsFromHounsfieldTable;
  bool mLoadImageMaterialsFromLabelTable;
  GateHounsfieldMaterialTable mHounsfieldMaterialTable;
  GateRangeMaterialTable mRangeMaterialTable;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  // Usefull :
  /// Builds a vector of the labels in the image
  void BuildLabelsVector( std::vector<LabelType>& );
  /// Remaps the labels form 0 to NbLabels-1. The vector is the vector of labels which has been computed by BuildLabelsVector() : it is updated from 0 to NbLabels-1. If marginAdded is true then assigns the new label 0 to the label -1 which was created for margin voxels (see LoadImage()).
  void RemapLabelsContiguously( std::vector<LabelType>&, bool marginAdded );
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Build distance map
  void BuildDistanceTransfo();
  G4String mDistanceTransfoOutput;
  bool mBuildDistanceTransfo;
  //-----------------------------------------------------------------------------

  bool mWriteHLabelImage;
  G4String mHLabelImageFilename;
  void DumpHLabelImage();
  //-----------------------------------------------------------------------------
  bool mWriteDensityImage;
  G4String mDensityImageFilename;
  void DumpDensityImage();
  G4String mMassImageFilename;
  void DumpMassImage();
  bool mImageMaterialsFromHounsfieldTableDone;
  bool mImageMaterialsFromRangeTableDone;

  //-----------------------------------------------------------------------------
  /// The name of the Image file
  G4String mImageFilename;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Pointer on label image
  ImageType* pImage;
  /// LabelToMaterialName
  LabelToMaterialNameType mLabelToMaterialName;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Half size of the volume
  /// *** BEWARE : this is less than Half Size of the image stored
  /// *** because a margin of 1 voxel in each direction is added to
  /// *** the image stored ***
  G4ThreeVector mHalfSize;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// IsoCenter
  G4ThreeVector    mIsoCenter;
  G4bool           mIsoCenterIsSetByUser;
  G4bool           mIsoCenterRotationFlag;
  G4ThreeVector    mInitialTranslation;
  G4RotationMatrix mTransformMatrix;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  G4Box* pBoxSolid;
  G4LogicalVolume* pBoxLog;
  G4VPhysicalVolume* pBoxPhys;

  //-----------------------------------------------------------------------------
  bool mIsBoundingBoxOnlyModeEnabled;
  unsigned int mUnderflow;
  unsigned int mOverflow;
  double mMaxOutOfRangeFraction;
};
// EO class GateVImageVolume
//-----------------------------------------------------------------------------

#endif
