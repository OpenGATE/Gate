/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*! \file
  \brief Implementation of GateVImageVolume
*/


#include <pthread.h>
#include <set>

#include "GateVImageVolume.hh"
#include "GateMiscFunctions.hh"
#include "GateMessageManager.hh"
#include "GateDetectorConstruction.hh"
#include "GateDMapVol.h"
#include "GateDMaplongvol.h"
#include "GateDMapdt.h"
#include "GateHounsfieldMaterialTable.hh"
#include <G4TransportationManager.hh>
#include "globals.hh"

typedef unsigned int uint;

//--------------------------------------------------------------------
/// Constructor with :
/// the path to the volume to create (for commands)
/// the name of the volume to create
/// Creates the messenger associated to the volume
GateVImageVolume::GateVImageVolume( const G4String& name,G4bool acceptsChildren,G4int depth) :
  GateVVolume(name,acceptsChildren,depth)
{
  GateMessageInc("Volume",5,"Begin GateVImageVolume("<<name<<")\n");
  mImageFilename="";
  pImage=0;
  mHalfSize = G4ThreeVector(0,0,0);
  mIsoCenterIsSetByUser = false;
  mIsoCenterRotationFlag = false;
  pOwnMaterial = theMaterialDatabase.GetMaterial("Air");
  mBuildDistanceTransfo = false;
  mLoadImageMaterialsFromHounsfieldTable = false;
  mLoadImageMaterialsFromLabelTable = false;
  mLabelToImageMaterialTableFilename = "none";
  mHounsfieldToImageMaterialTableFilename = "none";
  mRangeToImageMaterialTableFilename = "none";
  mWriteHLabelImage     = false;
  mWriteDensityImage    = false;
  mHLabelImageFilename  = "none";
  mDensityImageFilename = "none";
  mMassImageFilename    = "none";
  mIsBoundingBoxOnlyModeEnabled = false;
  mImageMaterialsFromHounsfieldTableDone = false;
  mUnderflow = 0;
  mOverflow = 0;
  mMaxOutOfRangeFraction = 0.0;
  GateMessageDec("Volume",5,"End GateVImageVolume("<<name<<")\n");

  // do not display all voxels, only bounding box
  pOwnVisAtt->SetDaughtersInvisible(true);
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
/// Destructor
GateVImageVolume::~GateVImageVolume()
{
  GateMessageInc("Volume",5,"Begin ~GateVImageVolume()\n");
  if (pImage) delete pImage;
  //if (pBoxPhys) delete pBoxPhys;
  if (pBoxLog) delete pBoxLog;
  if (pBoxSolid) delete pBoxSolid;
  GateMessageDec("Volume",5,"End ~GateVImageVolume()\n");
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::EnableBoundingBoxOnly(bool b) {
  mIsBoundingBoxOnlyModeEnabled = b;
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::SetMaxOutOfRangeFraction(double f){
  GateMessage("Volume",3,"setting a new threshold for the fraction of voxels with out-of-range HU values: " << f);
  if ( (f>=0.) && (f<=1.) ){
    mMaxOutOfRangeFraction=f;
    return;
  }
  GateError("Fraction should be given as a number between 0.0 and 1.0!!");
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::SetIsoCenter(const G4ThreeVector & i)
{
  mIsoCenter = i;
  mIsoCenterIsSetByUser = true;
  GateMessage("Volume",5,"IsoCenter = " << mIsoCenter << Gateendl);
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::UpdatePositionWithIsoCenter()
{
  if (mIsoCenterIsSetByUser) {
    static int n=0;
    if (n==0) {
      mInitialTranslation = GetVolumePlacement()->GetTranslation();
      n++;
    }

    const G4ThreeVector & tcurrent = mInitialTranslation;//GetVolumePlacement()->GetTranslation();

    GateMessage("Volume",3,"Current T = " << tcurrent << Gateendl);
    GateMessage("Volume",3,"Isocenter = " << GetIsoCenter() << Gateendl);
    GateMessage("Volume",3,"Origin = " << GetOrigin() << Gateendl);
    GateMessage("Volume",3,"Half = " << GetHalfSize() << Gateendl);
    GateMessage("Volume",3,"TransformMatrix = " << mTransformMatrix << Gateendl);

    // Compute translation
    G4ThreeVector q = mIsoCenter - GetOrigin(); // translation from the image corner
    if (mIsoCenterRotationFlag) {
      q -= GetHalfSize(); // translation from the image center
      q = tcurrent - q; // add to previous translation
      // Apply rotation if isocenter is given, consider rotation from this isocenter
      q = mTransformMatrix*q;
    }
    else {
      // Compute translation
      q -= mTransformMatrix*GetHalfSize();
      q = tcurrent - q;
    }
    GateMessage("Volume",3," Translation = " << q << Gateendl);
    GetVolumePlacement()->SetTranslation(q);
  }
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
G4double GateVImageVolume::GetHalfDimension(size_t axis)
{
  if (axis==0) return mHalfSize.x();
  else if (axis==1) return mHalfSize.y();
  else if (axis==2) return mHalfSize.z();
  GateError("Volumes have 3 dimensions! (0,1,2)");
  return -1.;
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
/// Sets the name of the Image file
void GateVImageVolume::SetImageFilename(const G4String& name)
{
  mImageFilename = name;
  if (mLabelToImageMaterialTableFilename != "none") ImageAndTableFilenamesOK();
  if (mHounsfieldToImageMaterialTableFilename != "none") ImageAndTableFilenamesOK();
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
/// Sets the name of the LabelToMaterial file
void GateVImageVolume::SetLabelToMaterialTableFilename(const G4String& name)
{
  if (mHounsfieldToImageMaterialTableFilename != "none") {
    GateError("Please set SetHUToMaterialFile or SetLabelToMaterialFile, not both. Abort.\n");
  }
  mLabelToImageMaterialTableFilename = name;
  mLoadImageMaterialsFromLabelTable = true;
  if (mImageFilename.length()>0) ImageAndTableFilenamesOK();
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Sets the name of the HUToMaterial file
void GateVImageVolume::SetHUToMaterialTableFilename(const G4String& name)
{
  if (mLabelToImageMaterialTableFilename != "none") {
    GateError("Please set SetHUToMaterialFile or SetLabelToMaterialFile, not both. Abort.\n");
  }
  mHounsfieldToImageMaterialTableFilename = name;
  mLoadImageMaterialsFromHounsfieldTable = true;
  if (mImageFilename.length()>0) ImageAndTableFilenamesOK();
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Sets the name of the RangeMaterial file
void GateVImageVolume::SetRangeMaterialTableFilename(const G4String& name)
{
  if (mLabelToImageMaterialTableFilename != "none") {
    GateError("Please set SetHUToMaterialFile or SetLabelToMaterialFile, not both. Abort.\n");
  }
  mRangeToImageMaterialTableFilename = name;
  if (mImageFilename.length()>0) ImageAndTableFilenamesOK();
}
//--------------------------------------------------------------------



//--------------------------------------------------------------------
/// Loads the image
void GateVImageVolume::LoadImage(bool add1VoxelMargin)
{
  GateMessageInc("Volume",4,"Begin GateVImageVolume::LoadImage("<<mImageFilename<<")\n");

  ImageType* tmp = new ImageType;

  if (mImageFilename == "test1" ) {
    // Creates a 32x32x32 image of size 20x20x20 cm3
    // with label 1 in the center 16x16x16 region and 0 elsewhere
    tmp->SetResolutionAndHalfSize(G4ThreeVector(32,32,32),
                                  G4ThreeVector(20*cm,20*cm,20*cm));
    tmp->Allocate();
    int i,j,k;
    for (i=0;i<32;i++)
      for (j=0;j<32;j++)
        for (k=0;k<32;k++)
          tmp->SetValue(i,j,k,0);
    for (i=8;i<24;i++)
      for (j=8;j<24;j++)
        for (k=8;k<24;k++)
          tmp->SetValue(i,j,k,1);
  }
  else if (mImageFilename == "test2" ) {
    // Creates a 32x32x32 image of size 20x20x20 cm3
    // with label 1 in the bottom 32x16x32 region and 0 elsewhere
    tmp->SetResolutionAndHalfSize(G4ThreeVector(32,32,32),
                                  G4ThreeVector(20*cm,20*cm,20*cm));
    tmp->Allocate();
    int i,j,k;
    for (i=0;i<32;i++)
      for (j=0;j<32;j++)
        for (k=0;k<32;k++)
          tmp->SetValue(i,j,k,0);
    for (i=0;i<32;i++)
      for (j=0;j<16;j++)
        for (k=0;k<32;k++)
          tmp->SetValue(i,j,k,1);
  }
  else {
    tmp->Read(mImageFilename);
    //G4cout << mImageFilename << Gateendl;
  }
  //tmp->PrintInfo();

  /// The volume's half size is the half size of the initial image, even if a margin is added
  mHalfSize = tmp->GetHalfSize();

  if (pImage) delete pImage;

  if (add1VoxelMargin) {
    // The image is copied with a margin of 1 voxel in all directions
    pImage = new ImageType;

    G4ThreeVector res (tmp->GetResolution().x() + 2,
                       tmp->GetResolution().y() + 2,
                       tmp->GetResolution().z() + 2);
    pImage->SetResolutionAndVoxelSize(res,tmp->GetVoxelSize());
    pImage->SetOrigin(tmp->GetOrigin());
    pImage->SetTransformMatrix(tmp->GetTransformMatrix());
    pImage->Allocate();
    //pImage->Fill(-1);
    pImage->SetOutsideValue( tmp->GetMinValue() - 1 );
    pImage->Fill(pImage->GetOutsideValue() );

    int i,j,k;
    for (k=0;k<res.z()-2;k++)
      for (j=0;j<res.y()-2;j++)
        for (i=0;i<res.x()-2;i++)
          pImage->SetValue(i+1,j+1,k+1,tmp->GetValue(i,j,k));

    delete tmp;
  }
  else {
    pImage = tmp;
    pImage->SetOutsideValue( pImage->GetMinValue() - 1 );
  }

  // Set volume origin from the image origin
  SetOrigin(pImage->GetOrigin());

  // Account for image rotation matrix: compose image and current rotations
  static bool pImageTransformHasBeenApplied = false;
  if (!pImageTransformHasBeenApplied) {
    pImageTransformHasBeenApplied = true;
    mTransformMatrix = pImage->GetTransformMatrix();
    mTransformMatrix.rotate(this->GetVolumePlacement()->GetRotationAngle(),
                            this->GetVolumePlacement()->GetRotationAxis());

    // Decompose to axis angle and set new rotation
    double delta;
    G4ThreeVector axis;
    mTransformMatrix.getAngleAxis(delta, axis);
    this->GetVolumePlacement()->SetRotationAngle(delta);
    this->GetVolumePlacement()->SetRotationAxis(axis);
  }

  GateMessage("Volume",4,"voxel size" << pImage->GetVoxelSize() << Gateendl);
  GateMessage("Volume",4,"origin" << GetOrigin() << Gateendl);
  GateMessageDec("Volume",4,"End GateVImageVolume::LoadImage("<<mImageFilename<<")\n");
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Loads the LabelToMaterial file
void GateVImageVolume::LoadImageMaterialsTable()
{
  GateMessageInc("Volume",4,"Begin GateVImageVolume::LoadImageMaterialsTable("<<mImageFilename<<")\n");

  if (mLoadImageMaterialsFromHounsfieldTable) {
    LoadImageMaterialsFromHounsfieldTable();
  }
  else {
    if (mLoadImageMaterialsFromLabelTable) LoadImageMaterialsFromLabelTable();
    else LoadImageMaterialsFromRangeTable();
  }
  GateMessage("Volume", 1, "Number of different materials in the image "
              << mImageFilename << " : " << mLabelToMaterialName.size() << Gateendl);

  GateMessageDec("Volume",4,"End GateVImageVolume::LoadImageMaterialsTable("<<mImageFilename<<")\n");
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::SetLabeledImageFilename(G4String filename) {
  mHLabelImageFilename = filename;
  mWriteHLabelImage = true;
  if (mImageMaterialsFromHounsfieldTableDone) DumpHLabelImage();
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::SetDensityImageFilename(G4String filename) {
  mDensityImageFilename = filename;
  mWriteDensityImage = true;
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::LoadImageMaterialsFromHounsfieldTable() {
  GateMessageInc("Volume",5,"Begin GateVImageVolume::LoadImageMaterialsFromHounsfieldTable("
                 <<mHounsfieldToImageMaterialTableFilename<<")\n");

  // Read H/matName file, fill GateHounsfieldMaterialTable>

  std::ifstream is;
  OpenFileInput(mHounsfieldToImageMaterialTableFilename, is);
  mHounsfieldMaterialTable.Reset();

  //FIXME: remove these two lines, we should load the HU-file as-is. It's up to the user to make sure it works.
  // GetOutsideValue returns the lowest value found in the image - 1. NOT the lowest value in mHounsfieldToImageMaterialTableFilename
  // DS/LG/DB Feb 2019: we keep these two lines for now because we suspect that they may be essential for
  // the extra layer of voxels in the "regionalized volumes".
  G4String parentMat = GetParentVolume()->GetMaterialName();
  mHounsfieldMaterialTable.AddMaterial(pImage->GetOutsideValue(),pImage->GetOutsideValue(),parentMat);

  double low  =  1e6; //must start oppositely for the comparisons to work.
  double high = -1e6;
  while (is) {
    skipComment(is);
    double h1,h2;
    G4String n;

    is >> h1;
    is >> h2;
    is >> n;

    low  = (h1<low)?h1:low; //set low to h1 if h1 is lower
    high = (h2>high)?h2:high; //set high to h2 if h2 is higher

    if (is) {
      if (h2 > pImage->GetOutsideValue()) {
        if (h1 < pImage->GetOutsideValue()+1)
          h1 = pImage->GetOutsideValue()+1;

        mHounsfieldMaterialTable.AddMaterial(h1,h2,n);
      }
    }
  }

  low  =  1e6;
  high = -1e6;
  GateHounsfieldMaterialTable::GateMaterialsVector vec = mHounsfieldMaterialTable.GetMaterials();
  for (size_t i=0 ; i < vec.size() ; i++)
    {
      GateMessage("Volume",5,"h1: " << vec[i].mH1 << ", h2: " << vec[i].mH2 << ", n: " << vec[i].mName  << Gateendl);

      low  = (vec[i].mH1<low )?vec[i].mH1:low; //set low to h1 if h1 is lower
      high = (vec[i].mH2>high)?vec[i].mH2:high; //set high to h2 if h2 is higher
    }

  // Bounds check
  GateMessage("Volume",5,"ImageMinValue: " << pImage->GetMinValue() << ", ImageMaxValue: " << pImage->GetMaxValue() << Gateendl);
  GateMessage("Volume",5,"HUMinValue   : " << low << ", HUMaxValue: " << high << Gateendl);

  if (pImage->GetMinValue() < low || pImage->GetMaxValue() > high) {
    GateWarning( "The image contains HU indices out of range of the HU range found in " <<
                 mHounsfieldToImageMaterialTableFilename << Gateendl <<
                 "HU    min, max: " << low << ", " << high << Gateendl <<
                 "Image min, max: " << pImage->GetMinValue() << ", " << pImage->GetMaxValue() << Gateendl );
    // GateError( "Abort." << Gateendl);
  }
  if (mHounsfieldMaterialTable.GetNumberOfMaterials() == 0 ) {
    GateError("No Hounsfield material defined in the file "
              << mHounsfieldToImageMaterialTableFilename << ". Abort.\n");
  }

  // Loop, create map H->label + verify
  mHounsfieldMaterialTable.MapLabelToMaterial(mLabelToMaterialName);

  // Loop change image label
  ImageType::iterator iter;
  iter = pImage->begin();
  while (iter != pImage->end()) {
    double label = mHounsfieldMaterialTable.GetLabelFromH(*iter);
    if (label<0) {
      GateMessage("Volume",1," I find H=" << *iter
                << " in the image, while Hounsfield range start at "
                << mHounsfieldMaterialTable[0].mH1 << Gateendl);
      label = 0;
      ++mUnderflow;
    }
    if (label>=mHounsfieldMaterialTable.GetNumberOfMaterials()) {
      GateMessage("Volume",1," I find H=" << *iter
                << " in the image, while Hounsfield range stop at "
                << mHounsfieldMaterialTable[mHounsfieldMaterialTable.GetNumberOfMaterials()-1].mH2
                << Gateendl);
      label = mHounsfieldMaterialTable.GetNumberOfMaterials() - 1;
      ++mOverflow;
    }
    //GateMessage("Core", 0, " pix = " << (*iter) << " lab = " << label << Gateendl);
    (*iter) = label;
    ++iter;
  }

  assert( pImage->GetNumberOfValues() > 0 );
  // double out_of_range_fraction = double(mUnderflow+mOverflow)/pImage->GetNumberOfValues(); // not yet
  double out_of_range_fraction = double(mOverflow)/pImage->GetNumberOfValues();
  if ( out_of_range_fraction > mMaxOutOfRangeFraction ){
    int n_bad_max(std::floor(mMaxOutOfRangeFraction * pImage->GetNumberOfValues()));
    GateMessage( "Volume", 0, "ERROR: too many HU values are out of the range of the materials table: " << Gateendl
                 << "******** " << mUnderflow << " underflows (HU<" << low << ") ******** " << Gateendl
                 << "******** " << mOverflow << " overflows (HU>" << high << ") ********" << Gateendl);
    if ( (mUnderflow > 0) && ( std::abs( mHalfSize.x() - pImage->GetHalfSize().x()) > 0.1*pImage->GetVoxelSize().x() ) ){
        const G4ThreeVector & size = pImage->GetResolution();
        int n_margin = pImage->GetNumberOfValues() - (((int)lrint(size.x())-2)*((int)lrint(size.y())-2)*((int)lrint(size.z())-2));
        GateMessage( "Volume", 0, "(" << n_margin << " of the " << mUnderflow << " underflows are in the 1 voxel extra margin.)" << Gateendl);
    }
    GateMessage( "Volume", 0,
                 "(The maximum 'bad' fraction is " << mMaxOutOfRangeFraction
              << " and the image has " << pImage->GetNumberOfValues() << " voxels, " << Gateendl
              << "so maximum " << n_bad_max << " out-of-range HU values are allowed." << Gateendl
              << "Possible solutions: fix your input image, "
              << "or expand the HU range of the HU-to-materials table. "  << Gateendl
              << "Or, if you really know what you are doing, "
              << "you can set the 'setMaxOutOfRangeFraction' option to a nonzero value larger than " << out_of_range_fraction << " .)" << Gateendl );
    GateError( "ABORT" );
  }
  // Debug
  // for(uint i=0; i<mHounsfieldMaterialTable.GetH1Vector().size(); i++) {
  //     double h = mHounsfieldMaterialTable.GetH1Vector()[i];
  //     GateMessage("Volume", 4, "H=" << h << " label = " << mHounsfieldMaterialTable.GetLabelFromH(h) << Gateendl);
  //     GateMessage("Volume", 4, " => H mean" << mHounsfieldMaterialTable.GetHMeanFromLabel(mHounsfieldMaterialTable.GetLabelFromH(h)) << Gateendl);
  //   }

  // Dump label image if needed
  mImageMaterialsFromHounsfieldTableDone = true;

  DumpHLabelImage();
  DumpDensityImage();
  DumpMassImage();
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::DumpHLabelImage() {
  // Dump image if needed
  if (mWriteHLabelImage) {
    ImageType output;
    output.SetResolutionAndVoxelSize(pImage->GetResolution(), pImage->GetVoxelSize());
    output.SetOrigin(pImage->GetOrigin());
    output.Allocate();

    //  GateHounsfieldMaterialTable::LabelToMaterialNameType lab2mat;
    //     mHounsfieldMaterialTable.MapLabelToMaterial(lab2mat);

    ImageType::const_iterator pi;
    ImageType::iterator po;
    pi = pImage->begin();
    po = output.begin();
    while (pi != pImage->end()) {
      if (1) { // HU mean or d mean or label
        // G4Material * mat =
        // 	  theMaterialDatabase.GetMaterial(lab2mat[*pi]);
        // 	GateDebugMessage("Volume", 2, "lab " << *pi << " = " << mat->GetName() << Gateendl);
        // 	po = mat->GetDensity;

        double HU = mHounsfieldMaterialTable.GetHMeanFromLabel((int)lrint(*pi));
        *po = HU;
        ++po;
        ++pi;
      }
    }

    // Write image
    output.Write(mHLabelImageFilename);
  }
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::DumpDensityImage() {
  // Dump image if needed
  if (mWriteDensityImage) {
    ImageType output;
    output.SetResolutionAndVoxelSize(pImage->GetResolution(), pImage->GetVoxelSize());
    output.SetOrigin(pImage->GetOrigin());
    output.Allocate();

    //  GateHounsfieldMaterialTable::LabelToMaterialNameType lab2mat;
    //     mHounsfieldMaterialTable.MapLabelToMaterial(lab2mat);

    ImageType::const_iterator pi;
    ImageType::iterator po;
    pi = pImage->begin();
    po = output.begin();
    while (pi != pImage->end()) {
      if (1) { // HU mean or d mean or label
        // G4Material * mat =
        // 	  theMaterialDatabase.GetMaterial(lab2mat[*pi]);
        // 	GateDebugMessage("Volume", 2, "lab " << *pi << " = " << mat->GetName() << Gateendl);
        // 	po = mat->GetDensity;
        double density = mLoadImageMaterialsFromHounsfieldTable ?
          mHounsfieldMaterialTable[(int)lrint(*pi)].md1 :
          mRangeMaterialTable[(int)lrint(*pi)].md1;
        *po = density / (g / cm3);
        ++po;
        ++pi;
      }
    }

    // Write image
    output.Write(mDensityImageFilename);
  }
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::DumpMassImage() {
  // Dump image if needed
  if (mMassImageFilename != "none") {
    ImageType output;
    output.SetResolutionAndVoxelSize(pImage->GetResolution(), pImage->GetVoxelSize());
    output.SetOrigin(pImage->GetOrigin());
    output.Allocate();

    ImageType::const_iterator pi = pImage->begin();
    ImageType::iterator       po = output.begin();

    while (pi != pImage->end()) {
      double density = mLoadImageMaterialsFromHounsfieldTable ?
                       mHounsfieldMaterialTable[(int)lrint(*pi)].md1 :
                       mRangeMaterialTable     [(int)lrint(*pi)].md1;

      double mass    = density * pImage->GetVoxelVolume();

      *po = mass / g; // dump mass in grams

      ++po;
      ++pi;
    }

    // Write image
    output.Write(mMassImageFilename);
  }
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::LoadImageMaterialsFromLabelTable() {
  // Never call
  //GateError("GateVImageVolume::LoadImageMaterialsFromLabelTable : disabled! \n");

  // ------------------------------------
  GateMessageInc("Volume",5,"Begin GateVImageVolume::LoadImageMaterialsFromLabelTable("
                 <<mLabelToImageMaterialTableFilename<<")\n");

  // open file
  std::ifstream is;
  OpenFileInput(mLabelToImageMaterialTableFilename, is);
  // read labels
  while (is) {
    skipComment(is);
    int label;
    if (is) {
      is >> label;
      G4cout << "label=" << label << Gateendl;
      // Verify that this label is not already mapped
      LabelToMaterialNameType::iterator lit = mLabelToMaterialName.find(label) ;
      G4String materialName;
      if (is) {
        is >> materialName;
        if (lit != mLabelToMaterialName.end()) {
          GateMessage("Volume",4,"*** WARNING *** Label already in table : Old value replaced \n");
          (*lit).second = materialName;
          continue;
        }
        else {
          mLabelToMaterialName[label] = materialName;
        }
      }
    }
  } // end while


  GateMessage("Volume",5,"GateVImageVolume -- Label \tMaterial\n");
  LabelToMaterialNameType::iterator lit;
  for (lit=mLabelToMaterialName.begin(); lit!=mLabelToMaterialName.end(); ++lit) {
    GateMessage("Volume",6,""<<(*lit).first << " \t" << (*lit).second << Gateendl);

  }

  GateMessageDec("Volume",5,"End GateVImageVolume::LoadLabelToMaterialTable("
                 <<mLabelToImageMaterialTableFilename<<")\n");
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::LoadImageMaterialsFromRangeTable() {
  m_voxelMaterialTranslation.clear();

  std::ifstream inFile;

  inFile.open(mRangeToImageMaterialTableFilename.c_str(),std::ios::in);
  mRangeMaterialTable.Reset();
  G4String parentMat = GetParentVolume()->GetMaterialName();
  mRangeMaterialTable.AddMaterial(pImage->GetOutsideValue(),pImage->GetOutsideValue(),parentMat);

  if (inFile.is_open()){
    G4String material;
    G4double r1;
    G4double r2;
    G4int firstline;

    G4double red, green, blue, alpha;
    G4bool visible;
    char buffer [200];

    inFile.getline(buffer,200);
    std::istringstream is(buffer);

    is >> firstline;

    if (is.eof()){

      for (G4int iCol=0; iCol<firstline; iCol++) {
        inFile.getline(buffer,200);
        is.clear();
        is.str(buffer);

        is >> r1 >> r2;
        is >> material;

        if (is.eof()){
          visible=true;
          red=0.5;
          green=blue=0.0;
          alpha=1;
        }else{
          is >> std::boolalpha >> visible >> red >> green >> blue >> alpha;
        }

        G4cout << " min max " << r1 << " " << r2 << "  material: " << material
               << std::boolalpha << ", visible " << visible << ", rgba(" << red<<',' << green << ',' << blue << ')' << Gateendl;

        if (r2> pImage->GetOutsideValue()){
          if (r1<pImage->GetOutsideValue()+1) r1=pImage->GetOutsideValue()+1;
          mRangeMaterialTable.AddMaterial(r1,r2,material);
        }
        else
          {
            GateMessage("Materials",0,"Failed to add material "<< material << " to Database" << Gateendl);
          }

        mRangeMaterialTable.MapLabelToMaterial(mLabelToMaterialName);


        m_voxelAttributesTranslation[theMaterialDatabase.GetMaterial(material) ] =
          new G4VisAttributes(visible, G4Colour(red, green, blue, alpha));
      }

    }else{

      inFile.close();
      std::ifstream is;
      OpenFileInput(mRangeToImageMaterialTableFilename, is);
      mRangeMaterialTable.Reset();

      while (is){

        is >> r1 >> r2;
        is >> material;


        if (r2> pImage->GetOutsideValue()){
          if (r1<pImage->GetOutsideValue()+1) r1=pImage->GetOutsideValue()+1;
          mRangeMaterialTable.AddMaterial(r1,r2,material);
        }
      }
      mRangeMaterialTable.MapLabelToMaterial(mLabelToMaterialName);
    }

  }
  else {G4cout << "Error opening file.\n";}

  ImageType::iterator iter;
  iter = pImage->begin();
  while (iter != pImage->end()) {
    double label = mRangeMaterialTable.GetLabelFromR(*iter);
    if (label<0) {
      GateError(" I find R=" << *iter
                << " in the image, while range start at "
                << mRangeMaterialTable[0].mR1 << Gateendl);
    }
    if (label>=mRangeMaterialTable.GetNumberOfMaterials()) {
      GateError(" I find R=" << *iter
                << " in the image, while range stop at "
                << mRangeMaterialTable[mRangeMaterialTable.GetNumberOfMaterials()-1].mR1
                << Gateendl);
    }
    //GateMessage("Core", 0, " pix = " << (*iter) << " lab = " << label << Gateendl);
    (*iter) = label;
    ++iter;
  }
  mImageMaterialsFromRangeTableDone = true;

  DumpDensityImage();
  DumpMassImage();
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Builds a vector of the labels in the image
void GateVImageVolume::BuildLabelsVector( std::vector<LabelType>& LabelsVector)
{
  //G4cout << "ok\n";
  GateMessage("Volume",5,"Begin GateVImageVolume::BuildLabelsVector()\n");
  std::set<LabelType> ens;
  ImageType::iterator i;
  for (i=pImage->begin(); i!=pImage->end(); ++i) {
    if ( ((*i)!=-1) &&
         (ens.find(int(*i)) == ens.end())
         ) {
      ens.insert(int(*i));
      LabelsVector.push_back(int(*i));
      GateMessage("Volume",5,"New label = "<<int(*i)<< Gateendl);
    }
  }
  GateMessage("Volume",5,"End GateVImageVolume::BuildLabelsVector()\n");
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Builds a label to material map
void GateVImageVolume::BuildLabelToG4MaterialVector( std::vector<G4Material*>& M )
{
  //G4cout << "ok2\n";
  GateMessage("Volume",4,"Begin GateVImageVolume::BuildLabelToG4MaterialVector\n");
  LabelToMaterialNameType::iterator lit;
  int l = 0;
  M.resize(0);
  for (lit=mLabelToMaterialName.begin(); lit!=mLabelToMaterialName.end(); ++lit) {
    if (l != (*lit).first  ) {
      GateError("Labels should be from 0 to n continuously ..."
                << "Current labels n " << (*lit).first  << " is number " << l );

    }

    GateMessage("Volume",4,((*lit).first) << " -> " << ((*lit).second) << Gateendl);

    M.push_back(theMaterialDatabase.GetMaterial((*lit).second));
    l++;
  }

  GateMessage("Volume",4,"End GateVImageVolume::BuildLabelToG4MaterialVector\n");
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Remaps the labels form 0 to NbLabels-1. If marginAdded is true
/// then assigns the new label 0 to the label -1 which was created for
/// margin voxels (see LoadImage).
void GateVImageVolume::RemapLabelsContiguously( std::vector<LabelType>& labels, bool /*marginAdded*/ )
{
  GateMessageInc("Volume",5,"Begin GateVImageVolume::RemapLabelsContiguously\n");
  std::map<LabelType,LabelType> lmap;

  LabelType cur = 0;

  /*if (marginAdded) {
    lmap[-1] = 0;
    cur++;
    }*/

  std::vector<LabelType>::iterator i;
  for (i=labels.begin(); i!=labels.end(); ++i, ++cur) {
    //GateMessage("Core", 0, "cur = " << cur << " i= " << *i << " lmap before = " << lmap[*i] << Gateendl);
    lmap[*i] = cur;
    //GateMessage("Core", 0, "cur = " << cur << " i= " << *i << " lmap after = " << lmap[*i] << Gateendl);
    *i = cur;
  }

  // updates the image
  ImageType::iterator j;
  for (j=pImage->begin(); j!=pImage->end(); ++j) {
    *j = lmap[(LabelType)*j];
  }

  // updates the material map
  LabelToMaterialNameType mmap;
  int k1 = 0; //entry in the label to material table
  LabelToMaterialNameType::iterator k;
  for (k=mLabelToMaterialName.begin(); k!=mLabelToMaterialName.end(); ++k) {
    // GateMessage("Core", 0, " k = "<<k1<<" first = " << lmap[(*k).first]
    //             << " second = " << (*k).second << Gateendl);
    if (lmap[(*k).first]!=0 || ( k1==0 && lmap[(*k).first]==0) ) mmap[ lmap[(*k).first] ] = (*k).second;
    k1++;
  }
  mLabelToMaterialName = mmap;

  // Update GateHounsfieldMaterialTable if needed
  // Is really no need to update name and material also??
  std::vector<GateHounsfieldMaterialTable::mMaterials> tmp(mHounsfieldMaterialTable.GetNumberOfMaterials());
  for(int i=0; i<mHounsfieldMaterialTable.GetNumberOfMaterials(); i++)
    {
      if (lmap[i]!=0 || ( i==0 && lmap[i]==0) )
        {
          tmp[lmap[i]].mH1 = mHounsfieldMaterialTable[i].mH1;
          tmp[lmap[i]].mH2 = mHounsfieldMaterialTable[i].mH2;
          tmp[lmap[i]].md1 = mHounsfieldMaterialTable[i].md1;
        }
    }
  for(int i=0; i<mHounsfieldMaterialTable.GetNumberOfMaterials(); i++)
    {
      mHounsfieldMaterialTable[i].mH1 = tmp[i].mH1;
      mHounsfieldMaterialTable[i].mH2 = tmp[i].mH2;
      mHounsfieldMaterialTable[i].md1 = tmp[i].md1;
    }
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::PrintInfo()
{
  GateMessage("Actor", 1, "GateVImageVolume Actor \n");
  pImage->PrintInfo();
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
int GateVImageVolume::GetNextVoxel(const G4ThreeVector& position,
                                   const G4ThreeVector& direction)
{
  return pImage->GetIndexFromPostPositionAndDirection(position, direction);
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::DestroyOwnSolidAndLogicalVolume()
{
  if (pBoxLog) delete pBoxLog;
  pBoxLog = 0;
  if (pBoxSolid) delete pBoxSolid;
  pBoxSolid = 0;
  // If I put these warnings in the destructor, then they will NEVER be displayed.
  if (mUnderflow>0) GateMessage("Volume",0,"There were " << mUnderflow << " voxels with HU values less than the minimum HU value in the HU-to-materials table." << Gateendl);
  if ( (mUnderflow > 0) && ( std::abs( mHalfSize.x() - pImage->GetHalfSize().x()) > 0.1*pImage->GetVoxelSize().x() ) ){
    const G4ThreeVector & size = pImage->GetResolution();
    int n_margin = pImage->GetNumberOfValues() - (((int)lrint(size.x())-2)*((int)lrint(size.y())-2)*((int)lrint(size.z())-2));
    GateMessage( "Volume", 0, "(" << n_margin << " of the " << mUnderflow << " underflows are in the 1 voxel extra margin.)" << Gateendl);
  }
  if (mOverflow>0) GateMessage("Volume",0,"There were " << mOverflow << " voxels with HU values greater than the maximum HU value in the HU-to-materials table." << Gateendl);
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::SetBuildDistanceTransfoFilename(G4String filename)
{
  mBuildDistanceTransfo = true;
  mDistanceTransfoOutput = filename;
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
void GateVImageVolume::BuildDistanceTransfo()
{
  GateMessage("Geometry", 1, "Building distante map image (dmap) for the image '"
              << mImageFilename << "' (it could be long for large image)."
              << Gateendl);

  // TEMPORARY : need isotrop !
  if ((pImage->GetVoxelSize().x() !=
       pImage->GetVoxelSize().y()) &&
      (pImage->GetVoxelSize().x() !=
       pImage->GetVoxelSize().z())) {
    GateError("Error : image spacing should be isotropic for building dmap");
    exit(0);
  }

  // Convert (copy) image into Vol structure
  const G4ThreeVector & size = pImage->GetResolution();
  GateMessage("Geometry", 1, "Image size is " << size << ".\n");
  Vol v((int)lrint(size.x()),
        (int)lrint(size.y()),
        (int)lrint(size.z()), 0);
  v.setVolumeCenter( v.sizeX()/2, v.sizeY()/2, v.sizeZ()/2 );
  voxel * p = v.getDataPointer();
  GateImage::const_iterator iter = pImage->begin();
  while (iter != pImage->end()) {
    /*
      if ((*iter < 0) || (*iter > 255)) //FIXME
      {
      GateError("Error image value not uchar =" << *iter << Gateendl);
      }
    */
    *p = static_cast<voxel>(lrint(*iter));
    ++p;
    ++iter;
  }

  //   GateDebugMessage("Geometry", 0, "im val = " << pImage->GetValue(22, 54, 34) << Gateendl);
  //   GateDebugMessage("Geometry", 0, "vo val = " << (float)v.get(22, 54, 34) << Gateendl);

  if (!v.isOK()) {
    fprintf( stderr, "Error construction vol ?");
    exit(0);
  }

  // Creating the longvol tmpOutput structure
  Longvol tmpOutput(v.sizeX(),v.sizeY(),v.sizeZ(),0);
  if (!tmpOutput.isOK()) {
    fprintf( stderr, "Couldn't init the longvol structure !\n" );
    exit(0);
  }

  // Each volume center is (0,0,0)x(sizeX,sizeY,sizeZ)
  tmpOutput.setVolumeCenter( tmpOutput.sizeX()/2, tmpOutput.sizeY()/2, tmpOutput.sizeZ()/2 );

  GateMessage("Geometry", 4, "Input Vol size: "<<
              tmpOutput.sizeX()<<"x"<<tmpOutput.sizeY()<<"x"<< tmpOutput.sizeZ()<< Gateendl);

  // Go ?
  GateMessage("Geometry", 4, "Start distance map computation ...\n");
  bool b = computeSEDT(v, tmpOutput, true, false, 1);
  GateMessage("Geometry", 4, "End ! b = " << b << Gateendl);

  // Convert (copy) image from Vol structure into GateImage
  GateDebugMessage("Geometry", 4, "Convert and output\n");
  GateImage output;
  output.SetResolutionAndHalfSize(pImage->GetResolution(), pImage->GetHalfSize());
  output.SetOrigin(pImage->GetOrigin());
  output.Allocate();
  double spacingFactor = pImage->GetVoxelSize().x();
  GateImage::iterator it = output.begin();
  lvoxel * pp = tmpOutput.getDataPointer();
  while (it < output.end()) {
    *it = (float)sqrt(*pp * spacingFactor);
    ++pp;
    ++it;
  }

  // Dump final result ...
  output.Write(mDistanceTransfoOutput);
  GateMessage("Geometry", 1, "Distance map write to disk in the file '" << mDistanceTransfoOutput << "'.\n");
  GateMessage("Geometry", 1, "You can now use it in the simulation. Use the macro 'distanceMap'. The macro 'buildAndDumpDistanceTransfo' is no more needed.\n");
}
//--------------------------------------------------------------------
