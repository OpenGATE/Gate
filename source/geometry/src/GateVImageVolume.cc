/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
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

typedef unsigned int uint;

//--------------------------------------------------------------------
/// Constructor with :
/// the path to the volume to create (for commands)
/// the name of the volume to create
/// Creates the messenger associated to the volume
GateVImageVolume::GateVImageVolume( const G4String& name,G4bool acceptsChildren,G4int depth) :
  GateVVolume(name,acceptsChildren,depth)
{
  GateMessageInc("Volume",5,"Begin GateVImageVolume("<<name<<")"<<G4endl);
  mImageFilename="";
  pImage=0;
  mHalfSize = G4ThreeVector(0,0,0);
  mIsoCenterIsSetByUser = false;
  pOwnMaterial = GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial("Air");
  mBuildDistanceTransfo = false;
  mLoadImageMaterialsFromHounsfieldTable = false;
  mLoadImageMaterialsFromLabelTable = false;
  mLabelToImageMaterialTableFilename = "none";
  mHounsfieldToImageMaterialTableFilename = "none";
  mRangeToImageMaterialTableFilename = "none";
  mWriteHLabelImage = false;
  mHLabelImageFilename = "none";
  mIsBoundingBoxOnlyModeEnabled = false;
  mImageMaterialsFromHounsfieldTableDone = false;
  GateMessageDec("Volume",5,"End GateVImageVolume("<<name<<")"<<G4endl);

  // do not display all voxels, only bounding box
  pOwnVisAtt->SetDaughtersInvisible(true);
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
/// Destructor
GateVImageVolume::~GateVImageVolume()
{
  GateMessageInc("Volume",5,"Begin ~GateVImageVolume()"<<G4endl);
  if(pImage) delete pImage;
  //if(pBoxPhys) delete pBoxPhys;
  if(pBoxLog) delete pBoxLog;
  if(pBoxSolid) delete pBoxSolid;
  GateMessageDec("Volume",5,"End ~GateVImageVolume()"<<G4endl);
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::EnableBoundingBoxOnly(bool b) {
  mIsBoundingBoxOnlyModeEnabled = b;
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::SetIsoCenter(const G4ThreeVector & i)
{
  mIsoCenter = i;
  mIsoCenterIsSetByUser = true;
  GateMessage("Volume",5,"IsoCenter = " << mIsoCenter << G4endl);
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

    GateMessage("Volume",3,"Current T = " << tcurrent << G4endl);
    GateMessage("Volume",3,"Isocenter = " << GetIsoCenter() << G4endl);
    GateMessage("Volume",3,"Origin = " << GetOrigin() << G4endl);
    GateMessage("Volume",3,"Half = " << GetHalfSize() << G4endl);
    GateMessage("Volume",3,"TransformMatrix = " << mTransformMatrix << G4endl);

    // Compute translation
    G4ThreeVector q = mIsoCenter - GetOrigin();
    q -= mTransformMatrix*GetHalfSize();
    q = tcurrent - q;
    GetVolumePlacement()->SetTranslation(q);
  }
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
G4double GateVImageVolume::GetHalfDimension(size_t axis)
{
  if(axis==0) return mHalfSize.x();
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
    GateError("Please set SetHUToMaterialFile or SetLabelToMaterialFile, not both. Abort." << G4endl);
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
    GateError("Please set SetHUToMaterialFile or SetLabelToMaterialFile, not both. Abort." << G4endl);
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
    GateError("Please set SetHUToMaterialFile or SetLabelToMaterialFile, not both. Abort." << G4endl);
  }
  mRangeToImageMaterialTableFilename = name;
  if (mImageFilename.length()>0) ImageAndTableFilenamesOK();
}
//--------------------------------------------------------------------



//--------------------------------------------------------------------
/// Loads the image
void GateVImageVolume::LoadImage(bool add1VoxelMargin)
{
  GateMessageInc("Volume",4,"Begin GateVImageVolume::LoadImage("<<mImageFilename<<")" << G4endl);

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
    //G4cout << mImageFilename << G4endl;
  }
  //tmp->PrintInfo();

  /// The volume's half size is the half size of the initial image, even if a margin is added
  mHalfSize = tmp->GetHalfSize();

  if (pImage) delete pImage;

  if (add1VoxelMargin) {
    // The image is copied with a margin of 1 voxel in all directions
    pImage = new ImageType;

    G4ThreeVector res ( tmp->GetResolution().x() + 2,
			tmp->GetResolution().y() + 2,
			tmp->GetResolution().z() + 2);
    pImage->SetResolutionAndVoxelSize(res,tmp->GetVoxelSize());
    pImage->SetOrigin(tmp->GetOrigin());
    pImage->SetTransformMatrix(tmp->GetTransformMatrix());
    pImage->Allocate();
    //pImage->Fill(-1);
    pImage->SetOutsideValue(  tmp->GetMinValue() - 1 );
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
    pImage->SetOutsideValue(  pImage->GetMinValue() - 1 );
  }

  // Set volume origin from the image origin
  SetOrigin(pImage->GetOrigin());

  // Account for image rotation matrix: compose image and current rotations
  static bool pImageTransformHasBeenApplied = false;
  if(!pImageTransformHasBeenApplied) {
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

  GateMessage("Volume",4,"voxel size" << pImage->GetVoxelSize() << G4endl);
  GateMessage("Volume",4,"origin" << GetOrigin() << G4endl);
  GateMessageDec("Volume",4,"End GateVImageVolume::LoadImage("<<mImageFilename<<")" << G4endl);
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Loads the LabelToMaterial file
void GateVImageVolume::LoadImageMaterialsTable()
{
  if (mLoadImageMaterialsFromHounsfieldTable) {
    LoadImageMaterialsFromHounsfieldTable();
  }
  else {
    if (mLoadImageMaterialsFromLabelTable) LoadImageMaterialsFromLabelTable();
    else LoadImageMaterialsFromRangeTable();
  }
  GateMessage("Volume", 1, "Number of different materials in the image "
              << mImageFilename << " : " << mLabelToMaterialName.size() << G4endl);
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
void GateVImageVolume::LoadImageMaterialsFromHounsfieldTable()
{
  GateMessageInc("Volume",5,"Begin GateVImageVolume::LoadImageMaterialsFromHounsfieldTable("
		 <<mHounsfieldToImageMaterialTableFilename<<")" << G4endl);

  // Read H/matName file, fill GateHounsfieldMaterialTable>

  std::ifstream is;
  OpenFileInput(mHounsfieldToImageMaterialTableFilename, is);
  mHounsfieldMaterialTable.Reset();
  // mHounsfieldMaterialTable.AddMaterial(pImage->GetOutsideValue(), pImage->GetOutsideValue()+1,"worldDefaultAir");
  G4String parentMat = GetParentVolume()->GetMaterialName();
  mHounsfieldMaterialTable.AddMaterial(pImage->GetOutsideValue(),pImage->GetOutsideValue()+1,parentMat);

  while (is) {
    skipComment(is);
    double h1,h2;

    is >> h1;
    is >> h2;
    G4String n;
    is >> n;
    if (is) {
      if(h2> pImage->GetOutsideValue()+1){
        if(h1<pImage->GetOutsideValue()+1) h1=pImage->GetOutsideValue()+1;
        mHounsfieldMaterialTable.AddMaterial(h1,h2,n);
      }
    }
  }
//  if (mHounsfieldMaterialTable.GetNumberOfMaterials() == 0) {
  if (mHounsfieldMaterialTable.GetNumberOfMaterials() == 1 ) {//there is a default mat = worldDefaultAir
    GateError("No Hounsfield material defined in the file "
	      << mHounsfieldToImageMaterialTableFilename << ". Abort" << G4endl);
  }

  // Loop, create map H->label + verify
  mHounsfieldMaterialTable.MapLabelToMaterial(mLabelToMaterialName);

  // Loop change image label
  ImageType::iterator iter;
  iter = pImage->begin();
  while (iter != pImage->end()) {
    double label = mHounsfieldMaterialTable.GetLabelFromH(*iter);
    if (label<0) {
      GateError(" I find H=" << *iter
		<< " in the image, while Hounsfield range start at "
		<< mHounsfieldMaterialTable.GetH1Vector()[0] << G4endl);
    }
    if (label>=mHounsfieldMaterialTable.GetNumberOfMaterials()) {
      GateError(" I find H=" << *iter
		<< " in the image, while Hounsfield range stop at "
		<< mHounsfieldMaterialTable.GetH2Vector()[mHounsfieldMaterialTable.GetH2Vector().size()-1]
		<< G4endl);
    }
    //GateMessage("Core", 0, " pix = " << (*iter) << " lab = " << label << G4endl);
    (*iter) = label;
    ++iter;
  }

  // Debug
  // for(uint i=0; i<mHounsfieldMaterialTable.GetH1Vector().size(); i++) {
//     double h = mHounsfieldMaterialTable.GetH1Vector()[i];
//     GateMessage("Volume", 4, "H=" << h << " label = " << mHounsfieldMaterialTable.GetLabelFromH(h) << G4endl);
//     GateMessage("Volume", 4, " => H mean" << mHounsfieldMaterialTable.GetHMeanFromLabel(mHounsfieldMaterialTable.GetLabelFromH(h)) << G4endl);
//   }

  // Dump label image if needed
  mImageMaterialsFromHounsfieldTableDone = true;
  DumpHLabelImage();
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
	// 	  GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(lab2mat[*pi]);
	// 	GateDebugMessage("Volume", 2, "lab " << *pi << " = " << mat->GetName() << G4endl);
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
void GateVImageVolume::LoadImageMaterialsFromLabelTable()
{
  // Never call
  //GateError("GateVImageVolume::LoadImageMaterialsFromLabelTable : disabled! " << G4endl);

  // ------------------------------------
  GateMessageInc("Volume",5,"Begin GateVImageVolume::LoadImageMaterialsFromLabelTable("
		 <<mLabelToImageMaterialTableFilename<<")" << G4endl);

  // open file
  std::ifstream is;
  OpenFileInput(mLabelToImageMaterialTableFilename, is);
  // read labels
  while (is) {
    skipComment(is);
    int label;
    if (is) {
      is >> label;
      G4cout << "label=" << label << G4endl;
      // Verify that this label is not already mapped
      LabelToMaterialNameType::iterator lit = mLabelToMaterialName.find(label) ;
      G4String materialName;
      if (is) {
	is >> materialName;
	if (lit != mLabelToMaterialName.end()) {
	  GateMessage("Volume",4,"*** WARNING *** Label already in table : Old value replaced "<<G4endl);
	  (*lit).second = materialName;
	  continue;
	}
	else {
	  mLabelToMaterialName[label] = materialName;
	}
      }
    }
  } // end while


  GateMessage("Volume",5,"GateVImageVolume -- Label \tMaterial" << G4endl);
  LabelToMaterialNameType::iterator lit;
  for (lit=mLabelToMaterialName.begin(); lit!=mLabelToMaterialName.end(); ++lit) {
    GateMessage("Volume",6,""<<(*lit).first << " \t" << (*lit).second << G4endl);

  }

  GateMessageDec("Volume",5,"End GateVImageVolume::LoadLabelToMaterialTable("
		 <<mLabelToImageMaterialTableFilename<<")" << G4endl);
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::LoadImageMaterialsFromRangeTable()
{
  m_voxelMaterialTranslation.clear();

  std::ifstream inFile;

  inFile.open(mRangeToImageMaterialTableFilename.c_str(),std::ios::in);
  mRangeMaterialTable.Reset();
  G4String parentMat = GetParentVolume()->GetMaterialName();
  mRangeMaterialTable.AddMaterial(pImage->GetOutsideValue(),pImage->GetOutsideValue()+1,parentMat);

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
    << std::boolalpha << ", visible " << visible << ", rgba(" << red<<',' << green << ',' << blue << ')' << G4endl;

    if(r2> pImage->GetOutsideValue()+1){
      if(r1<pImage->GetOutsideValue()+1) r1=pImage->GetOutsideValue()+1;
        mRangeMaterialTable.AddMaterial(r1,r2,material);
    }

  mRangeMaterialTable.MapLabelToMaterial(mLabelToMaterialName);


  m_voxelAttributesTranslation[GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial(material) ] =
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

  if(r2> pImage->GetOutsideValue()+1){
    if(r1<pImage->GetOutsideValue()+1) r1=pImage->GetOutsideValue()+1;
      mRangeMaterialTable.AddMaterial(r1,r2,material);
  }
  }
  mRangeMaterialTable.MapLabelToMaterial(mLabelToMaterialName);
  }

  }
  else {G4cout << "Error opening file." << G4endl;}

  ImageType::iterator iter;
  iter = pImage->begin();
  while (iter != pImage->end()) {
    double label = mRangeMaterialTable.GetLabelFromR(*iter);
    if (label<0) {
      GateError(" I find R=" << *iter
		<< " in the image, while range start at "
		<< mRangeMaterialTable.GetR1Vector()[0] << G4endl);
    }
    if (label>=mRangeMaterialTable.GetNumberOfMaterials()) {
      GateError(" I find R=" << *iter
		<< " in the image, while range stop at "
		<< mRangeMaterialTable.GetR2Vector()[mRangeMaterialTable.GetR2Vector().size()-1]
		<< G4endl);
    }
    //GateMessage("Core", 0, " pix = " << (*iter) << " lab = " << label << G4endl);
    (*iter) = label;
    ++iter;
  }
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Builds a vector of the labels in the image
void GateVImageVolume::BuildLabelsVector( std::vector<LabelType>& LabelsVector)
{
  //G4cout << "ok" << G4endl;
  GateMessage("Volume",5,"Begin GateVImageVolume::BuildLabelsVector()" << G4endl);
  std::set<LabelType> ens;
  ImageType::iterator i;
  for (i=pImage->begin(); i!=pImage->end(); ++i) {
    if ( ((*i)!=-1) &&
	 (ens.find(int(*i)) == ens.end())
	 ) {
      ens.insert(int(*i));
      LabelsVector.push_back(int(*i));
      GateMessage("Volume",5,"New label = "<<int(*i)<<G4endl);
    }
  }
  GateMessage("Volume",5,"End GateVImageVolume::BuildLabelsVector()" << G4endl);
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Builds a label to material map
void GateVImageVolume::BuildLabelToG4MaterialVector( std::vector<G4Material*>& M )
{
  //G4cout << "ok2" << G4endl;
  GateMessage("Volume",4,"Begin GateVImageVolume::BuildLabelToG4MaterialVector" << G4endl);
  LabelToMaterialNameType::iterator lit;
  int l = 0;
  M.resize(0);
  for (lit=mLabelToMaterialName.begin(); lit!=mLabelToMaterialName.end(); ++lit) {
    if (l != (*lit).first  ) {
      GateError("Labels should be from 0 to n continuously ..."
                << "Current labels n " << (*lit).first  << " is number " << l );

    }

    GateMessage("Volume",4,((*lit).first) << " -> " << ((*lit).second) << G4endl);

    M.push_back(GateDetectorConstruction::GetGateDetectorConstruction()->mMaterialDatabase.GetMaterial((*lit).second));
    l++;
  }

  GateMessage("Volume",4,"End GateVImageVolume::BuildLabelToG4MaterialVector" << G4endl);
}
//--------------------------------------------------------------------

//--------------------------------------------------------------------
/// Remaps the labels form 0 to NbLabels-1. If marginAdded is true
/// then assigns the new label 0 to the label -1 which was created for
/// margin voxels (see LoadImage).
void GateVImageVolume::RemapLabelsContiguously( std::vector<LabelType>& labels, bool /*marginAdded*/ )
{
  GateMessageInc("Volume",5,"Begin GateVImageVolume::RemapLabelsContiguously" << G4endl);
  std::map<LabelType,LabelType> lmap;

  LabelType cur = 0;

  /*if (marginAdded) {
    lmap[-1] = 0;
    cur++;
  }*/

  std::vector<LabelType>::iterator i;
  for (i=labels.begin(); i!=labels.end(); ++i, ++cur) {
    //GateMessage("Core", 0, "cur = " << cur << " i= " << *i << " lmap before = " << lmap[*i] << G4endl);
    lmap[*i] = cur;
    //GateMessage("Core", 0, "cur = " << cur << " i= " << *i << " lmap after = " << lmap[*i] << G4endl);
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
    //             << " second = " << (*k).second << G4endl);
    if(lmap[(*k).first]!=0 || ( k1==0 && lmap[(*k).first]==0) ) mmap[ lmap[(*k).first] ] = (*k).second;
    k1++;
  }
  mLabelToMaterialName = mmap;

  // Update GateHounsfieldMaterialTable if needed
  if (mHounsfieldMaterialTable.GetNumberOfMaterials() != 0) {
    std::vector<double> & h1 = mHounsfieldMaterialTable.GetH1Vector();
    std::vector<double> tmp(h1.size());
   for(uint i=0; i<h1.size(); i++) {
      if(lmap[i]!=0 || ( i==0 && lmap[i]==0) )  tmp[lmap[i]] = h1[i];
    }
    for(uint i=0; i<h1.size(); i++) {
      //GateMessage("Core", 0, "i = " << i
      //             << " h1 = " << h1[i]
      //             << " new h1 = " << tmp[i] << G4endl);
       h1[i] = tmp[i];
    }
    std::vector<double> & h2 = mHounsfieldMaterialTable.GetH2Vector();
    for(uint i=0; i<h2.size(); i++) {
       if(lmap[i]!=0 || ( i==0 && lmap[i]==0) )  tmp[lmap[i]] = h2[i];
    }
    for(uint i=0; i<h2.size(); i++) {
      h2[i] = tmp[i];
    }

    std::vector<double> & d1 = mHounsfieldMaterialTable.GetDVector();
    for(uint i=0; i<d1.size(); i++) {
     if(lmap[i]!=0 || ( i==0 && lmap[i]==0) )   tmp[lmap[i]] = d1[i];
    }
    for(uint i=0; i<d1.size(); i++) {
     d1[i] = tmp[i];
    }

  }
}
//--------------------------------------------------------------------


//--------------------------------------------------------------------
void GateVImageVolume::PrintInfo()
{
  GateMessage("Actor", 1, "GateVImageVolume Actor " << G4endl);
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
	      << G4endl);

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
  GateMessage("Geometry", 1, "Image size is " << size << "." << G4endl);
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
	GateError("Error image value not uchar =" << *iter << G4endl);
      }
    */
    *p = static_cast<voxel>(lrint(*iter));
    ++p;
    ++iter;
  }

  //   GateDebugMessage("Geometry", 0, "im val = " << pImage->GetValue(22, 54, 34) << G4endl);
  //   GateDebugMessage("Geometry", 0, "vo val = " << (float)v.get(22, 54, 34) << G4endl);

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
	      tmpOutput.sizeX()<<"x"<<tmpOutput.sizeY()<<"x"<< tmpOutput.sizeZ()<<G4endl);

  // Go ?
  GateMessage("Geometry", 4, "Start distance map computation ..." << G4endl);
  bool b = computeSEDT(v, tmpOutput, true, false, 1);
  GateMessage("Geometry", 4, "End ! b = " << b << G4endl);

  // Convert (copy) image from Vol structure into GateImage
  GateDebugMessage("Geometry", 4, "Convert and output" << G4endl);
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
  GateMessage("Geometry", 1, "Distance map write to disk in the file '" << mDistanceTransfoOutput << "'." << G4endl);
  GateMessage("Geometry", 1, "You can now use it in the simulation. Use the macro 'distanceMap'. The macro 'buildAndDumpDistanceTransfo' is no more needed." << G4endl);
}
//--------------------------------------------------------------------
