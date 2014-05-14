/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*! /file
  /brief Implementation of GateImageRegionalizedVolume
*/

#include "GateImageRegionalizedVolume.hh"
#include "GateImageRegionalizedVolumeMessenger.hh"
#include "GateImageRegionalizedSubVolume.hh"
#include "GateImageRegionalizedVolumeSolid.hh"
#include "GateMessageManager.hh"
#include  "GateSystemListManager.hh"
// From G4.9.0 the geometrical tolerances are stored in
// a singleton of G4GeometryTolerance
#include "G4GeometryTolerance.hh"
#include "GateMultiSensitiveDetector.hh"
#include "GateUserActions.hh"
#include "GateImage.hh"
#include "GatePhantomSD.hh"
#include "GateDetectorConstruction.hh"

//-----------------------------------------------------------------------------
/// Constructor with :
/// the path to the volume to create (for commands)
/// the name of the volume to create
/// Creates the messenger associated to the volume
GateImageRegionalizedVolume::GateImageRegionalizedVolume(const G4String& name,
							 G4bool acceptsChildren,
							 G4int depth)
  : GateVImageVolume(name,acceptsChildren,depth)
{
  GateMessageInc("Volume",5,"GateImageRegionalizedVolume() - begin"<<G4endl);

  // messenger
  pMessenger = new GateImageRegionalizedVolumeMessenger(this);
  // Retrieves surface tolerance from G4GeometryTolerance instance
  kCarTolerance = G4GeometryTolerance::GetInstance()->GetSurfaceTolerance();
  mDistanceMapFilename = "none";
  pDistanceMap = 0;
  GateMessageDec("Volume",5,"GateImageRegionalizedVolume() - end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateImageRegionalizedVolume::~GateImageRegionalizedVolume()
{
  GateMessageInc("Volume",5,"~GateImageRegionalizedVolume() - begin"<<G4endl);

  for(std::map<LabelType,GateImageRegionalizedSubVolume*>::iterator i = mLabelToSubVolume.begin();
      i!=mLabelToSubVolume.end(); i++) {
    for(std::vector<GateImageRegionalizedSubVolume*>::iterator it = mSubVolume.begin();
        it!=mSubVolume.end(); it++)  {
      if(i->second==(*it)){
        delete (*it);
        (*it) = NULL;
        i->second  = NULL;
      }
    }
    if(i->second) delete i->second ;
    i->second  = NULL;
  }
  for(std::vector<GateImageRegionalizedSubVolume*>::iterator it = mSubVolume.begin();
      it!=mSubVolume.end(); it++) {
    if((*it)) {
      delete (*it);
      (*it) = NULL;
    }
  }

  if(pDistanceMap) delete pDistanceMap;
  if(pMessenger) delete pMessenger;
  GateMessageInc("Volume",5,"~GateImageRegionalizedVolume() - end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Construct
G4LogicalVolume* GateImageRegionalizedVolume::ConstructOwnSolidAndLogicalVolume(G4Material* mater,
										G4bool /*flagUpdateOnly*/)
{
  GateMessage("Volume",1,"Begin GateImageRegionalizedVolume::ConstructOwnSolidAndLogicalVolume() - begin" << G4endl);

  //---------------------------
  if (mIsBoundingBoxOnlyModeEnabled) {
    GateError("Sorry not possible to user BoundingBoxOnlyMode with RegionalizedVolume. Use NestedParameterised instead");
  }

  // If needed, compute the distance map
  if (mBuildDistanceTransfo) BuildDistanceTransfo();

  //  EnableSmartVoxelOptimisation(false);
  G4String boxname = GetObjectName() + "_solid";

  pBoxSolid = new GateImageRegionalizedVolumeSolid(boxname,this);
  pBoxLog = new G4LogicalVolume(pBoxSolid, mater, GetLogicalVolumeName());

  // ---------------------------
  LoadDistanceMap();

  // Set position if IsoCenter is Set
  UpdatePositionWithIsoCenter();

  // AddPhysVolToOptimizedNavigator(GetPhysicalVolume());
  return pBoxLog;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Creates the sub volumes
void  GateImageRegionalizedVolume::ImageAndTableFilenamesOK()
{
  CreateSubVolumes();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Creates the sub volumes
void  GateImageRegionalizedVolume::CreateSubVolumes()
{
  GateMessageInc("Volume",1,"Begin GateImageRegionalizedVolume::CreateSubVolumes() - begin" << G4endl);

  //-----------------------------
  // Load data
  LoadImage(true);
  //  LoadImage(false);
  LoadImageMaterialsTable();

  std::vector<LabelType> labels;
  BuildLabelsVector(labels);
  RemapLabelsContiguously(labels,true);

  // Creation of the sub-volumes
  std::vector<LabelType>::iterator i;
  for (i=labels.begin();i!=labels.end();++i) {
    GateMessage("Volume",4,"* Label "<< *i <<G4endl);
    // Gets material
    G4String matName = GetMaterialNameFromLabel(*i);
    GateMessage("Volume",4,"  - Material name <"<< matName<<">"<<G4endl);

    G4String name = GetObjectName()+matName;

    GateImageRegionalizedSubVolume * subvolume = new GateImageRegionalizedSubVolume(name,true,0);
    pChildList->AddChild(subvolume);

    GateSystemListManager::GetInstance()->CheckScannerAutoCreation(subvolume);

    mSubVolume.push_back(subvolume);
    mLabelToSubVolume[*i] = subvolume;

    //  subvolume->SetParentVolumeName(GetObjectName());
    subvolume->SetVolume(this);
    //    subvolume->SetImage(GetImage());
    subvolume->SetLabel(*i);
    //   subvolume->SetHalfSize(GetHalfSize());
    subvolume->SetMaterialName(matName);
    // subvolume->SetPosition(GetPosition());
  }
  // Resize the vectors (+1 because label 0 = outside)
  mInside.resize(labels.size()+1);
  mDistanceToIn.resize(labels.size()+1);

  mLastInsidePointIsValid = false;
  mInsideComputedByLastDTO = false;
  mLastDTIPointIsValid = false;

  GateMessageDec("Volume",1,"Begin GateImageRegionalizedVolume::CreateSubVolumes() - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Loads the distance map
void GateImageRegionalizedVolume::LoadDistanceMap()
{
  GateMessageInc("Volume",3,"GateImageRegionalizedVolume::LoadDistanceMap("<<mDistanceMapFilename<<") - begin" << G4endl);

  if (mDistanceMapFilename == "none") {
    GateError("ImageRegionalized Volume <" << GetObjectName()
	      << "> : No distance map provided, the navigation could not be optimized."
	      << G4endl
	      << "Please use /gate/" << GetObjectName() << "/geometry/buildAndDumpDistanceTransfo dmap.mhd"
	      << G4endl
	      << "to generate the dmap and then, /gate/patient/geometry/distanceMap dmap.mhd to use it."
	      << G4endl);
    pDistanceMap = new DistanceMapType;
    pDistanceMap->SetResolutionAndHalfSize(GetImage()->GetResolution(), GetImage()->GetHalfSize());
    pDistanceMap->Allocate();
    pDistanceMap->Fill(0.0);
    pDistanceMap->PrintInfo();
    return;
  }
  if (pDistanceMap) delete pDistanceMap;
  pDistanceMap = new DistanceMapType;
  pDistanceMap->Read(mDistanceMapFilename);

  // Check size
  if (!pDistanceMap->HasSameResolutionThan(GetImage())) {
    GateError("Error distance map image does not have the same size than the image." << G4endl);
  }

  GateMessageDec("Volume",3,"GateImageRegionalizedVolume::LoadDistanceMap("<<mDistanceMapFilename<<") - end" << G4endl);
}
//-----------------------------------------------------------------------------

//------------------------------------------------
// Methods used by SubVolumeSolids
//------------------------------------------------

//-----------------------------------------------------------------------------
EInside GateImageRegionalizedVolume::Inside(const G4ThreeVector& p, LabelType label)
{
  GateDebugMessage("Volume", 6, "-----------------------------------------------------" << G4endl);
  GateDebugMessage("Volume", 6,
		   "\tGateImageRegionalizedVolume[" << GetObjectName() << "]::Inside("
		   << p <<",lab=" << label << "["<<GetMaterialNameFromLabel(label) << "])"
		   <<G4endl);

  static G4ThreeVector sLastPoint(0,0,0);

  if (sLastPoint == p) {
    return mInside[label];
  }
  else {
    sLastPoint = p;
  }

  // Reset the vector
  for (std::vector<EInside>::iterator i = mInside.begin(); i!=mInside.end(); ++i)
    *i = kOutside;

  // Outside the BBox ? A ENLEVER ????
  if (GetSolid()->Inside(p) == kOutside ) {
    GateDebugMessage("Volume", 6, "\t** result = **OUTSIDE** the box" << G4endl);
    return kOutside;
  }

  // Else ...
  int i,j,k;
  GetImage()->GetCoordinatesFromPosition(p,i,j,k);
  GateDebugMessage("Volume", 6, "\tvox coords : "<<i<<" "<<j<<" "<<k<<G4endl);

  GateDebugMessage("Volume", 6, "\tix = " << GetImage()->GetXVoxelCornerFromXCoordinate(i) <<G4endl);
  GateDebugMessage("Volume", 6, "\tkCarTolerance = " << kCarTolerance <<G4endl);
  GateDebugMessage("Volume", 6, "\tp.x()-GetImage()->GetXVoxelCornerFromXCoordinate(i)=" << p.x()-GetImage()->GetXVoxelCornerFromXCoordinate(i)<<G4endl);

  bool xsurf[2],ysurf[2],zsurf[2];
  // x surface ?
  if (p.x()-GetImage()->GetXVoxelCornerFromXCoordinate(i) <= kCarTolerance*0.5) {
    xsurf[0] = true;
    xsurf[1] = false;
  }
  else {
    xsurf[0] = false;
    if (GetImage()->GetXVoxelCornerFromXCoordinate(i+1)-p.x() <= kCarTolerance*0.5)
      xsurf[1] = true;
    else xsurf[1] = false;
  }

  // y surface ?
  if (p.y()-GetImage()->GetYVoxelCornerFromYCoordinate(j) <= kCarTolerance*0.5) {
    ysurf[0] = true;
    ysurf[1] = false;
  }
  else {
    ysurf[0] = false;
    if (GetImage()->GetYVoxelCornerFromYCoordinate(j+1)-p.y() < kCarTolerance*0.5)
      ysurf[1] = true;
    else ysurf[1] = false;
  }

  // z surface ?
  if (p.z()-GetImage()->GetZVoxelCornerFromZCoordinate(k) <= kCarTolerance*0.5) {
    zsurf[0] = true;
    zsurf[1] = false;
  }
  else {
    zsurf[0] = false;
    if (GetImage()->GetZVoxelCornerFromZCoordinate(k+1)-p.z() <= kCarTolerance*0.5)
      zsurf[1] = true;
    else zsurf[1] = false;
  }

  GateDebugMessage("Volume", 6, "\tsurface ? xsurf " << xsurf[0] << " " << xsurf[1] << G4endl);
  GateDebugMessage("Volume", 6, "\tsurface ? ysurf " << ysurf[0] << " " << ysurf[1] << G4endl);
  GateDebugMessage("Volume", 6, "\tsurface ? zsurf " << zsurf[0] << " " << zsurf[1] << G4endl);


  // label of the center voxel
  LabelType cl = (int)GetImage()->GetValue(i,j,k);
  GateDebugMessage("Volume", 6, "\tvox label : "<<cl<<G4endl);
  // By default (if not on surface), the point is inside the region cl
  mInside[cl] = kInside;


  // Now we have to update the mInside vector if the point is on a
  // surface. In Gate case, we have to see if the neighboring voxels
  // have or not the same label. If they have the same, set it to
  // kInside. Warning : todo when the point is on a surface but alos
  // when the point is on a edge or on a vertex. In Gate latter cases,
  // several neighbors must be updated. It rarely appends, but it
  // appends !
  GateDebugMessage("Volume", 6, "\tvox coord : " << i << " " << j << " " << k << G4endl);
  LabelType nl;
#define TEST(a,b,c)							\
  GateDebugMessage("Volume", 6, "\t Test voisin " << a << " " << b << " " << c <<G4endl); \
  if ((nl=(LabelType)GetImage()->GetValue(a,b,c)) != cl) {		\
    GateDebugMessage("Volume", 6, "\t ==> nl=" << nl << " => kSurface" <<G4endl); \
    mInside[cl] = mInside[nl] = kSurface;				\
  }

  // Get neighbors coordinate if point is on a surface
  int ii,jj,kk;
  if (xsurf[0]) ii=i-1;
  else {
    if (xsurf[1]) ii=i+1;
    else ii=i;
  }
  if (ysurf[0]) jj=j-1;
  else {
    if (ysurf[1]) jj=j+1;
    else jj=j;
  }
  if (zsurf[0]) kk=k-1;
  else {
    if (zsurf[1]) kk=k+1;
    else kk=k;
  }

  // Face
  if (ii != i) { TEST(ii, j, k); }
  if (jj != j) { TEST(i, jj, k); }
  if (kk != k) { TEST(i, j, kk); }

  // Edge
  if ((ii != i) && (jj != j)) { TEST(ii, jj, k); }
  if ((ii != i) && (kk != k)) { TEST(ii, j, kk); }
  if ((jj != j) && (kk != k)) { TEST(i, jj, kk); }

  // Vertex
  if ((ii != i) && (jj != j) && (kk != k)) { TEST(ii, jj, kk); }

  /*
    if (xsurf[0]) {
    GateDebugMessage("Volume", 6, "\t xsurf, voisin=" << (LabelType)GetImage()->GetValue(i-1,j,k)<<G4endl);
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j,k)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    if (ysurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j-1,k)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    if (zsurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j-1,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j-1,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    else if (ysurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j+1,k)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    if (zsurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j+1,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j+1,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    else {
    if (zsurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i-1,j,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    }
    else if (xsurf[1]) {
    GateDebugMessage("Volume", 6, "\t xsurf, voisin=" << (LabelType)GetImage()->GetValue(i+1,j,k)<<G4endl);
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j,k)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    if (ysurf[0]) {
    GateDebugMessage("Volume", 6, "\t ysurf, voisin=" << (LabelType)GetImage()->GetValue(i+1,j-1,k)<<G4endl);
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j-1,k)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    if (zsurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j-1,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j-1,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    else if (ysurf[1]) {
    GateDebugMessage("Volume", 6, "\t ysurf, voisin=" << (LabelType)GetImage()->GetValue(i+1,j+1,k)<<G4endl);
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j+1,k)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    if (zsurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j+1,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j+1,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    else {
    if (zsurf[0]) {
    GateDebugMessage("Volume", 6, "\t zsurf, voisin=" << (LabelType)GetImage()->GetValue(i+1,j,k-1)<<G4endl);
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    GateDebugMessage("Volume", 6, "\t zsurf, voisin=" << (LabelType)GetImage()->GetValue(i+1,j,k+1)<<G4endl);
    if ((nl=(LabelType)GetImage()->GetValue(i+1,j,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    }
    else {
    if (ysurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i,j-1,k)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    if (zsurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i,j-1,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i,j-1,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    else if (ysurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i,j+1,k)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    if (zsurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i,j+1,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i,j+1,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    else {
    if (zsurf[0]) {
    if ((nl=(LabelType)GetImage()->GetValue(i,j,k-1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    else if (zsurf[1]) {
    if ((nl=(LabelType)GetImage()->GetValue(i,j,k+1)) != cl)
    mInside[cl] = mInside[nl] = kSurface;
    }
    }
    }
    // ok, we have parsed the complete 26-neighbourhood of the point !
    */

  GateDebugMessage("Volume", 6, "\t** Result Inside for " << label << " = " << mInside[label] << G4endl);

  return mInside[label];
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Calculate the distance to the nearest surface of a shape from an
// outside point. The distance can be an underestimate.
G4double GateImageRegionalizedVolume::DistanceToIn(const G4ThreeVector& p, LabelType label)
{
  GateDebugMessage("Volume", 6, "---------------------------------------------------==" << G4endl);
  GateDebugMessage("Volume", 6,"\tGateImageRegionalizedVolume[" << GetObjectName()
		   << "]::DistanceToIn_ISO(" << p <<",lab=" << label
		   << "[" << GetMaterialNameFromLabel(label) << ", " << label << "])" << G4endl);

  G4ThreeVector coord = GetImage()->GetCoordinatesFromPosition(p);
  GateDebugMessage("Volume",6,"Coord " << coord << G4endl);

  // If position has same label ?
  int index = GetImage()->GetIndexFromCoordinates(coord);
  GateDebugMessage("Volume",6,"Index " << index << G4endl);

  if (GetImage()->GetValue(index) == label) {
    GateDebugMessage("Volume",6," DISTANCE TO IN (iso)  = 0 (INSIDE)" << G4endl);
    return 0.0;
  }

  // If not : check if on a side
  GateImage::ESide side = GetImage()->GetSideFromPointAndCoordinate(p, coord);
  GateDebugMessage("Volume",6,"Side = " << side << G4endl);

  if (side != GateImage::kUndefined) { // point is on a side
    int l = (int)GetImage()->GetNeighborValueFromCoordinate(side, coord);
    GateDebugMessage("Volume",6,"Neighbor is " << l << G4endl);
    if (l == label) {
      GateDebugMessage("Volume",6,"*** INSIDE *** (neighbor label identical)" << G4endl);
      GateDebugMessage("Volume",6," DISTANCE TO IN (iso)  = 0 (SURFACE/INSIDE)" << G4endl);
      return 0.0;
    }
    else { // Could be : about half a voxel according to distance ...
      G4double d=0.0;
      GateDebugMessage("Volume",6," DISTANCE TO IN (iso)  = " << d << G4endl);
      return d;
    }
  }

  // point is not on a side
  index = pDistanceMap->GetIndexFromPosition(p); // no side problem
  G4double d = pDistanceMap->GetValue(index);
  GateDebugMessage("Volume",6,"Is not on a side, index dmap = " << index << G4endl);
  GateDebugMessage("Volume",6," DISTANCE TO IN (iso)  = " << d << G4endl);

  return d;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4double GateImageRegionalizedVolume::DistanceToIn(const G4ThreeVector& p,
						   const G4ThreeVector& v,
						   LabelType label)
{
  GateDebugMessage("Volume", 7, "---------------------------------------------------==" << G4endl);
  GateDebugMessage("Volume", 7,"\tGateImageRegionalizedVolume["<<GetObjectName()
		   <<"]::DistanceToInWithv("
		   <<p<<","<<v
		   <<",lab="<<label<<"["<<GetMaterialNameFromLabel(label)<<"])"
		   <<G4endl);

  //  bool done=false;

  G4int TrackID = -1;
  if ( GateUserActions::GetUserActions()->GetCurrentTrack() )
    TrackID = GateUserActions::GetUserActions()->GetCurrentTrack()->GetTrackID();

  // If already computed, just return the value for the region
  if ( mLastDTIPointIsValid &&
       (mLastDTIPoint == p) &&
       (mLastDTIDirection == v) &&
       (mLastTrackID == TrackID)
       ) {
    GateDebugMessage("Volume",7,"\t** Same point and Track as before : result = "
		     << mDistanceToIn[label] << G4endl);
    return mDistanceToIn[label];
  }

  // Else compute the vector mDistanceToIn
  if ( mLastTrackID != TrackID ) {
    GateDebugMessage("Volume",7,"\t** NEW TRACK : OLD="<<mLastTrackID<<" NEW="<<TrackID<<G4endl);
  }
  else {
    GateDebugMessage("Volume",7,"\t** NEW POINT"<<G4endl);
  }
  GateDebugMessage("Volume",7,"\t### Computing complete distance vector ###" << G4endl);

  // memo last point
  mLastDTIPoint = p;
  mLastDTIDirection = v;
  mLastDTIPointIsValid = true;
  mLastTrackID =TrackID;

  // Reset the vector
  for (std::vector<G4double>::iterator i = mDistanceToIn.begin();
       i!=mDistanceToIn.end();
       ++i)
    *i = kInfinity;

  // Distance to bbox of the image
  G4double dbox = GetSolid()->DistanceToIn(p,v);

  if (dbox == kInfinity) {
    GateError("DistanceToIn : DIST = INFINITY (no inter with bbox)");
    return kInfinity;
  }

  GateDebugMessage("Volume",7,"\t\t dbox" << dbox << G4endl);

  // Initial position
  G4ThreeVector pos = p + dbox * v;

  // current image index
  int index = GetImage()->GetIndexFromPositionAndDirection(pos,v);

  //if (GetImage()->GetValue(index) == 0) {
  if ((dbox > 0.0) && (index == 0)) {

    // Propagate until reach the first region

    // resolution of the image
    G4ThreeVector r = GetImage()->GetResolution();
    // size of the voxels
    G4ThreeVector vox = GetImage()->GetVoxelSize();

    // Current position
    G4double posx(pos.x()),posy(pos.y()),posz(pos.z());

    GateDebugMessage("Volume",7,"\tRegion Label = "<<label<<G4endl);
    GateDebugMessage("Volume",7,"\tCoords = "<<GetImage()->GetCoordinatesFromPosition(p)<<G4endl);
    GateDebugMessage("Volume",7,"\tImage Label before propag. = "<<GetImage()->GetValue(p)<<G4endl);

    // Directions of propagation
    G4double dx(-1),dy(-1),dz(-1);
    if (v.x()==0) dx = 0;
    else if (v.x()>0) dx = 1;
    if (v.y()==0) dy = 0;
    else if (v.y()>0) dy = 1;
    if (v.z()==0) dz = 0;
    else if (v.z()>0) dz = 1;

    //  G4ThreeVector p2(posx,posy,posz);
    GateDebugMessage("Volume",7,"\t\t Go Inside Loop ... " <<G4endl);

    // offsets to add when changing voxel
    int dindexx = (int)(1.0 * dx);
    int dindexy = (int)(GetImage()->GetLineSize() * dy);
    int dindexz = (int)(GetImage()->GetPlaneSize() * dz);

    // Current integer position (voxel coordinates)
    G4ThreeVector i = GetImage()->GetCoordinatesFromIndex(index);

    // Position of the next x/y/z-parallel planes to intersect
    G4ThreeVector pp = GetImage()->GetVoxelCornerFromCoordinates(i);
    G4double ppx(pp.x()),ppy(pp.y()),ppz(pp.z());
    if (dx>0) ppx += vox.x() ;
    if (dy>0) ppy += vox.y() ;
    if (dz>0) ppz += vox.z() ;
    // Values to add to pp when changing voxel
    G4double dppx = vox.x()*dx;
    G4double dppy = vox.y()*dy;
    G4double dppz = vox.z()*dz;

    // (Curvi)linear abcissa parameter of the next intersection with x/y/z-parallzl planes
    G4double tx(1e30),ty(1e30),tz(1e30);


    // Propagate while current voxel value is 0 (outside)
    // ------------== DEBUG ESSAI ---------------------==
    while ( GetImage()->GetValue(index) == 0 ) {
      //while ( (GetImage()->GetValue(index) == currentLabel) ) {
      // ------------== DEBUG ESSAI ---------------------==

      GateDebugMessage("Volume",7,"\t\t --- STEP ---" <<G4endl);
      GateDebugMessage("Volume",7,"\t\t pos   = "<<posx<<","<<posy<<","<<posz<<G4endl);
      GateDebugMessage("Volume",7,"\t\t pp    = "<<ppx<<","<<ppy<<","<<ppz<<G4endl);
      GateDebugMessage("Volume",7,"\t\t index = "<<index<<G4endl);
      GateDebugMessage("Volume",7,"\t\t label = "<<GetImage()->GetValue(index)<<" vs. "<<label<<G4endl);

      // compute t
      if (dx!=0) tx = (ppx - posx) / v.x();
      if (dy!=0) ty = (ppy - posy) / v.y();
      if (dz!=0) tz = (ppz - posz) / v.z();
      GateDebugMessage("Volume",7,"\t\t    = "<<tx<<","<<ty<<","<<tz<<G4endl);
      //
      if ( tx < ty ) {
	if ( tx < tz ) {
	  posx += tx * v.x();
	  posy += tx * v.y();
	  posz += tx * v.z();
	  index += dindexx;
	  ppx += dppx;
	}
	else {
	  posx += tz * v.x();
	  posy += tz * v.y();
	  posz += tz * v.z();
	  index += dindexz;
	  ppz += dppz;
	}
      }
      //
      else {
	if ( ty < tz ) {
	  posx += ty * v.x();
	  posy += ty * v.y();
	  posz += ty * v.z();
	  index += dindexy;
	  ppy += dppy;
	}
	else {
	  posx += tz * v.x();
	  posy += tz * v.y();
	  posz += tz * v.z();
	  index += dindexz;
	  ppz += dppz;
	}
      }
    }
    // End propagation

  } // end if dbox ....

  GateDebugMessage("Volume",7,"\t\t --- EXIT ---" <<G4endl);
  GateDebugMessage("Volume",7,"\t\t index = "<<index<<G4endl);
  GateDebugMessage("Volume",7,"\t\t Image Label after propag. = "<<GetImage()->GetValue(index)<<G4endl);

  // ------------== DEBUG ESSAI ---------------------==
  // if (!done)
  mDistanceToIn[ (LabelType)GetImage()->GetValue(index) ] = dbox;
  // ------------== DEBUG ESSAI ---------------------==

  GateDebugMessage("Volume",7,"\t** DIST = "<<dbox <<G4endl);
  GateDebugMessage("Volume",7,"\t** DIST = mDistanceToIn[ label ] = "<<mDistanceToIn[ label ] <<G4endl);
  return mDistanceToIn[ label ];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateImageRegionalizedVolume::DistanceToOut(const G4ThreeVector& p,
						    const G4ThreeVector& v,
						    const G4bool calcNorm,
						    G4bool *validNorm,
						    G4ThreeVector *n,
						    LabelType label)
{
  GateDebugMessage("Volume", 8, "---------------------------------------------------==" << G4endl);
  GateDebugMessage("Volume",8,"\tGateImageRegionalizedVolume["<<GetObjectName()
		   <<"]::DistanceToOut("
		   <<p<<","<<v
		   <<",lab="<<label<<"["<<GetMaterialNameFromLabel(label)<<"])"
		   <<G4endl);

  // ------------------------------------------------------==
  // Ray tracing algorithm

  // resolution of the image
  //const G4ThreeVector & res = GetImage()->GetResolution();
  // size of the voxels
  const G4ThreeVector & vox = GetImage()->GetVoxelSize();

  // Current position
  G4double posx(p.x()),posy(p.y()),posz(p.z());

  // Directions of propagation
  G4double dx(-1),dy(-1),dz(-1);
  if (v.x()==0) dx = 0;
  else if (v.x()>0) dx = 1;
  if (v.y()==0) dy = 0;
  else if (v.y()>0) dy = 1;
  if (v.z()==0) dz = 0;
  else if (v.z()>0) dz = 1;

  // current image index
  int index = GetImage()->GetIndexFromPositionAndDirection(p,v);
  GateDebugMessage("Volume",8,"\t\tGetIndex(p,v) = "<< index << G4endl);

  // ------------------------------------------------------==
  // ASSERT DEBUG
  // if (index == -1) {
  // 	GateError("point is outside image ??? p = " << p << G4endl);
  //   }
  // ------------------------------------------------------==

  // ------------------------------------------------------==
  // Current integer position (voxel coordinates)
  G4ThreeVector i = GetImage()->GetCoordinatesFromIndex(index);
  GateDebugMessage("Volume",8,"\t\tCoord = "<< i << G4endl);

  // Test if outside given label (surface)
  GateDebugMessage("Volume",8,"\t\tLabel = " << GetImage()->GetValue(index) << G4endl);
  if (GetImage()->GetValue(index) != label) {
    GateDebugMessage("Volume",8,"\t\tInitialisation **OUTSIDE** the region -> return 0.0"<<G4endl);

    /*
    // Gate case rarely occur, but occur ... what to do ?
    LabelType l = (LabelType)GetImage()->GetNeighborValueFromCoordinate(side, i);
    GateDebugMessage("Volume",8,"\t\tNeigh label = " << l << G4endl);
    if (l != label) {
    // CA ARRIVE (RARE) !!!
    GateError("Label neighbor different from label !");
    }
    */

    if (calcNorm) {
      GateImage::ESide side = GetImage()->GetSideFromPointAndCoordinate(p, i);
      GateDebugMessage("Volume",8,"\t\tSide        = " << side << G4endl);
      GateDebugMessage("Volume",6," calcNorm is USED ... " <<G4endl);
      n->set(0,0,0);
      *validNorm = true;
      switch (side) {
      case GateImage::kMX: n->setX(1); break;
      case GateImage::kPX: n->setX(-1); break;
      case GateImage::kMY: n->setY(1); break;
      case GateImage::kPY: n->setY(-1); break;
      case GateImage::kMZ: n->setZ(1); break;
      case GateImage::kPZ: n->setZ(-1); break;
      default:
	*validNorm = false;
	GateDebugMessage("Volume",6,"Error ! not on surface ????");
	// Gate case rarely occur, but occur ... what to do ?
      }
      GateDebugMessage("Volume",6," validNorm = " << *validNorm << G4endl);
      GateDebugMessage("Volume",6," norm = " << *n << G4endl);
    }

    GateDebugMessage("Volume",6,"\t***** DISTANCE = " << 0.0 << G4endl);

    return 0.0;
  }

  // ------------------------------------------------------==
  // Initialisation for ray tracing

  if ((v.x() == v.y()) && (v.x() == v.z()) && (v.x() == 0.0)) {
    GateDebugMessage("Volume",6,"Direction vanish !! return " << kInfinity << G4endl);
    return kInfinity;
  }

  // offsets to add when changing voxel
  int dindexx = (int)lrint(1.0 * dx);
  int dindexy = (int)lrint(GetImage()->GetLineSize() * dy);
  int dindexz = (int)lrint(GetImage()->GetPlaneSize() * dz);
  GateDebugMessage("Volume",8,"\t\t dindexx " << dindexx << G4endl);
  GateDebugMessage("Volume",8,"\t\t dindexy " << dindexy << G4endl);
  GateDebugMessage("Volume",8,"\t\t dindexz " << dindexz << G4endl);

  // Position of the next x/y/z-parallel planes to intersect
  G4ThreeVector pp = GetImage()->GetVoxelCornerFromCoordinates(i);
  GateDebugMessage("Volume",8,"\t\tCorner = "<< pp << G4endl);

  G4double ppx(pp.x()),ppy(pp.y()),ppz(pp.z());
  if (dx>0) ppx += vox.x() ;
  if (dy>0) ppy += vox.y() ;
  if (dz>0) ppz += vox.z() ;
  // Values to add to pp when changing voxel
  G4double dppx = vox.x()*dx;
  G4double dppy = vox.y()*dy;
  G4double dppz = vox.z()*dz;

  // (Curvi)linear abcissa parameter of the next intersection with x/y/z-parallzl planes
  G4double tx(1e30),ty(1e30),tz(1e30);
  // last step intersected a x,y or z parallel plane (0,1,2) ?
  int last_step = 0;
  GateDebugMessage("Volume",8,"\t\t dir   = "<<v.x()<<","<<v.y()<<","<<v.z()<<G4endl);

  int nbStep = 0; // DEBUG

  //---------------------------------------------------------------
  // Ray tracing

  // Propagate while current voxel value is the region's label
  bool stop = false;
  do {
    GateDebugMessage("Volume",9,"\t\t do loop index = " << index << G4endl);
    GateDebugMessage("Volume",9,"\t\t do loop pos   = " << posx << " " << posy << " " << posz << G4endl);
    GateDebugMessage("Volume",9,"\t\t do loop v     = " << v << G4endl);
    GateDebugMessage("Volume",9,"\t\t v loop  pp    = " << ppx << " " << ppy << " " << ppz << G4endl);
    GateDebugMessage("Volume",9,"\t\t v loop  dpp   = " << dppx << " " << dppy << " " << dppz << G4endl);
    GateDebugMessage("Volume",9,"\t\t v loop  t     = " << tx << " " << ty << " " << tz << G4endl);
    GateDebugMessage("Volume",9,"\t\t v loop  d     = " << dx << " " << dy << " " << dz << G4endl);

    // First test : didn't seem to accelerate...-> comment out for the moment
    // To test : use dmap only for a minimum jump length
    /*
    // Use distance map to jump ?
    if ( use_dmap && ( dm = pDistanceMap->GetValue(index) > 0 ) ) {
    posx += dm * nvx;
    posy += dm * nvy;
    posz += dm * nvz;
    int nindex = GetImage()->GetIndexFromPositionAndDirection(G4ThreeVector(posx,posy,posz),v);
    if (nindex == index) {
    use_dmap = false;
    }
    else {
    index = nindex;
    // update ppx,ppy,ppz
    // Current integer position (voxel coordinates)
    i = GetImage()->GetCoordinatesFromIndex(index);
    // Position of the next x/y/z-parallel planes to intersect
    pp = GetImage()->GetVoxelCornerFromCoordinates(i);
    ppx = pp.x();
    ppy = pp.y();
    ppz = pp.z();
    if (dx>0) ppx += vox.x() ;
    if (dy>0) ppy += vox.y() ;
    if (dz>0) ppz += vox.z() ;
    }
    }
    // dmap==0 : Simple step to the next voxel
    else {

    use_dmap = true;
    */

    // compute t
    if (dx!=0) tx = (ppx - posx) / v.x();
    if (dy!=0) ty = (ppy - posy) / v.y();
    if (dz!=0) tz = (ppz - posz) / v.z();
    //
    if ( tx < ty ) {
      if ( tx < tz ) {
	last_step = 0;
	posx += tx * v.x();
	posy += tx * v.y();
	posz += tx * v.z();
	index += dindexx;
	ppx += dppx;
      }
      else {
	last_step = 2;
	posx += tz * v.x();
	posy += tz * v.y();
	posz += tz * v.z();
	index += dindexz;
	ppz += dppz;
      }
    }
    //
    else {
      if ( ty < tz ) {
	last_step = 1;
	posx += ty * v.x();
	posy += ty * v.y();
	posz += ty * v.z();
	index += dindexy;
	ppy += dppy;
      }
      else {
	last_step = 2;
	posx += tz * v.x();
	posy += tz * v.y();
	posz += tz * v.z();
	index += dindexz;
	ppz += dppz;
      }
    }
    //	 } // use dmap

    nbStep++;
    if (nbStep > 1000) {
      //GateError("nbStep > 1000");
    }

    GateDebugMessage("Volume", 9, "\t\t nbStep " << nbStep << G4endl);
    GateDebugMessage("Volume", 9, "\t\t index " << index << G4endl);
    GateDebugMessage("Volume", 9, "\t\t vox center" << GetImage()->GetVoxelCenterFromIndex(index) << G4endl);
    GateDebugMessage("Volume", 9, "\t\t half size" << GetImage()->GetHalfSize() << G4endl);

    if ((posx <= -GetImage()->GetHalfSize().x()) ||
        (posx >= GetImage()->GetHalfSize().x())) {
      GateDebugMessage("Volume", 9, "VOXOUT X " << posx << " " << GetImage()->GetHalfSize() << G4endl);
      stop = true;
    }
    if ((posy <= -GetImage()->GetHalfSize().y()) ||
        (posy >= GetImage()->GetHalfSize().y())) {
      GateDebugMessage("Volume", 9, "VOXOUT Y " << posy << " " << GetImage()->GetHalfSize() << G4endl);
      stop = true;
    }
    if ((posz <= -GetImage()->GetHalfSize().z()) ||
        (posz >= GetImage()->GetHalfSize().z())) {
      GateDebugMessage("Volume", 9, "VOXOUT Z " << posz << " " << GetImage()->GetHalfSize() << G4endl);
      stop = true;
    }

  }
  while ( (!stop) && (GetImage()->GetValue(index) == label) );

  GateDebugMessage("Volume",9,"\t\t --- EXIT ---" <<G4endl);
  GateDebugMessage("Volume",9,"\t\t pos   = "<< posx <<"," << posy << "," << posz << G4endl);
  GateDebugMessage("Volume",9,"\t\t pp    = "<< ppx <<"," << ppy << "," << ppz << G4endl);
  GateDebugMessage("Volume",9,"\t\t index = "<< index << G4endl);
  GateDebugMessage("Volume",9,"\t\t step  = "<< nbStep << G4endl);
  GateDebugMessage("Volume",7,"\t\t Image Label after propag. = " << GetImage()->GetValue(index)<<G4endl);

  /*
    if (nbStep != 0) {
    mMeanNbStep +=  nbStep;
    mNbRT++;
    }
    else mNbZeroStep++;
  */
  // End of propagation

  // Compute distance
  G4double dpx = posx - p.x();
  G4double dpy = posy - p.y();
  G4double dpz = posz - p.z();
  G4double dist = sqrt ( dpx*dpx + dpy*dpy + dpz*dpz );

  // Normal to the surface depends on last_step :
  if (calcNorm) {
    GateDebugMessage("Volume",6," calcNorm is USED ... " <<G4endl);
    n->set(0,0,0);
    *validNorm = true;
    if (last_step == 0) {
      n->setX(dx);
    }
    else if (last_step == 1) {
      n->setY(dy);
    }
    else {
      n->setZ(dz);
    }
    GateDebugMessage("Volume",6," validNorm = " << *validNorm << G4endl);
    GateDebugMessage("Volume",6," norm = " << *n << G4endl);
  }

  GateDebugMessage("Volume",8,"\t***** DISTANCE = "<<dist<<G4endl);

  return dist;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateImageRegionalizedVolume::DistanceToOut(const G4ThreeVector& p, LabelType label)
{
  GateDebugMessage("Volume", 6, "---------------------------------------------------==" << G4endl);
  GateDebugMessage("Volume", 6, "\tGateImageRegionalizedVolume[" << GetObjectName()
		   << "]::DistanceToOutIso(" << p
		   << ",lab=" << label << "[" << GetMaterialNameFromLabel(label) << "])"
		   << G4endl);

  // Cache :
  static G4ThreeVector lPreviousP;
  static G4double lPreviousD;
  static LabelType lPreviousL;
  if ((label == lPreviousL) && (p == lPreviousP)){
    GateDebugMessage("Volume",6,"Point is in cache !" << G4endl);
    GateDebugMessage("Volume",6," DISTANCETOOUT = " << lPreviousD <<G4endl);
    //	mNbCache++;
    return lPreviousD;
  }
  else {
    lPreviousP = p;
    lPreviousL = label;
  }

  G4ThreeVector coord = GetImage()->GetCoordinatesFromPosition(p);
  GateDebugMessage("Volume",6,"Coord " << coord << G4endl);


  /* ------------------------==
     The "side" test seems to be good but  costly.
     So it is removed.
     ---------------------------== */


  /* GateImage::ESide side = GetImage()->GetSideFromPointAndCoordinate(p, coord);
     GateDebugMessage("Volume",6,"Side = " << side << G4endl);
  */

  int index = GetImage()->GetIndexFromCoordinates(coord);
  LabelType l = (LabelType)GetImage()->GetValue(index);
  GateDebugMessage("Volume",6,"Index = " << index << " la=" << l << G4endl);

  // Not on a side
  if (l == label) {
    index = pDistanceMap->GetIndexFromPosition(p);
    G4double d = pDistanceMap->GetValue(index);
    if (d!=0) {
      lPreviousD = d;
      return d;
    }

    GateDebugMessage("Volume",6,"Dmap index = " << index << " d=" << d << G4endl);
    // find min distance to voxel side
    const G4ThreeVector & v = GetImage()->GetVoxelCornerFromCoordinates(coord);
    const G4ThreeVector & s = GetImage()->GetVoxelSize();
    GateDebugMessage("Volume",6,"Voxel = " << v << G4endl);
    GateDebugMessage("Volume",6,"Smac  = " << s << G4endl);
    GateDebugMessage("Volume", 6, "mx = " << p.x()-v.x() << G4endl);
    GateDebugMessage("Volume", 6, "px = " << v.x()+s.x()-p.x() << G4endl);
    GateDebugMessage("Volume", 6, "my = " << p.y()-v.y() << G4endl);
    GateDebugMessage("Volume", 6, "py = " << v.y()+s.y()-p.y() << G4endl);
    GateDebugMessage("Volume", 6, "mz = " << p.z()-v.z() << G4endl);
    GateDebugMessage("Volume", 6, "pz = " << v.z()+s.z()-p.z() << G4endl);

    G4double offset = p.x()-v.x();
    offset = std::min(offset, v.x()+s.x()-p.x());
    offset = std::min(offset, v.y()+s.y()-p.y());
    offset = std::min(offset, v.z()+s.z()-p.z());
    offset = std::min(offset, p.y()-v.y());
    offset = std::min(offset, p.z()-v.z());

    d += offset;
    GateDebugMessage("Volume",6," DISTANCETOOUT = " << d <<G4endl);
    lPreviousD = d;
    return d;

    // WARNING : do the following only if (d == 0) is sufficient
    // However I do it every time to as close as possible to Boxes ...
    // Maybe it slow down the simulations

    // WARNING : THIS IS FALSE : CONSIDER CORNER ....
    /*
      G4ThreeVector pos(coord);
      int dx = (int)lrint(d/GetImage()->GetVoxelSize().x())+1;
      int dy = (int)lrint(d/GetImage()->GetVoxelSize().y())+1;
      int dz = (int)lrint(d/GetImage()->GetVoxelSize().z())+1;

      offset = 9999999;
      float value;
      // kPX
      pos.setX(std::min(GetImage()->GetResolution().x()-1, std::max(0.0,coord.x()+dx)));
      value = GetImage()->GetValue(GetImage()->GetIndexFromCoordinates(pos));
      GateDebugMessage("Volume", 6, "Check pos = " << pos << " " << value << G4endl);
      if (value != label) {
      offset = std::min(offset, v.x()+s.x()-p.x());
      GateDebugMessage("Volume", 6, "Offset = " << offset << G4endl);
      }

      // kMX
      pos.setX(std::min(GetImage()->GetResolution().x()-1, std::max(0.0,coord.x()-dx)));
      value = GetImage()->GetValue(GetImage()->GetIndexFromCoordinates(pos));
      GateDebugMessage("Volume", 6, "Check pos = " << pos << " " << value << G4endl);
      if (value != label) {
      offset = std::min(offset, p.x() - v.x());
      GateDebugMessage("Volume", 6, "Offset = " << offset << G4endl);
      }

      // kPY
      pos=coord;
      pos.setY(std::min(GetImage()->GetResolution().y()-1, std::max(0.0,coord.y()+dy)));
      value = GetImage()->GetValue(GetImage()->GetIndexFromCoordinates(pos));
      GateDebugMessage("Volume", 6, "Check pos = " << pos << " " << value << G4endl);
      if (value != label) {
      offset = std::min(offset, v.y()+s.y()-p.y());
      GateDebugMessage("Volume", 6, "Offset = " << offset << G4endl);
      }

      // kMY
      pos.setY(std::min(GetImage()->GetResolution().y()-1, std::max(0.0,coord.y()-dy)));
      value = GetImage()->GetValue(GetImage()->GetIndexFromCoordinates(pos));
      GateDebugMessage("Volume", 6, "Check pos = " << pos << " " << value << G4endl);
      if (value != label) {
      offset = std::min(offset, p.y() - v.y());
      GateDebugMessage("Volume", 6, "Offset = " << offset << G4endl);
      }

      // kPZ
      pos=coord;
      pos.setZ(std::min(GetImage()->GetResolution().z()-1, std::max(0.0,coord.z()+dz)));
      value = GetImage()->GetValue(GetImage()->GetIndexFromCoordinates(pos));
      GateDebugMessage("Volume", 6, "Check pos = " << pos << " " << value << G4endl);
      if (value != label) {
      offset = std::min(offset, v.z()+s.z()-p.z());
      GateDebugMessage("Volume", 6, "Offset = " << offset << G4endl);
      }

      // kMZ
      pos.setZ(std::min(GetImage()->GetResolution().z()-1, std::max(0.0,coord.z()-dz)));
      value = GetImage()->GetValue(GetImage()->GetIndexFromCoordinates(pos));
      GateDebugMessage("Volume", 6, "Check pos = " << pos << " " << value << G4endl);
      if (value != label) {
      offset = std::min(offset, p.z() - v.z());
      GateDebugMessage("Volume", 6, "Offset = " << offset << G4endl);
      }

      if (offset != 9999999) d += offset;

      GateDebugMessage("Volume",6," DISTANCETOOUT = " << d <<G4endl);
      lPreviousD = d;
      return d;
    */
  }

  GateDebugMessage("Volume",6," DISTANCETOOUT = " << 0.0 <<G4endl);
  lPreviousD = 0.0;
  return 0.0;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4ThreeVector GateImageRegionalizedVolume::SurfaceNormal(const G4ThreeVector& p, LabelType label)
{
  GateDebugMessage("Volume", 6, "---------------------------------------------------==" << G4endl);
  GateDebugMessage("Volume", 6,"\tGateImageRegionalizedVolume["<<GetObjectName()
		   << "]::SurfaceNormal("
		   << p
		   << ",lab="<<label<<"["<<GetMaterialNameFromLabel(label)<<"])"
		   << G4endl);
  //
  // Check surface
  G4ThreeVector coord = GetImage()->GetCoordinatesFromPosition(p);
  GateDebugMessage("Volume",6,"Coord " << coord << G4endl);

  GateImage::ESide side = GetImage()->GetSideFromPointAndCoordinate(p, coord);
  GateDebugMessage("Volume",6,"Side = " << side << G4endl);

  if (side == GateImage::kUndefined) { // not on surface
    // GateError("Not on surface ??");


    // WARNING !!!!!!!!!!!



    GateDebugMessage("Volume",5,"SurfaceNormal :: Not on surface ??" << G4endl);
    // IT APPENDS (rare) !! WHY ???? I DONT KNOW WHAT TO DO
    G4ThreeVector normal(1.0,0.0,0.0);
    return normal;
  }

  int index = GetImage()->GetIndexFromCoordinates(coord);
  LabelType l = (LabelType)GetImage()->GetValue(index);
  GateDebugMessage("Volume",6,"Index = " << index << " la=" << l << G4endl);

  // ASSERT DEBUG
  LabelType nl =  (LabelType)GetImage()->GetNeighborValueFromCoordinate(side, coord);
  GateDebugMessage("Volume",6,"Neighbor lab = " << nl << G4endl);
  if (nl == l) {
    GateDebugMessage("Volume",6,"Neighbors has same label !??");
    // IT APPENDS (rare) !! WHY ???? I DONT KNOW WHAT TO DO
    G4ThreeVector normal(1.0,0.0,0.0);
    return normal;
  }

  G4ThreeVector normal(0.0,0.0,0.0);
  double swap;
  if (label == l) swap = 1;
  else swap = -1;
  switch(side) {
  case GateImage::kMX: normal.setX(-swap); break;
  case GateImage::kPX: normal.setX(swap); break;
  case GateImage::kMY: normal.setY(-swap); break;
  case GateImage::kPY: normal.setY(swap); break;
  case GateImage::kMZ: normal.setZ(-swap); break;
  case GateImage::kPZ: normal.setZ(swap); break;
  default: GateError("Side not known!");
  }

  GateDebugMessage("Volume", 9,"\t norm = " << normal << G4endl);
  return normal;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Used by navigation to know the next entered subvolume
// returns 0 if the particle exits the volume
G4VPhysicalVolume* GateImageRegionalizedVolume::GetEnteredPhysicalVolume( const G4ThreeVector& globalPoint,
									  const G4ThreeVector* globalDirection )
{
  GateDebugMessage("Navigation", 4,"GetEnteredPhysicalVolume("<<globalPoint<<","<<globalDirection<<")"<<std::endl);

  int index;
  if (globalDirection)
    {
      index = GetImage()->GetIndexFromPositionAndDirection(globalPoint,
							   *globalDirection);
    }
  else
    {
      index = GetImage()->GetIndexFromPosition(globalPoint);
    }
  LabelType l = (LabelType)GetImage()->GetValue(index);

  GateDebugMessage("Navigation",4,"label = "<<l<<std::endl);
  if (l==0) return 0;

  GateDebugMessage("Navigation",4,"phys = "<<mLabelToSubVolume[l]->GetPhysicalVolume()->GetName()<<std::endl);

  return ( mLabelToSubVolume[l]->GetPhysicalVolume() );
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Methods used by GateImageRegionalizedVolumeNavigation
G4double GateImageRegionalizedVolume::ComputeSafety(const G4ThreeVector& p)
{
  GateDebugMessage("Navigation", 4,"GateImageRegionalizedVolume["
		   << GetObjectName()
		   << "]::ComputeSafety(" << p <<")" << G4endl);

  int index = pDistanceMap->GetIndexFromPosition(p); // no side problem
  G4double d = pDistanceMap->GetValue(index);

  GateDebugMessage("Navigation",6," Safety = " << d << G4endl);

  return d;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateImageRegionalizedVolume::ComputeStep(const G4ThreeVector& /*position*/, // warning if no debug
						  const G4ThreeVector& /*direction*/,// warning if no debug
						  const G4double /*currentProposedStepLength*/,
						  //G4double& newSafety,
						  //              G4NavigationHistory& history,
						  G4bool& /*validExitNormal*/,
						  G4ThreeVector& /*exitNormal*/,
						  G4bool& /*exiting*/,
						  G4bool& /*entering*/,
						  G4VPhysicalVolume *(*/*pBlockedPhysical*/))
{

  // GateDebugMessage("Navigation",4,"GateImageRegionalizedVolume["
  // 		   <<GetObjectName()
  // 		   <<"]::ComputeStep("
  // 		   <<position<<","<<direction
  // 		   <<G4endl);
  return 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageRegionalizedVolume::PropagateSensitiveDetectorToChild(GateMultiSensitiveDetector * /*msd*/)
{
  //GateDebugMessage("Volume", 5, "Add SD to child" << G4endl);
  // logXRep->SetSensitiveDetector(msd);
  // logYRep->SetSensitiveDetector(msd);
  // logZRep->SetSensitiveDetector(msd);
}
//---------------------------------------------------------------------------


//-----------------------------------------------------------------------------------------
// We reimplement this function to set the SD to the sub volumes
void GateImageRegionalizedVolume::AttachPhantomSD()
{
  GateVVolume::AttachPhantomSD();
  for(unsigned int i=0; i< mSubVolume.size(); i++) {
    mSubVolume[i]->AttachPhantomSD();
  }
}
//------------------------------------------------------------------------------------------
