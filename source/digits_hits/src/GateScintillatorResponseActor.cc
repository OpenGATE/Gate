/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GateScintillatorResponseActor :
  \brief 
*/
#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

// Gate
#include "GateScintillatorResponseActor.hh"
#include "GateScatterOrderTrackInformationActor.hh"
#include <rtkReg23ProjectionGeometry.h>

//-----------------------------------------------------------------------------
GateScintillatorResponseActor::GateScintillatorResponseActor(G4String name, G4int depth):
  GateVImageActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateScintillatorResponseActor() -- begin"<<G4endl);
  pMessenger = new GateScintillatorResponseActorMessenger(this);
  SetStepHitType("pre");
  GateDebugMessageDec("Actor",4,"GateScintillatorResponseActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateScintillatorResponseActor::~GateScintillatorResponseActor()
{
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateScintillatorResponseActor::Construct()
{
  GateDebugMessageInc("Actor", 4, "GateScintillatorResponseActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct(); // mImage is not allocated here
  mImage.Allocate();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnableUserSteppingAction(true);

  // the image index will be computed according to the preStep
  if (mStepHitType != PreStepHitType) {
    GateWarning("The stepHitType must be 'pre', we force it.");
    SetStepHitType("pre");
  }
  SetStepHitType("pre");

  // Allocate scatter image
  if (mIsScatterImageEnabled) {
    mImageScatter.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mImageScatter.Allocate();
    mImageScatter.SetOrigin(mOrigin);
  }

  GateVVolume * v = GetVolume();
  G4VPhysicalVolume * phys = v->GetPhysicalVolume();
  mDetectorToWorld = G4AffineTransform(phys->GetRotation(), phys->GetTranslation());
  // If detector is inside another volume
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    mDetectorToWorld = mDetectorToWorld * x;
  }


  // Print information
  GateMessage("Actor", 1, 
              "\tScintillatorResponse ScintillatorResponseActor    = '" << GetObjectName() << "'" << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateScintillatorResponseActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateScintillatorResponseActor::ReadMuAbsortionList(G4String muAbsortionlist)
{
  G4double energy, muAttenuation, muAbsortion;
  std::ifstream inMuAbsortionFile;
  mUserMuAbsortionMap.clear( );

  inMuAbsortionFile.open( muAbsortionlist);
  if( !inMuAbsortionFile ) { // file couldn't be opened
    G4cout << "Error: file could not be opened" << G4endl;
    exit( 1);
  }
  while ( !inMuAbsortionFile.eof( ))
    {
      inMuAbsortionFile >> energy >> muAttenuation >> muAbsortion;
      energy = energy*MeV;
      mUserMuAbsortionMap[ energy] = muAbsortion;
    }
  inMuAbsortionFile.close( );
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
/// Save data
void GateScintillatorResponseActor::SaveData()
{
  G4int rID = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();
  char filename[1024];
  // Printing all particles
  GateVImageActor::SaveData();
  if(mSaveFilename != "")
  {
    sprintf(filename, mSaveFilename, rID);
    mImage.Write(filename);
    // Printing just scatter
    if(mIsScatterImageEnabled)
    {
      G4String fn = removeExtension(filename)+"-scatter."+G4String(getExtension(filename));
      mImageScatter.Write(fn);
    }
  }
  // Printing scatter of each order
  if(mScatterOrderFilename != "")
  {
    for(unsigned int k = 0; k<mScintillatorResponsePerOrderImages.size(); k++)
    {
      sprintf(filename, mScatterOrderFilename, rID, k+1);
      mScintillatorResponsePerOrderImages[k]->Write((G4String)filename);
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateScintillatorResponseActor::ResetData()
{
  mImage.Fill(0);
  if(mIsScatterImageEnabled) {
    mImageScatter.Fill(0);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateScintillatorResponseActor::BeginOfEventAction(const G4Event * e)
{ 
  GateVActor::BeginOfEventAction(e);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateScintillatorResponseActor::UserSteppingActionInVoxel(const int index, const G4Step* step)
{
  GateDebugMessageInc("Actor", 4, "GateScintillatorResponseActor -- UserSteppingActionInVoxel - begin" << G4endl);

  // Is this necessary?
  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateScintillatorResponseActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }
  
  GateScatterOrderTrackInformation * info = dynamic_cast<GateScatterOrderTrackInformation *>(step->GetTrack()->GetUserInformation());

  /* http://geant4.org/geant4/support/faq.shtml
     To check that the particle has just entered in the current volume
     (i.e. it is at the first step in the volume; the preStepPoint is at the boundary):
  */
  if (step->GetPreStepPoint()->GetStepStatus() == fGeomBoundary)
  {
    // Photon incident energy
    double photonEnergy = (step->GetPreStepPoint()->GetKineticEnergy());


    // Mu Absortion Coeficient (linear interpolation to obtain the right value from the list)
    std::map< G4double, G4double >::iterator iterMuAbsortionMap = mUserMuAbsortionMap.end();
    iterMuAbsortionMap =  mUserMuAbsortionMap.lower_bound( photonEnergy);
    if( iterMuAbsortionMap == mUserMuAbsortionMap.end())
    {
        G4cout << " Photon Energy outside the Mu Absortion Coeficient list" << G4endl;
        exit(1);
    }
    double upperEn = iterMuAbsortionMap->first;
    double upperMu = iterMuAbsortionMap->second;
    iterMuAbsortionMap--;
    double lowerEn = iterMuAbsortionMap->first;
    double lowerMu = iterMuAbsortionMap->second;
    double muAbsortionInterp = ((( upperMu - lowerMu)/( upperEn - lowerEn)) * ( photonEnergy - upperEn) + upperMu);


    // cos(Theta) computation: scalar product: NormalVectorToDetectorSurface * IncidentPhotonDirection
    G4ThreeVector normalVectorDetec = mDetectorToWorld.TransformAxis(G4ThreeVector(0,0,1));
    G4ThreeVector incidentPhotDirec = step->GetPreStepPoint()->GetMomentumDirection();
    rtk::Reg23ProjectionGeometry::VectorType itkVectorDetec, itkPhotonDirec;
    for(unsigned int i=0; i<3; i++)
    {
      itkVectorDetec[i] = normalVectorDetec[i];
      itkPhotonDirec[i] = incidentPhotDirec[i];
    }
    const double itkVectorDetecNorm = itkVectorDetec.GetNorm();
    const double itkPhotonDirecNorm = itkPhotonDirec.GetNorm();
    double cosT = itkVectorDetec * itkPhotonDirec / ( itkVectorDetecNorm * itkPhotonDirecNorm);


    // Pixel Area
    double pixelArea = (mVoxelSize[0]*mVoxelSize[1]);

    // Dose deposed into the pixel Detector above the Scintillator by one photon
    double doseScintillator = muAbsortionInterp * photonEnergy / ( pixelArea * cosT);


    mImage.AddValue(index, doseScintillator);
    // Scatter order
    if(info)
    {
      unsigned int order = info->GetScatterOrder();
      if(order)
      {
        while(order>mScintillatorResponsePerOrderImages.size() && order>0)
        {
          GateImage * voidImage = new GateImage;
          voidImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
          voidImage->Allocate();
          voidImage->SetOrigin(mOrigin);
          voidImage->Fill(0);
          mScintillatorResponsePerOrderImages.push_back( voidImage );
        }
          mScintillatorResponsePerOrderImages[order-1]->AddValue(index, doseScintillator);
      }
    }
    
    if(mIsScatterImageEnabled &&
       !step->GetTrack()->GetParentID() &&
       !step->GetTrack()->GetDynamicParticle()->GetPrimaryParticle()->GetMomentum().isNear( step->GetTrack()->GetDynamicParticle()->GetMomentum()))
        {
            mImageScatter.AddValue(index, doseScintillator);
        }
    }

  GateDebugMessageDec("Actor", 4, "GateScintillatorResponseActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------

#endif // GATE_USE_RTK
