/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateNanoActor : This actor produces a voxelised image of the energy deposited by the optical
                               photons that were absorbed by the nano material (through the Physics process
                               NanoAbsorption) and a voxelised image of the diffusion of this energy obtained
                               as the solution of the heat equation at a later time.
  \brief
*/

#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>
#include "GateNanoActor.hh"
#include "GateMiscFunctions.hh"
#include "G4VProcess.hh"

#include "GateMHDImage.hh"
#include "GateImageT.hh"
#include "GateMiscFunctions.hh"
#include "GateMachine.hh"


//-----------------------------------------------------------------------------
GateNanoActor::GateNanoActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateNanoActor() -- begin"<<G4endl);

  mCurrentEvent=-1;
  mIsNanoAbsorptionImageEnabled = false;

  pMessenger = new GateNanoActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateNanoActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateNanoActor::~GateNanoActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

void GateNanoActor::setGaussianSigma(G4double sigma)
{
  gaussian_sigma = sigma;
}
//-----------------------------------------------------------------------------

void GateNanoActor::setTime(G4double t)
{
  diffusion_time = t;
}
//-----------------------------------------------------------------------------

void GateNanoActor::setDiffusivity(G4double diffusivity)
{
  material_diffusivity = diffusivity;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Construct
void GateNanoActor::Construct() {

  GateDebugMessageInc("Actor", 4, "GateNanoActor -- Construct - begin" << G4endl);

  GateVImageActor::Construct();

  // Record the stepHitType
  mUserStepHitType = mStepHitType;

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableEndOfRunAction(true); // for save
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);


  // Check if at least one image is enabled
  if (!mIsNanoAbsorptionImageEnabled)  {
    GateError("The NanoActor " << GetObjectName()
              << " does not have any image enabled ...\n Please select at least one");
  }

  // Output Filename
  mNanoAbsorptionFilename = G4String(removeExtension(mSaveFilename))+"-AbsorptionMap."+G4String(getExtension(mSaveFilename));
  mHeatDiffusionFilename = G4String(removeExtension(mSaveFilename))+"-HeatDiffusionMap."+G4String(getExtension(mSaveFilename));


  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mNanoAbsorptionImage);


  // Resize and allocate images
  if (mIsNanoAbsorptionImageEnabled) {

    mNanoAbsorptionImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mNanoAbsorptionImage.Allocate();
    mNanoAbsorptionImage.SetFilename(mNanoAbsorptionFilename);

  }

  // Print information
  GateMessage("Actor", 1,
              "\tNanoActor    = '" << GetObjectName() << "'" << G4endl <<
              "\tNanoAbsorptionFilename      = " << mNanoAbsorptionFilename << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateNanoActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateNanoActor::SaveData() {
  GateVActor::SaveData(); // (not needed because done into GateImageWithStatistic)

  if (mIsNanoAbsorptionImageEnabled) mNanoAbsorptionImage.SaveData(mCurrentEvent+1);


}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::ResetData() {

  if (mIsNanoAbsorptionImageEnabled) mNanoAbsorptionImage.Reset();

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateNanoActor -- Begin of Run" << G4endl);
}
//-----------------------------------------------------------------------------

// default callback for EndOfRunAction allowing to call Save

void GateNanoActor::EndOfRunAction(const G4Run* r)
{  

  GateVActor::EndOfRunAction(r);

  typedef itk::Image<float, 3>   ImageType;
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typedef itk::ImageFileWriter<ImageType> WriterType;
  typedef itk::RecursiveGaussianImageFilter<ImageType, ImageType >  GaussianFilterType;


  WriterType::Pointer writer = WriterType::New();
  ReaderType::Pointer inputfileReader = ReaderType::New();

  inputfileReader->SetFileName( mNanoAbsorptionFilename );

// NEW
//ImageType::IndexType pixelIndex;
//pixelIndex[0] = x;      // x position of the pixel
//pixelIndex[1] = y;      // y position of the pixel
//pixelIndex[2] = z;      // z position of the pixel
//ImageType::PixelType      pixelValue = inputfileReader->GetOutput()->GetPixel( pixelIndex );
// NEW


  GaussianFilterType::Pointer RecursiveGaussianImageFilter = GaussianFilterType::New();
  RecursiveGaussianImageFilter->SetDirection( 0 );
  RecursiveGaussianImageFilter->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilter->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilter->SetInput(inputfileReader->GetOutput());
  RecursiveGaussianImageFilter->SetSigma(sqrt(2.0*material_diffusivity*diffusion_time));
//  RecursiveGaussianImageFilter->SetSigma(gaussian_sigma);
  RecursiveGaussianImageFilter->Update();


  GaussianFilterType::Pointer RecursiveGaussianImageFilterY = GaussianFilterType::New();
  RecursiveGaussianImageFilterY->SetDirection( 1 );
  RecursiveGaussianImageFilterY->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterY->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterY->SetInput(RecursiveGaussianImageFilter->GetOutput());
  RecursiveGaussianImageFilterY->SetSigma(sqrt(2.0*material_diffusivity*diffusion_time));
  RecursiveGaussianImageFilterY->Update();

  GaussianFilterType::Pointer RecursiveGaussianImageFilterZ = GaussianFilterType::New();
  RecursiveGaussianImageFilterZ->SetDirection( 2 );
  RecursiveGaussianImageFilterZ->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterZ->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterZ->SetInput(RecursiveGaussianImageFilterY->GetOutput());
  RecursiveGaussianImageFilterZ->SetSigma(sqrt(2.0*material_diffusivity*diffusion_time));
  RecursiveGaussianImageFilterZ->Update();

  writer->SetFileName( mHeatDiffusionFilename );
  writer->SetInput( RecursiveGaussianImageFilterZ->GetOutput() );
  writer->Update(); 

}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Callback at each event
void GateNanoActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateNanoActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::UserPreTrackActionInVoxel(const int /*index*/, const G4Track* track) {

  if(track->GetDefinition()->GetParticleName() == "opticalphoton") { mStepHitType = PostStepHitType; }
  else { mStepHitType = mUserStepHitType; }

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {

  GateDebugMessageInc("Actor", 4, "GateNanoActor -- UserSteppingActionInVoxel - begin" << G4endl);

  const double edep = step->GetPostStepPoint()->GetKineticEnergy()/eV;  // in eV

  const G4String process = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();

  // if no energy is deposited or energy is deposited outside image => do nothing
  if (step->GetPostStepPoint()->GetKineticEnergy() == 0) {
    GateDebugMessage("Actor", 5, "edep == 0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateNanoActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateNanoActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }



  if (mIsNanoAbsorptionImageEnabled) {
    GateDebugMessage("Actor", 2, "GateNanoActor -- UserSteppingActionInVoxel:\tedep = " << G4BestUnit(edep, "Energy") << G4endl);
  }
  
  if (mIsNanoAbsorptionImageEnabled) {
	if ( process == "NanoAbsorption" )  mNanoAbsorptionImage.AddValue(index, edep);
  }


  GateDebugMessageDec("Actor", 4, "GateNanoActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------















