/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \class GateNanoActor
  \author vesna.cuplov@gmail.com
  \brief Class GateNanoActor : This actor produces voxelised images of the heat diffusion in tissue.

                                                                    absorption map        heat diffusion map
         laser                _________                               _________              _________
         optical photons     |         |                             |         |            |         |          
         ~~~~~~~>            | nano    |    GATE                     |         |            |   xx    |
         ~~~~~~~>            | objects |    Simulation Results ==>   |   xx    |     +      |  xxxx   |
         ~~~~~~~>            | in the  |    (voxelised images)       |   xx    |            |  xxxx   |
         ~~~~~~~>            | phantom |                             |         |            |   xx    |
                             |_________|                             |_________|            |_________|


  Parameters of the simulation given by the User in the macro:
	- setDiffusivity: tissue thermal diffusivity in mm2/s
	- setTime: diffusion time in s
	- setBloodPerfusionRate: blood perfusion rate in s-1 for the advection term
	- setBloodDensity: blood density (kg/m3)
	- setBloodHeatCapacity: blood heat capacity kJ/(kg C)
	- setTissueDensity: tissue density (kg/m3)
	- setTissueHeatCapacity: tissue heat capacity kJ/(kg C)

	- OPTIONAL: setSimulationScale to get more fluence.
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
#include "GateApplicationMgr.hh"
#include <sys/time.h>
#include <iostream>
#include <string>

//-----------------------------------------------------------------------------

GateNanoActor::GateNanoActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateNanoActor() -- begin"<<G4endl);

  mCurrentEvent=-1;

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

//-----------------------------------------------------------------------------
void GateNanoActor::setTime(G4double t)
{
  mUserDiffusionTime = t;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setDiffusivity(G4double diffusivity)
{
  mUserMaterialDiffusivity = diffusivity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setBloodPerfusionRate(G4double bloodperfusionrate)
{
  mUserBloodPerfusionRate = bloodperfusionrate;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setBloodDensity(G4double blooddensity)
{
  mUserBloodDensity = blooddensity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setBloodHeatCapacity(G4double bloodheatcapacity)
{
  mUserBloodHeatCapacity = bloodheatcapacity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setTissueDensity(G4double tissuedensity)
{
  mUserTissueDensity = tissuedensity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setTissueHeatCapacity(G4double tissueheatcapacity)
{
  mUserTissueHeatCapacity = tissueheatcapacity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setScale(G4double simuscale)
{
  mUserSimulationScale = simuscale;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNanoActor::setNumberOfTimeFrames(G4int numtimeframe)
{
  mUserNumberOfTimeFrames = numtimeframe;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Constructor
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

  // Output Filenames
  mAbsorptionFilename = G4String(removeExtension(mSaveFilename))+"-AbsorptionMap."+G4String(getExtension(mSaveFilename));
  mFinalAbsorptionFilename = G4String(removeExtension(mSaveFilename))+"-FinalAbsorptionMap."+G4String(getExtension(mSaveFilename));
  mFinalHeatDiffusionFilename = G4String(removeExtension(mSaveFilename))+"-FinalHeatDiffusionMap."+G4String(getExtension(mSaveFilename));

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mAbsorptionImage);

  // Resize and allocate images
    mAbsorptionImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mAbsorptionImage.Allocate();
    mAbsorptionImage.SetFilename(mAbsorptionFilename);

  // Print information
  GateMessage("Actor", 1,
              "\tNanoActor    = '" << GetObjectName() << "'" << G4endl <<
              "\tAbsorptionFilename      = " << mAbsorptionFilename << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateNanoActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateNanoActor::SaveData() {

  mAbsorptionImage.SaveData(mCurrentEvent+1);

}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateNanoActor::ResetData() {

  mAbsorptionImage.Reset();

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);

  GateDebugMessage("Actor", 3, "GateNanoActor -- Begin of Run" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::EndOfRunAction(const G4Run* r)
{  
  GateVActor::EndOfRunAction(r);

//clock_t startTime = clock();

  typedef itk::Image<float, 3>   ImageType;  
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typedef itk::ImageFileWriter<ImageType> WriterType;
  typedef itk::RecursiveGaussianImageFilter<ImageType, ImageType >  GaussianFilterType;
  typedef itk::ImageRegionIterator< ImageType > IteratorType; 
  typedef itk::AddImageFilter< ImageType, ImageType, ImageType > AddFilterType;

  WriterType::Pointer writer = WriterType::New();
  ReaderType::Pointer inputfileReader = ReaderType::New();

  inputfileReader->SetFileName( mAbsorptionFilename );

/////////////////////////////////////////////////////////////////////////////////////////
// Convert nano absorption map from photon deposited energy in eV to temperature       //
/////////////////////////////////////////////////////////////////////////////////////////
//  MULTIPLY IMAGE BY A SCALAR:

  // Retrieve the parameters of the experiment (in seconds)
  G4double timeStart = GateApplicationMgr::GetInstance()->GetTimeStart()/s;
  G4double timeStop  = GateApplicationMgr::GetInstance()->GetTimeStop()/s;
  G4double duration  = timeStop-timeStart;  // Total acquisition duration

  // Retrieve the nanoactor image voxel size (in meter)
//  G4double voxelsizex = GateVImageActor::GetVoxelSize().getX()/m;
//  G4double voxelsizey = GateVImageActor::GetVoxelSize().getY()/m;
//  G4double voxelsizez = GateVImageActor::GetVoxelSize().getZ()/m;

//  std::cout << "nanoactor image voxelsizex = " << voxelsizex << std::endl;
//  std::cout << "nanoactor image voxelsizey = " << voxelsizey << std::endl;
//  std::cout << "nanoactor image voxelsizez = " << voxelsizez << std::endl;
  
// Boost in Temperature for Nanoparticles:
//  deltaT= mUserSimulationScale*mUserNanoDensity*pow(voxelsizex/2,2)*mUserNanoAbsorptionCS*(1.6E-19/(duration*voxelsizex*voxelsizey))/(2*mUserTissueThermalConductivity);
  deltaT= 1;

  typedef itk::MultiplyImageFilter< ImageType, ImageType, ImageType > FilterType;
  FilterType::Pointer multiplyFilter = FilterType::New();
  multiplyFilter->SetInput( inputfileReader->GetOutput() );
  multiplyFilter->SetConstant( deltaT );
  multiplyFilter->Update();


/////////////////////////////////////////////////////////////////////////////////////////
// DYNAMIC PROCESS - HEAT DIFFUSES DURING IRRADIATION (DURING ABSORPTION OF PHOTONS)   //
/////////////////////////////////////////////////////////////////////////////////////////

// From the photon absorption map of total DAQ, extract a sample of 1s:

	ImageType::Pointer ImageAbsorptionSample = ImageType::New();
	ImageAbsorptionSample = multiplyFilter->GetOutput();

	float pixValSample=0;
	float newValSample=0;

	IteratorType itSample( ImageAbsorptionSample, ImageAbsorptionSample->GetRequestedRegion() );

	for (itSample.GoToBegin(); !itSample.IsAtEnd(); ++itSample)
	{
		pixValSample=itSample.Get();
		newValSample= pixValSample/mUserNumberOfTimeFrames;
		itSample.Set( newValSample );
	}

	ImageAbsorptionSample->Update();  // Sample of the nano absorption map


////////////////////////////////////////////////////////////
// HEAT DIFFUSION APPLIED ON THE ABSORPTION SAMPLE        //
////////////////////////////////////////////////////////////

// PART 1: Create diffusion images for the "ImageAbsorptionSample" for each cummulative time interval: 1s, 2s, .....

if ( mUserNumberOfTimeFrames > 1 ) {

  ImageType::Pointer ImageConductionSample[mUserNumberOfTimeFrames];

for(int i=1; i!=mUserNumberOfTimeFrames; ++i) {
  GaussianFilterType::Pointer RecursiveGaussianImageFilterX = GaussianFilterType::New();
  RecursiveGaussianImageFilterX->SetDirection( 0 );
  RecursiveGaussianImageFilterX->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterX->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterX->SetInput(ImageAbsorptionSample);
  RecursiveGaussianImageFilterX->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*(i*duration/mUserNumberOfTimeFrames)));
  RecursiveGaussianImageFilterX->Update();

  GaussianFilterType::Pointer RecursiveGaussianImageFilterY = GaussianFilterType::New();
  RecursiveGaussianImageFilterY->SetDirection( 1 );
  RecursiveGaussianImageFilterY->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterY->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterY->SetInput(RecursiveGaussianImageFilterX->GetOutput());
  RecursiveGaussianImageFilterY->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*(i*duration/mUserNumberOfTimeFrames)));
  RecursiveGaussianImageFilterY->Update();

  GaussianFilterType::Pointer RecursiveGaussianImageFilterZ = GaussianFilterType::New();
  RecursiveGaussianImageFilterZ->SetDirection( 2 );
  RecursiveGaussianImageFilterZ->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterZ->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterZ->SetInput(RecursiveGaussianImageFilterY->GetOutput());
  RecursiveGaussianImageFilterZ->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*(i*duration/mUserNumberOfTimeFrames)));
  RecursiveGaussianImageFilterZ->Update();

  ImageConductionSample[i] = RecursiveGaussianImageFilterZ->GetOutput();
  ImageConductionSample[i]->DisconnectPipeline();

}

// PART 2:  Add the blood perfusion term in the solution of the diffusion equation:

  ImageType::Pointer ImageConductionAdvectionSample[mUserNumberOfTimeFrames];

for(int i=1; i!=mUserNumberOfTimeFrames; ++i) {

 ImageConductionAdvectionSample[i] = ImageConductionSample[i];
 ImageConductionAdvectionSample[i]->DisconnectPipeline();

	float pixVal=0;
	float newVal=0;
	IteratorType it( ImageConductionAdvectionSample[i], ImageConductionAdvectionSample[i]->GetRequestedRegion() );
	for (it.GoToBegin(); !it.IsAtEnd(); ++it)
	{
		pixVal=it.Get();
	        newVal= pixVal*std::exp(-(mUserBloodDensity*mUserBloodHeatCapacity)/(mUserTissueDensity*mUserTissueHeatCapacity)*mUserBloodPerfusionRate*(i));
		it.Set( newVal );
	}
	ImageConductionAdvectionSample[i]->Update();
}


//////////////////////////////////////////////////////////////////////////////
// ADD ALL SAMPLE DIFFUSED IMAGES TO CREATE THE FINAL ABSORPTION MAP        //
//////////////////////////////////////////////////////////////////////////////


  AddFilterType::Pointer addFilter = AddFilterType::New();
  ImageType::Pointer array[mUserNumberOfTimeFrames];

  ImageType::Pointer ImageTemp = ImageType::New();
  ImageTemp = ImageConductionAdvectionSample[1];


  for(int i=2; i!=mUserNumberOfTimeFrames; ++i) {
  array[i] = ImageConductionAdvectionSample[i];
  array[i]->DisconnectPipeline();

  addFilter->SetInput1( ImageTemp );
  addFilter->SetInput2( array[i] );
  addFilter->Update();
  ImageTemp = addFilter->GetOutput();
}

// FINAL Absorption image:
  AddFilterType::Pointer addFilterFinal = AddFilterType::New();
  addFilterFinal->SetInput1( ImageTemp );
  addFilterFinal->SetInput2( ImageAbsorptionSample );
  addFilterFinal->Update();

  writer->SetFileName( mFinalAbsorptionFilename );
  writer->SetInput( addFilterFinal->GetOutput() );
  writer->Update(); 




//////////////////////////////////////////////////////////////////////////////
// APPLY HEAT DIFFUSION ON THE FINAL ABSORPTION MAP IMAGE                   //
//////////////////////////////////////////////////////////////////////////////

// conduction

  GaussianFilterType::Pointer RecursiveGaussianImageFilterX = GaussianFilterType::New();
  RecursiveGaussianImageFilterX->SetDirection( 0 );
  RecursiveGaussianImageFilterX->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterX->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterX->SetInput(addFilterFinal->GetOutput());
  RecursiveGaussianImageFilterX->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  RecursiveGaussianImageFilterX->Update();

  GaussianFilterType::Pointer RecursiveGaussianImageFilterY = GaussianFilterType::New();
  RecursiveGaussianImageFilterY->SetDirection( 1 );
  RecursiveGaussianImageFilterY->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterY->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterY->SetInput(RecursiveGaussianImageFilterX->GetOutput());
  RecursiveGaussianImageFilterY->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  RecursiveGaussianImageFilterY->Update();

  GaussianFilterType::Pointer RecursiveGaussianImageFilterZ = GaussianFilterType::New();
  RecursiveGaussianImageFilterZ->SetDirection( 2 );
  RecursiveGaussianImageFilterZ->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterZ->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterZ->SetInput(RecursiveGaussianImageFilterY->GetOutput());
  RecursiveGaussianImageFilterZ->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  RecursiveGaussianImageFilterZ->Update();


// blood perfusion

	ImageType::Pointer ImageConductionAdvection = ImageType::New();
	ImageConductionAdvection = RecursiveGaussianImageFilterZ->GetOutput();

	float pixVal2=0;
	float newVal2=0;
	IteratorType it2( ImageConductionAdvection, ImageConductionAdvection->GetRequestedRegion() );
	for (it2.GoToBegin(); !it2.IsAtEnd(); ++it2)
	{
		pixVal2=it2.Get();
	        newVal2= pixVal2*std::exp(-(mUserBloodDensity*mUserBloodHeatCapacity)/(mUserTissueDensity*mUserTissueHeatCapacity)*mUserBloodPerfusionRate*mUserDiffusionTime);
		it2.Set( newVal2 );
	}
	ImageConductionAdvection->Update();

  writer->SetFileName( mFinalHeatDiffusionFilename );
  writer->SetInput( ImageConductionAdvection ); // heat diffusion
  writer->Update(); 

}

else {

//////////////////////////////////////////////////////////////////////////////
// APPLY HEAT DIFFUSION ON THE FINAL ABSORPTION MAP IMAGE                   //
//////////////////////////////////////////////////////////////////////////////

// conduction

  GaussianFilterType::Pointer RecursiveGaussianImageFilterX = GaussianFilterType::New();
  RecursiveGaussianImageFilterX->SetDirection( 0 );
  RecursiveGaussianImageFilterX->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterX->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterX->SetInput(multiplyFilter->GetOutput());
  RecursiveGaussianImageFilterX->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  RecursiveGaussianImageFilterX->Update();

  GaussianFilterType::Pointer RecursiveGaussianImageFilterY = GaussianFilterType::New();
  RecursiveGaussianImageFilterY->SetDirection( 1 );
  RecursiveGaussianImageFilterY->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterY->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterY->SetInput(RecursiveGaussianImageFilterX->GetOutput());
  RecursiveGaussianImageFilterY->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  RecursiveGaussianImageFilterY->Update();

  GaussianFilterType::Pointer RecursiveGaussianImageFilterZ = GaussianFilterType::New();
  RecursiveGaussianImageFilterZ->SetDirection( 2 );
  RecursiveGaussianImageFilterZ->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterZ->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterZ->SetInput(RecursiveGaussianImageFilterY->GetOutput());
  RecursiveGaussianImageFilterZ->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  RecursiveGaussianImageFilterZ->Update();


// blood perfusion

	ImageType::Pointer ImageConductionAdvection = ImageType::New();
	ImageConductionAdvection = RecursiveGaussianImageFilterZ->GetOutput();

	float pixVal2=0;
	float newVal2=0;
	IteratorType it2( ImageConductionAdvection, ImageConductionAdvection->GetRequestedRegion() );
	for (it2.GoToBegin(); !it2.IsAtEnd(); ++it2)
	{
		pixVal2=it2.Get();
	        newVal2= pixVal2*std::exp(-(mUserBloodDensity*mUserBloodHeatCapacity)/(mUserTissueDensity*mUserTissueHeatCapacity)*mUserBloodPerfusionRate*mUserDiffusionTime);
		it2.Set( newVal2 );
	}
	ImageConductionAdvection->Update();

  writer->SetFileName( mFinalHeatDiffusionFilename );
  writer->SetInput( ImageConductionAdvection ); // heat diffusion
  writer->Update(); 
}

//std::cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC << " seconds." << std::endl;


}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
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


    GateDebugMessage("Actor", 2, "GateNanoActor -- UserSteppingActionInVoxel:\tedep = " << G4BestUnit(edep, "Energy") << G4endl);

   if ( process == "NanoAbsorption" )  mAbsorptionImage.AddValue(index, edep);

  GateDebugMessageDec("Actor", 4, "GateNanoActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------






