/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

// This actor is only compiled if ITK is available
#include "GateConfiguration.h"
#ifdef  GATE_USE_ITK

/*
  \class GateThermalActor
  \author vesna.cuplov@gmail.com
  \brief Class GateThermalActor : This actor produces voxelised images of the heat diffusion in tissue.

                                                                    absorption map        heat diffusion map
         laser                _________                               _________              _________
         optical photons     |         |                             |         |            |         |
         ~~~~~~~>            |         |    GATE                     |         |            |   xx    |
         ~~~~~~~>            | phantom |    Simulation Results ==>   |   xx    |     +      |  xxxx   |
         ~~~~~~~>            |         |    (voxelised images)       |   xx    |            |  xxxx   |
         ~~~~~~~>            |         |                             |         |            |   xx    |
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

#include "GateThermalActor.hh"
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

GateThermalActor::GateThermalActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateThermalActor() -- begin"<<G4endl);

  mCurrentEvent=-1;

  pMessenger = new GateThermalActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateThermalActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Destructor
GateThermalActor::~GateThermalActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::setTime(G4double t)
{
  mUserDiffusionTime = t;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::setDiffusivity(G4double diffusivity)
{
  mUserMaterialDiffusivity = diffusivity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::setBloodPerfusionRate(G4double bloodperfusionrate)
{
  mUserBloodPerfusionRate = bloodperfusionrate;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::setBloodDensity(G4double blooddensity)
{
  mUserBloodDensity = blooddensity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::setBloodHeatCapacity(G4double bloodheatcapacity)
{
  mUserBloodHeatCapacity = bloodheatcapacity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::setTissueDensity(G4double tissuedensity)
{
  mUserTissueDensity = tissuedensity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::setTissueHeatCapacity(G4double tissueheatcapacity)
{
  mUserTissueHeatCapacity = tissueheatcapacity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::setScale(G4double simuscale)
{
  mUserSimulationScale = simuscale;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateThermalActor::setNumberOfTimeFrames(G4int numtimeframe)
{
  mUserNumberOfTimeFrames = numtimeframe;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Constructor
void GateThermalActor::Construct() {

  GateDebugMessageInc("Actor", 4, "GateThermalActor -- Construct - begin" << G4endl);

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
  mHeatDiffusionFilename = G4String(removeExtension(mSaveFilename))+"-HeatDiffusionMap."+G4String(getExtension(mSaveFilename));

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mAbsorptionImage);

  // Resize and allocate images
  mAbsorptionImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mAbsorptionImage.Allocate();
  mAbsorptionImage.SetFilename(mAbsorptionFilename);

  // Print information
  GateMessage("Actor", 1,
              "\tThermalActor    = '" << GetObjectName() << "'" << G4endl <<
              "\tAbsorptionFilename      = " << mAbsorptionFilename << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateThermalActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateThermalActor::SaveData() {

  mAbsorptionImage.SaveData(mCurrentEvent+1);

}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateThermalActor::ResetData() {

  mAbsorptionImage.Reset();

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);

  GateDebugMessage("Actor", 3, "GateThermalActor -- Begin of Run" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::EndOfRunAction(const G4Run* r)
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
  //  deltaT= 1;
  deltaT= mUserSimulationScale;

  //  std::cout << "Simulation Scale = " << deltaT << std::endl;

  typedef itk::MultiplyImageFilter< ImageType, ImageType, ImageType > FilterType;
  FilterType::Pointer multiplyFilter = FilterType::New();
  multiplyFilter->SetInput( inputfileReader->GetOutput() );
  multiplyFilter->SetConstant( deltaT );
  multiplyFilter->Update();


  /////////////////////////////////////////////////////////////////////////////////////////
  // DYNAMIC PROCESS - HEAT DIFFUSES DURING IRRADIATION (DURING ABSORPTION OF PHOTONS)   //
  /////////////////////////////////////////////////////////////////////////////////////////

  // From the photon absorption map of total DAQ, extract a sample:

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

	ImageAbsorptionSample->Update();  // Sample of the absorption map

  //  writer->SetFileName( G4String(removeExtension(mSaveFilename)) +"-AbsorptionSample." + G4String(getExtension(mSaveFilename)) );
  //  writer->SetInput( ImageAbsorptionSample );
  //  writer->Update();


  ////////////////////////////////////////////////////////////
  // HEAT DIFFUSION APPLIED ON THE ABSORPTION SAMPLE        //
  ////////////////////////////////////////////////////////////

  // PART 1: Create diffusion images for the "ImageAbsorptionSample" for all time frames:

  if ( mUserNumberOfTimeFrames > 1 ) {

    ImageType::Pointer ImageConductionSample[mUserNumberOfTimeFrames-1];

    for(int i=0; i!=mUserNumberOfTimeFrames-1; ++i) {
      GaussianFilterType::Pointer RecursiveGaussianImageFilterX = GaussianFilterType::New();
      RecursiveGaussianImageFilterX->SetDirection( 0 );
      RecursiveGaussianImageFilterX->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
      RecursiveGaussianImageFilterX->SetNormalizeAcrossScale( false );
      RecursiveGaussianImageFilterX->SetInput(ImageAbsorptionSample);
      RecursiveGaussianImageFilterX->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*((i+1)*duration/mUserNumberOfTimeFrames)));
      RecursiveGaussianImageFilterX->Update();

      GaussianFilterType::Pointer RecursiveGaussianImageFilterY = GaussianFilterType::New();
      RecursiveGaussianImageFilterY->SetDirection( 1 );
      RecursiveGaussianImageFilterY->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
      RecursiveGaussianImageFilterY->SetNormalizeAcrossScale( false );
      RecursiveGaussianImageFilterY->SetInput(RecursiveGaussianImageFilterX->GetOutput());
      RecursiveGaussianImageFilterY->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*((i+1)*duration/mUserNumberOfTimeFrames)));
      RecursiveGaussianImageFilterY->Update();

      GaussianFilterType::Pointer RecursiveGaussianImageFilterZ = GaussianFilterType::New();
      RecursiveGaussianImageFilterZ->SetDirection( 2 );
      RecursiveGaussianImageFilterZ->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
      RecursiveGaussianImageFilterZ->SetNormalizeAcrossScale( false );
      RecursiveGaussianImageFilterZ->SetInput(RecursiveGaussianImageFilterY->GetOutput());
      RecursiveGaussianImageFilterZ->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*((i+1)*duration/mUserNumberOfTimeFrames)));
      RecursiveGaussianImageFilterZ->Update();

      ImageConductionSample[i] = RecursiveGaussianImageFilterZ->GetOutput();
      ImageConductionSample[i]->DisconnectPipeline();

      // writer for validation:   ///////////////////
      //        std::ostringstream temp;
      //        temp << i+1;
      //  writer->SetFileName( G4String(removeExtension(mSaveFilename)) +"-ConductionSample_"+ temp.str() +"s."+ G4String(getExtension(mSaveFilename)) );
      //  writer->SetInput( ImageConductionSample[i] );
      //  writer->Update();
      ////////////////////////////////////////////////////////

    }


    // PART 2:  Add the blood perfusion term in the solution of the diffusion equation

    ImageType::Pointer ImageConductionAdvectionSample[mUserNumberOfTimeFrames-1];

    for(int i=0; i!=mUserNumberOfTimeFrames-1; ++i) {
      ImageConductionAdvectionSample[i] = ImageConductionSample[i];
      ImageConductionAdvectionSample[i]->DisconnectPipeline();

      float pixVal=0;
      float newVal=0;
      IteratorType it( ImageConductionAdvectionSample[i], ImageConductionAdvectionSample[i]->GetRequestedRegion() );
      for (it.GoToBegin(); !it.IsAtEnd(); ++it)
        {
          pixVal=it.Get();
	        newVal= pixVal*std::exp(-(mUserBloodDensity*mUserBloodHeatCapacity)/(mUserTissueDensity*mUserTissueHeatCapacity)*mUserBloodPerfusionRate*((i+1)*duration/mUserNumberOfTimeFrames));
          it.Set( newVal );

        }
      ImageConductionAdvectionSample[i]->Update();

      // writer for validation:   ///////////////////
      //        std::ostringstream temp;
      //        temp << i+1;
      //  writer->SetFileName( G4String(removeExtension(mSaveFilename)) +"-SampleDiffusion_"+ temp.str() +"s."+ G4String(getExtension(mSaveFilename)) );
      //  writer->SetInput( ImageConductionAdvectionSample[i] );
      //  writer->Update();
      ////////////////////////////////////////////////////////////////

    }


    //////////////////////////////////////////////////////////////////////////////
    // ADD ALL SAMPLE DIFFUSED IMAGES TO CREATE THE FINAL ABSORPTION MAP        //
    //////////////////////////////////////////////////////////////////////////////


    AddFilterType::Pointer addFilter = AddFilterType::New();
    ImageType::Pointer array[mUserNumberOfTimeFrames];

    ImageType::Pointer ImageTemp = ImageType::New();
    ImageTemp = ImageConductionAdvectionSample[0];

    for(int i=1; i!=mUserNumberOfTimeFrames-1; ++i) {
      array[i] = ImageConductionAdvectionSample[i];
      array[i]->DisconnectPipeline();
      addFilter->SetInput1( ImageTemp );
      addFilter->SetInput2( array[i] );
      addFilter->Update();
      ImageTemp = addFilter->GetOutput();
    }

    //////////////////////////////////////////////////////////////////////////////
    //                       FINAL ABSORPTION MAP IMAGE                         //
    //////////////////////////////////////////////////////////////////////////////
    AddFilterType::Pointer addFilterFinal = AddFilterType::New();
    addFilterFinal->SetInput1( ImageTemp );
    addFilterFinal->SetInput2( ImageAbsorptionSample );
    addFilterFinal->Update();

    writer->SetFileName( mAbsorptionFilename );
    writer->SetInput( addFilterFinal->GetOutput() );
    writer->Update();




    //////////////////////////////////////////////////////////////////////////////
    // APPLY HEAT DIFFUSION ON THE FINAL ABSORPTION MAP IMAGE                   //
    //////////////////////////////////////////////////////////////////////////////

    // conduction

    GaussianFilterType::Pointer RecursiveGaussianImageFilterX = GaussianFilterType::New();
    RecursiveGaussianImageFilterX->SetDirection( 0 );
    RecursiveGaussianImageFilterX->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
    RecursiveGaussianImageFilterX->SetNormalizeAcrossScale( false );
    RecursiveGaussianImageFilterX->SetInput(addFilterFinal->GetOutput());
    RecursiveGaussianImageFilterX->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
    RecursiveGaussianImageFilterX->Update();

    GaussianFilterType::Pointer RecursiveGaussianImageFilterY = GaussianFilterType::New();
    RecursiveGaussianImageFilterY->SetDirection( 1 );
    RecursiveGaussianImageFilterY->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
    RecursiveGaussianImageFilterY->SetNormalizeAcrossScale( false );
    RecursiveGaussianImageFilterY->SetInput(RecursiveGaussianImageFilterX->GetOutput());
    RecursiveGaussianImageFilterY->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
    RecursiveGaussianImageFilterY->Update();

    GaussianFilterType::Pointer RecursiveGaussianImageFilterZ = GaussianFilterType::New();
    RecursiveGaussianImageFilterZ->SetDirection( 2 );
    RecursiveGaussianImageFilterZ->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
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

    writer->SetFileName( mHeatDiffusionFilename );
    writer->SetInput( ImageConductionAdvection ); // heat diffusion
    writer->Update();

  }

  else {


    //////////////////////////////////////////////////////////////////////////////
    //                       FINAL ABSORPTION MAP IMAGE                         //
    //////////////////////////////////////////////////////////////////////////////

    writer->SetFileName( mAbsorptionFilename );
    writer->SetInput( multiplyFilter->GetOutput() );
    writer->Update();


    //////////////////////////////////////////////////////////////////////////////
    // APPLY HEAT DIFFUSION ON THE FINAL ABSORPTION MAP IMAGE                   //
    //////////////////////////////////////////////////////////////////////////////

    // conduction

    GaussianFilterType::Pointer RecursiveGaussianImageFilterX = GaussianFilterType::New();
    RecursiveGaussianImageFilterX->SetDirection( 0 );
    RecursiveGaussianImageFilterX->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
    RecursiveGaussianImageFilterX->SetNormalizeAcrossScale( false );
    RecursiveGaussianImageFilterX->SetInput(multiplyFilter->GetOutput());
    RecursiveGaussianImageFilterX->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
    RecursiveGaussianImageFilterX->Update();

    GaussianFilterType::Pointer RecursiveGaussianImageFilterY = GaussianFilterType::New();
    RecursiveGaussianImageFilterY->SetDirection( 1 );
    RecursiveGaussianImageFilterY->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
    RecursiveGaussianImageFilterY->SetNormalizeAcrossScale( false );
    RecursiveGaussianImageFilterY->SetInput(RecursiveGaussianImageFilterX->GetOutput());
    RecursiveGaussianImageFilterY->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
    RecursiveGaussianImageFilterY->Update();

    GaussianFilterType::Pointer RecursiveGaussianImageFilterZ = GaussianFilterType::New();
    RecursiveGaussianImageFilterZ->SetDirection( 2 );
    RecursiveGaussianImageFilterZ->SetOrder( itk::GaussianOrderEnum::ZeroOrder );
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

    writer->SetFileName( mHeatDiffusionFilename );
    writer->SetInput( ImageConductionAdvection ); // heat diffusion
    writer->Update();
  }

  //std::cout << double( clock() - startTime ) / (double)CLOCKS_PER_SEC << " seconds." << std::endl;


}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateThermalActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);

  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateThermalActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::UserPreTrackActionInVoxel(const int /*index*/, const G4Track* track) {

  if(track->GetDefinition()->GetParticleName() == "opticalphoton") { mStepHitType = PostStepHitType; }
  else { mStepHitType = mUserStepHitType; }

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateThermalActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {

  GateDebugMessageInc("Actor", 4, "GateThermalActor -- UserSteppingActionInVoxel - begin" << G4endl);

  const double edep = step->GetPostStepPoint()->GetKineticEnergy()/eV;  // in eV

  const G4String process = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();

  // if no energy is deposited or energy is deposited outside image => do nothing
  if (step->GetPostStepPoint()->GetKineticEnergy() == 0) {
    GateDebugMessage("Actor", 5, "edep == 0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateThermalActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateThermalActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }


  GateDebugMessage("Actor", 2, "GateThermalActor -- UserSteppingActionInVoxel:\tedep = " << G4BestUnit(edep, "Energy") << G4endl);

  if ( process == "NanoAbsorption" || process == "OpticalAbsorption" )  mAbsorptionImage.AddValue(index, edep);

  GateDebugMessageDec("Actor", 4, "GateThermalActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------

#endif // end define USE_ITK
