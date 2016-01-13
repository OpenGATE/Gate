/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \class GateNanoActor
  \author vesna.cuplov@gmail.com
  \brief Class GateNanoActor : This actor produces a voxelised image of the energy deposited by optical
                               photons in the nano material (physics process NanoAbsorption): absorption map.
                               This absorption map corresponds to an initial condition (heat at t=0). The heat 
                               is then diffused and a second voxelised image is obtained as the solution of the 
                               heat equation at a later time.

                                                                    absorption map        heat diffusion map
         laser                _________                               _________              _________
         optical photons     |         |                             |         |            |         |          
         ~~~~~~~>            | nano    |    GATE                     |         |            |   xx    |
         ~~~~~~~>            | objects |    Simulation Results ==>   |   xx    |     +      |  xxxx   |
         ~~~~~~~>            | in the  |    (voxelised images)       |   xx    |            |  xxxx   |
         ~~~~~~~>            | phantom |                             |         |            |   xx    |
                             |_________|                             |_________|            |_________|


  Parameters of the simulation given by the User in the macro:
	- setNanoAbsorptionCrossSection: nano material absorption (extinction) cross-section in m2
	- setNanoDensity: nano objects density (number of nano objects per m3)
	- setDiffusivity: tissu thermal diffusivity in mm2/s
	- setTime: diffusion time in s
	- setBodyTemperature: body temperature
	- setBloodTemperature: blood temperature
	- setBloodPerfusionRate: blood perfusion rate in s-1 for the advection term
	- setBloodDensity: blood density (kg/m3)
	- setBloodHeatCapacity: blood heat capacity kJ/(kg C)
	- setTissueDensity: tissue density (kg/m3)
	- setTissueHeatCapacity: tissue heat capacity kJ/(kg C)
	- setTissueThermalConductivity: tissue thermal conductivity in W/(mxK)

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
void GateNanoActor::setBodyTemperature(G4double bodytemperature)
{
  mUserBodyTemperature = bodytemperature;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setBloodTemperature(G4double bloodtemperature)
{
  mUserBloodTemperature = bloodtemperature;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setNanoMaximumTemperature(G4double nanotemperature)
{
  mUserNanoTemperature = nanotemperature;
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
void GateNanoActor::setTissueThermalConductivity(G4double tissuethermalconductivity)
{
  mUserTissueThermalConductivity = tissuethermalconductivity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setNanoAbsorptionCrossSection(G4double nanoabsorptionCS)
{
  mUserNanoAbsorptionCS = nanoabsorptionCS;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setNanoDensity(G4double nanodensity)
{
  mUserNanoDensity = nanodensity;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::setScale(G4double simuscale)
{
  mUserSimulationScale = simuscale;
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
  mNanoAbsorptionFilename = G4String(removeExtension(mSaveFilename))+"-AbsorptionMap."+G4String(getExtension(mSaveFilename));
  mNanoAbsorptioninTemperatureFilename = G4String(removeExtension(mSaveFilename))+"-AbsorptioninTemperatureMap."+G4String(getExtension(mSaveFilename));
  mHeatConductionFilename = G4String(removeExtension(mSaveFilename))+"-HeatConductionMap."+G4String(getExtension(mSaveFilename));
  mHeatConductionAdvectionFilename = G4String(removeExtension(mSaveFilename))+"-HeatConductionAdvectionMap."+G4String(getExtension(mSaveFilename));

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mNanoAbsorptionImage);

  // Resize and allocate images
    mNanoAbsorptionImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mNanoAbsorptionImage.Allocate();
    mNanoAbsorptionImage.SetFilename(mNanoAbsorptionFilename);

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
//  GateVActor::SaveData();

  mNanoAbsorptionImage.SaveData(mCurrentEvent+1);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateNanoActor::ResetData() {

  mNanoAbsorptionImage.Reset();
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

  typedef itk::Image<float, 3>   ImageType;  
  typedef itk::ImageFileReader<ImageType> ReaderType;
  typedef itk::ImageFileWriter<ImageType> WriterType;
  typedef itk::RecursiveGaussianImageFilter<ImageType, ImageType >  GaussianFilterType;
  typedef itk::ImageRegionIterator< ImageType > IteratorType; 

  WriterType::Pointer writer = WriterType::New();
  ReaderType::Pointer inputfileReader = ReaderType::New();

  inputfileReader->SetFileName( mNanoAbsorptionFilename );

/////////////////////////////////////////////////////////////////////////////////////////
// Convert nano absorption map from photon deposited energy in eV to temperature       //
/////////////////////////////////////////////////////////////////////////////////////////

//  Simple conversion eV in temperature: not used.
//  Rescale image intensity between a min temperature (body temperature) 
//  and a max temperature (from litterature, depends on the nano object):
//  typedef itk::RescaleIntensityImageFilter< ImageType, ImageType > RescaleFilterType;
//  RescaleFilterType::Pointer rescaleFilter = RescaleFilterType::New();
//  rescaleFilter->SetInput(inputfileReader->GetOutput());
//  rescaleFilter->SetOutputMinimum(mUserBodyTemperature); 
//  rescaleFilter->SetOutputMaximum(mUserNanoTemperature);
//  rescaleFilter->Update();
//  writer->SetFileName( mNanoAbsorptioninTemperatureFilename );
//  writer->SetInput( rescaleFilter->GetOutput() );
//  writer->Update(); 
/////////////////////////////////////////////////////////////////////////////////////////

//  More realistic conversion eV in temperature:
//  MULTIPLY IMAGE BY A SCALAR:

  // Retrieve the parameters of the experiment (in seconds)
  G4double timeStart = GateApplicationMgr::GetInstance()->GetTimeStart()/s;
  G4double timeStop  = GateApplicationMgr::GetInstance()->GetTimeStop()/s;
//  G4double timeStep  = GateApplicationMgr::GetInstance()->GetTimeSlice()/s;
  G4double duration  = timeStop-timeStart;  // Total acquisition duration

//  std::cout << "simulation duration = " << duration << std::endl;

  // Retrieve the nanoactor image voxel size (in meter)
  G4double voxelsizex = GateVImageActor::GetVoxelSize().getX()/m;
  G4double voxelsizey = GateVImageActor::GetVoxelSize().getY()/m;
  G4double voxelsizez = GateVImageActor::GetVoxelSize().getZ()/m;

//  std::cout << "nanoactor image voxelsizex = " << voxelsizex << std::endl;
//  std::cout << "nanoactor image voxelsizey = " << voxelsizey << std::endl;
//  std::cout << "nanoactor image voxelsizez = " << voxelsizez << std::endl;
  
// Heat per voxel generated by a concentration of nano objects:
// Ref: "Thermophysical and biological responses of gold nanoparticle laser heating", Z. Qin and J. C. Bischof
// Chem. Soc. Rev. 41 (2012) 
// delta T per voxel = N x Rvoxel^2 x Qnano/2k
// N = density = number of nano objects per m3
// Rvoxel = voxel radius
// k = thermal conductivity
// Qnano = heat generated by the nano objects in a voxel = Cabs x I = nano object absorption cross-section (m2) x laser fluence per voxel (W/m2)
// The absorption map image is given in eV which is the energy deposited by optical photons. Therefore, the fluence per voxel is given by:
// (#eV in a voxel) x 1.6E-19 / (DAQ(s) x voxel area in m2) ==> unit of J/(sxm2) = W/m2

// Simulation Scale:
// Typically, if during a 500s DAQ simulation, 20eV is deposited in a voxel of (0.25)^3 mm^3, the fluence in that voxel is: 9E-14 W/m2
// In the litterature, fluences are of order of 1E4-1E5 W/m2
// To speed up the simulation, we use a Simulation Scale (mUserSimulationScale)
// In the simulation macro: nano_density is in 1/m3, nano_absorptionCS is in m2 and thermal conductivity is in W/(mxK)

  deltaT= mUserSimulationScale*mUserNanoDensity*pow(voxelsizex/2,2)*mUserNanoAbsorptionCS*(1.6E-19/(duration*voxelsizex*voxelsizey))/(2*mUserTissueThermalConductivity);

//  std::cout << "nano object absorption cross-section = " << mUserNanoAbsorptionCS << std::endl;
//  std::cout << "nano material density = " << mUserNanoDensity << std::endl;
//  std::cout << "fluence per voxel = " << 1.6E-19/(duration*voxelsizex*voxelsizey) << std::endl;
//  std::cout << "deltaT per voxel  = " << deltaT << std::endl;

  typedef itk::MultiplyImageFilter< ImageType, ImageType, ImageType > FilterType;
  FilterType::Pointer multiplyFilter = FilterType::New();
  multiplyFilter->SetInput( inputfileReader->GetOutput() );
  multiplyFilter->SetConstant( deltaT );
  multiplyFilter->Update();

// Add body temperature to the deltaT image

	ImageType::Pointer ImageAbsorptioninT = ImageType::New();
	ImageAbsorptioninT = multiplyFilter->GetOutput();

	float pixValT=0;
	float newValT=0;
	IteratorType itT( ImageAbsorptioninT, ImageAbsorptioninT->GetRequestedRegion() );

	for (itT.GoToBegin(); !itT.IsAtEnd(); ++itT)
	{
		pixValT=itT.Get();
		newValT= pixValT+mUserBodyTemperature;
		itT.Set( newValT );
	}

	ImageAbsorptioninT->Update();  // nano absorption map in temperature

  writer->SetFileName( mNanoAbsorptioninTemperatureFilename );
  writer->SetInput( ImageAbsorptioninT );
  writer->Update(); 
 

//////////////////////////////////////////
// Heat diffusion - Conduction only     //
//////////////////////////////////////////

  GaussianFilterType::Pointer RecursiveGaussianImageFilterX = GaussianFilterType::New();
  RecursiveGaussianImageFilterX->SetDirection( 0 );
  RecursiveGaussianImageFilterX->SetOrder( GaussianFilterType::ZeroOrder );
  RecursiveGaussianImageFilterX->SetNormalizeAcrossScale( false );
  RecursiveGaussianImageFilterX->SetInput(ImageAbsorptioninT);
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

  writer->SetFileName( mHeatConductionFilename );
  writer->SetInput( RecursiveGaussianImageFilterZ->GetOutput() ); // heat diffusion (conduction only) map in temperature
  writer->Update(); 


///////////////////////////////////////////////////
// Heat diffusion - Conduction and Advection     //
///////////////////////////////////////////////////

// The solution of the conduction+advection heat equation is:
// T(t) = [T(t=0)-Tblood] (convolved with) 1/(4pi*thermaldiffusivity*diffusiontime)^(3/2) exp[-(x2+y2+z2)/4*thermaldiffusivity*diffusiontime]
//         * exp(-bloodperfusionterm*diffusiontime) + Tblood
// T(t=0): initial conditions = heat at t=0 ==> this is the absorption map in temperature.


// First: Create the [T(t=0)-Tblood] image:

	ImageType::Pointer Image1 = ImageType::New();
	Image1 = ImageAbsorptioninT;

	float pixVal1=0;
	float newVal1=0;
	IteratorType it1( Image1, Image1->GetRequestedRegion() );

	for (it1.GoToBegin(); !it1.IsAtEnd(); ++it1)
	{
		pixVal1=it1.Get();
		newVal1= pixVal1-mUserBloodTemperature;
		it1.Set( newVal1 );
	}

	Image1->Update();  // T(t=0)-Tblood image


// Second: Perform the convolution of the T(t=0)-Tblood image with the gaussian:

  GaussianFilterType::Pointer ConvolveImage1X = GaussianFilterType::New();
  ConvolveImage1X->SetDirection( 0 );
  ConvolveImage1X->SetOrder( GaussianFilterType::ZeroOrder );
  ConvolveImage1X->SetNormalizeAcrossScale( false );
  ConvolveImage1X->SetInput(Image1);
  ConvolveImage1X->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  ConvolveImage1X->Update();

  GaussianFilterType::Pointer ConvolveImage1Y = GaussianFilterType::New();
  ConvolveImage1Y->SetDirection( 1 );
  ConvolveImage1Y->SetOrder( GaussianFilterType::ZeroOrder );
  ConvolveImage1Y->SetNormalizeAcrossScale( false );
  ConvolveImage1Y->SetInput(ConvolveImage1X->GetOutput());
  ConvolveImage1Y->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  ConvolveImage1Y->Update();

  GaussianFilterType::Pointer ConvolveImage1Z = GaussianFilterType::New();
  ConvolveImage1Z->SetDirection( 2 );
  ConvolveImage1Z->SetOrder( GaussianFilterType::ZeroOrder );
  ConvolveImage1Z->SetNormalizeAcrossScale( false );
  ConvolveImage1Z->SetInput(ConvolveImage1Y->GetOutput());
  ConvolveImage1Z->SetSigma(sqrt(2.0*mUserMaterialDiffusivity*mUserDiffusionTime));
  ConvolveImage1Z->Update();

// Third: Multiply by the blood perfusion term and add Tblood:

	ImageType::Pointer ConductionAdvectionImage = ImageType::New();
	ConductionAdvectionImage = ConvolveImage1Z->GetOutput();

	float pixVal2=0;
	float newVal2=0;
	IteratorType it2( ConductionAdvectionImage, ConductionAdvectionImage->GetRequestedRegion() );

	for (it2.GoToBegin(); !it2.IsAtEnd(); ++it2)
	{
		pixVal2=it2.Get();
	        newVal2= pixVal2*std::exp(-(mUserBloodDensity*mUserBloodHeatCapacity)/(mUserTissueDensity*mUserTissueHeatCapacity)*mUserBloodPerfusionRate*mUserDiffusionTime)+mUserBloodTemperature;
		it2.Set( newVal2 );
	}

	ConductionAdvectionImage->Update();

  writer->SetFileName( mHeatConductionAdvectionFilename );
  writer->SetInput( ConductionAdvectionImage );  // heat diffusion (conduction and advection) map in temperature
  writer->Update();

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

   if ( process == "NanoAbsorption" )  mNanoAbsorptionImage.AddValue(index, edep);

  GateDebugMessageDec("Actor", 4, "GateNanoActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------







