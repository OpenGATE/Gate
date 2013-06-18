/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

// Gate 
#include "GateHybridForcedDetectionActor.hh"
#include "GateMiscFunctions.hh"
#include "GateScatterOrderTrackInformationActor.hh"

// G4
#include <G4Event.hh>
#include <G4MaterialTable.hh>
#include <G4ParticleTable.hh>
#include <G4VEmProcess.hh>
#include <G4TransportationManager.hh>
#include <G4LivermoreComptonModel.hh>
#include <G4SteppingManager.hh>

// rtk
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>

// itk
#include <itkImportImageFilter.h>
#include <itkChangeInformationImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkBinaryFunctorImageFilter.h>

#define TRY_AND_EXIT_ON_ITK_EXCEPTION(execFunc)                         \
  try                                                                   \
    {                                                                   \
    execFunc;                                                          \
    }                                                                   \
  catch( itk::ExceptionObject & err )                                   \
    {                                                                   \
    std::cerr << "ExceptionObject caught with " #execFunc << std::endl; \
    std::cerr << err << std::endl;                                      \
    exit(EXIT_FAILURE);                                                 \
    }

//-----------------------------------------------------------------------------
/// Constructors
GateHybridForcedDetectionActor::GateHybridForcedDetectionActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateHybridForcedDetectionActor() -- begin"<<G4endl);  
  pActorMessenger = new GateHybridForcedDetectionActorMessenger(this);
  mDetectorResolution[0] = mDetectorResolution[1] = mDetectorResolution[2] = 1;
  GateDebugMessageDec("Actor",4,"GateHybridForcedDetectionActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateHybridForcedDetectionActor::~GateHybridForcedDetectionActor() 
{
  delete pActorMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateHybridForcedDetectionActor::Construct()
{
  GateVActor::Construct();
  //  Callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  //   EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);
  ResetData();
  mEMCalculator = new G4EmCalculator;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin of Run
void GateHybridForcedDetectionActor::BeginOfRunAction(const G4Run*r)
{
  GateVActor::BeginOfRunAction(r);
  mNumberOfEventsInRun = 0;

  // Get information on the source
  GateSourceMgr * sm = GateSourceMgr::GetInstance();
  if (sm->GetNumberOfSources() == 0) {
    GateError("No source set. Abort.");
  }
  if (sm->GetNumberOfSources() != 1) {
    GateWarning("Several sources found, we consider the first one.");
  }
  mSource = sm->GetSource(0);
    // Checks. FIXME: check on rot1 and rot2 would be required
  if(mSource->GetAngDist()->GetDistType() != "focused")
    GateError("Forced detection only supports point or focused sources.");
  if(mSource->GetPosDist()->GetPosDisType() != "Plane")
    GateError("Forced detection only supports Plane distributions.");

  // Create list of energies
  double energyMax = 0.;
  std::vector<double> energyList;
  std::vector<double> energyWeightList;
  G4String st = mSource->GetEneDist()->GetEnergyDisType();
  if (st == "Mono") {
    energyList.push_back(mSource->GetEneDist()->GetMonoEnergy());
    energyWeightList.push_back(1.);
    energyMax = std::max(energyMax, energyList.back());
  }
  else if (st == "User") { // histo
    G4PhysicsOrderedFreeVector h = mSource->GetEneDist()->GetUserDefinedEnergyHisto ();
    double weightSum = 0.;
    for(unsigned int i=0; i<h.GetVectorLength(); i++) {
      double E = h.Energy(i);
      energyList.push_back(E);
      energyWeightList.push_back(h.Value(E));
      weightSum += energyWeightList.back();
      energyMax = std::max(energyMax, energyList.back());
    }
    for(unsigned int i=0; i<h.GetVectorLength(); i++)
      energyWeightList[i] /= weightSum;
  }
  else
    GateError("Error, source type is not Mono or User. Abort.");


  // Search for voxelized volume. If more than one, crash (yet).
  GateVImageVolume* gate_image_volume = NULL;
  for(std::map<G4String, GateVVolume*>::const_iterator it  = GateObjectStore::GetInstance()->begin();
                                                       it != GateObjectStore::GetInstance()->end();
                                                       it++)
    {
    if(dynamic_cast<GateVImageVolume*>(it->second))
    {
      if(gate_image_volume != NULL)
        GateError("There is more than one voxelized volume and don't know yet how to cope with this.");
      else
        gate_image_volume = dynamic_cast<GateVImageVolume*>(it->second);
    }
  }
  if(!gate_image_volume)
    GateError("You need a voxelized volume in your scene.");

  // TODO: loop on volumes to check that they contain world material only

  // Conversion of CT to ITK and to int values
  mGateVolumeImage = ConvertGateImageToITKImage(gate_image_volume);

  // Create projection images
  mPrimaryImage = CreateVoidProjectionImage();
  mComptonImage = CreateVoidProjectionImage();
  mRayleighImage = CreateVoidProjectionImage();
  mFluorescenceImage = CreateVoidProjectionImage();

  mComptonPerOrderImages.clear();
  mRayleighPerOrderImages.clear();
  mFluorescencePerOrderImages.clear();

  // Create geometry and param of output image
  PointType primarySourcePosition;
  ComputeGeometryInfoInImageCoordinateSystem(gate_image_volume,
                                             mDetector,
                                             mSource,
                                             primarySourcePosition,
                                             mDetectorPosition,
                                             mDetectorRowVector,
                                             mDetectorColVector);

  // There are two geometry objects. One stores all projection images
  // (one per run) and the other contains the geometry of one projection
  // image.
  mGeometry->AddReg23Projection(primarySourcePosition,
                                mDetectorPosition,
                                mDetectorRowVector,
                                mDetectorColVector);
  GeometryType::Pointer oneProjGeometry = GeometryType::New();
  oneProjGeometry->AddReg23Projection(primarySourcePosition,
                                      mDetectorPosition,
                                      mDetectorRowVector,
                                      mDetectorColVector);

  // Create primary projector and compute primary
  mPrimaryProbe.Start();
  PrimaryProjectionType::Pointer primaryProjector = PrimaryProjectionType::New();
  primaryProjector->InPlaceOn();
  primaryProjector->SetInput(mPrimaryImage);
  primaryProjector->SetInput(1, mGateVolumeImage );
  primaryProjector->SetGeometry( oneProjGeometry.GetPointer() );
  primaryProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mPrimaryImage,
                                                                            mDetectorRowVector,
                                                                            mDetectorColVector);
  primaryProjector->GetProjectedValueAccumulation().SetVolumeSpacing( mGateVolumeImage->GetSpacing() );
  primaryProjector->GetProjectedValueAccumulation().SetInterpolationWeights( primaryProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights() );
  primaryProjector->GetProjectedValueAccumulation().SetEnergyWeightList( &energyWeightList );
  primaryProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                        energyList,
                                                                        gate_image_volume);
  primaryProjector->GetProjectedValueAccumulation().Init( primaryProjector->GetNumberOfThreads() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION(primaryProjector->Update());
  mPrimaryImage = primaryProjector->GetOutput();
  mPrimaryImage->DisconnectPipeline();

  // Compute flat field if required
  if(mAttenuationFilename != "")
  {
    // Constant image source of 1x1x1
    typedef rtk::ConstantImageSource< InputImageType > ConstantImageSourceType;
    ConstantImageSourceType::PointType origin;
    ConstantImageSourceType::SizeType dim;
    ConstantImageSourceType::SpacingType spacing;
    ConstantImageSourceType::Pointer flatFieldSource  = ConstantImageSourceType::New();
    origin[0] = 0.;
    origin[1] = 0.;
    origin[2] = 0.;
    dim[0] = 1;
    dim[1] = 1;
    dim[2] = 1;
    spacing[0] = 1.;
    spacing[1] = 1.;
    spacing[2] = 1.;
    flatFieldSource->SetOrigin( origin );
    flatFieldSource->SetSpacing( spacing );
    flatFieldSource->SetSize( dim );
    flatFieldSource->SetConstant( primaryProjector->GetProjectedValueAccumulation().GetMaterialMuMap()->GetLargestPossibleRegion().GetSize()[0]-1 );

    mFlatFieldImage = CreateVoidProjectionImage();
    primaryProjector->SetInput(mFlatFieldImage);
    primaryProjector->SetInput(1, flatFieldSource->GetOutput() );
    TRY_AND_EXIT_ON_ITK_EXCEPTION(primaryProjector->Update());
    mFlatFieldImage = primaryProjector->GetOutput();
  }
  mPrimaryProbe.Stop();

  // Prepare Compton
  mComptonProjector = ComptonProjectionType::New();
  mComptonProjector->InPlaceOn();
  mComptonProjector->SetInput(mComptonImage);
  mComptonProjector->SetInput(1, mGateVolumeImage );
  mComptonProjector->SetGeometry( oneProjGeometry.GetPointer() );
  mComptonProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mComptonImage,
                                                                             mDetectorRowVector,
                                                                             mDetectorColVector);
  mComptonProjector->GetProjectedValueAccumulation().SetVolumeSpacing( mGateVolumeImage->GetSpacing() );
  mComptonProjector->GetProjectedValueAccumulation().SetInterpolationWeights( mComptonProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights() );
  mComptonProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                         1.*keV,
                                                                         energyMax,
                                                                         gate_image_volume);
  mComptonProjector->GetProjectedValueAccumulation().Init( mComptonProjector->GetNumberOfThreads() );

  // Prepare Rayleigh
  mRayleighProjector = RayleighProjectionType::New();
  mRayleighProjector->InPlaceOn();
  mRayleighProjector->SetInput(mRayleighImage);
  mRayleighProjector->SetInput(1, mGateVolumeImage );
  mRayleighProjector->SetGeometry( oneProjGeometry.GetPointer() );
  mRayleighProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mRayleighImage,
                                                                              mDetectorRowVector,
                                                                              mDetectorColVector);
  mRayleighProjector->GetProjectedValueAccumulation().SetVolumeSpacing( mGateVolumeImage->GetSpacing() );
  mRayleighProjector->GetProjectedValueAccumulation().SetInterpolationWeights( mRayleighProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights() );
  mRayleighProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                         1.*keV,
                                                                         energyMax,
                                                                         gate_image_volume);
  mRayleighProjector->GetProjectedValueAccumulation().Init( mRayleighProjector->GetNumberOfThreads() );

  // Prepare Fluorescence
  mFluorescenceProjector = FluorescenceProjectionType::New();
  mFluorescenceProjector->InPlaceOn();
  mFluorescenceProjector->SetInput(mFluorescenceImage);
  mFluorescenceProjector->SetInput(1, mGateVolumeImage );
  mFluorescenceProjector->SetGeometry( oneProjGeometry.GetPointer() );
  mFluorescenceProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mFluorescenceImage,
                                                                              mDetectorRowVector,
                                                                              mDetectorColVector);
  mFluorescenceProjector->GetProjectedValueAccumulation().SetVolumeSpacing( mGateVolumeImage->GetSpacing() );
  mFluorescenceProjector->GetProjectedValueAccumulation().SetInterpolationWeights( mFluorescenceProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights() );
  mFluorescenceProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                         1.*keV,
                                                                         energyMax,
                                                                         gate_image_volume);
  mFluorescenceProjector->GetProjectedValueAccumulation().Init( mFluorescenceProjector->GetNumberOfThreads() );

  // Create a single event if asked for it
  if(mSingleInteractionFilename!="") {
    // d and p are in World coordinates and they must be in CT coordinates
    G4ThreeVector d = m_WorldToCT.TransformAxis(mSingleInteractionDirection);
    G4ThreeVector p = m_WorldToCT.TransformPoint(mSingleInteractionPosition);

    //Convert to ITK
    PointType point;
    VectorType direction;
    for(unsigned int i=0; i<3; i++) {
      point[i] = p[i];
      direction[i] = d[i];
    }

    if(mSingleInteractionType == "Compton") {
      mComptonProjector->InPlaceOff();
      GeometryType::Pointer oneProjGeometry = GeometryType::New();
      oneProjGeometry->AddReg23Projection(point,
                                          mDetectorPosition,
                                          mDetectorRowVector,
                                          mDetectorColVector);
      mComptonProjector->SetInput(mComptonImage);
      mComptonProjector->SetGeometry( oneProjGeometry.GetPointer() );
      mComptonProjector->GetProjectedValueAccumulation().SetEnergyZAndWeight( mSingleInteractionEnergy,
                                                                              mSingleInteractionZ,
                                                                              1. );
      mComptonProjector->GetProjectedValueAccumulation().SetDirection( direction );
      TRY_AND_EXIT_ON_ITK_EXCEPTION(mComptonProjector->Update());
      mSingleInteractionImage = mComptonProjector->GetOutput();
      mSingleInteractionImage->DisconnectPipeline();
      mComptonProjector->InPlaceOn();
    }
    if(mSingleInteractionType == "Rayleigh") {
      mRayleighProjector->InPlaceOff();
      GeometryType::Pointer oneProjGeometry = GeometryType::New();
      oneProjGeometry->AddReg23Projection(point,
                                          mDetectorPosition,
                                          mDetectorRowVector,
                                          mDetectorColVector);
      mRayleighProjector->SetInput(mRayleighImage);
      mRayleighProjector->SetGeometry( oneProjGeometry.GetPointer() );
      mRayleighProjector->GetProjectedValueAccumulation().SetEnergyZAndWeight( mSingleInteractionEnergy,
                                                                               mSingleInteractionZ,
                                                                               1. );
      mRayleighProjector->GetProjectedValueAccumulation().SetDirection( direction );
      TRY_AND_EXIT_ON_ITK_EXCEPTION(mRayleighProjector->Update());
      mSingleInteractionImage = mRayleighProjector->GetOutput();
      mSingleInteractionImage->DisconnectPipeline();
      mRayleighProjector->InPlaceOn();
    }
    if(mSingleInteractionType == "Fluorescence") {
      mFluorescenceProjector->InPlaceOff();
      GeometryType::Pointer oneProjGeometry = GeometryType::New();
      oneProjGeometry->AddReg23Projection(point,
                                          mDetectorPosition,
                                          mDetectorRowVector,
                                          mDetectorColVector);
      mFluorescenceProjector->SetInput(mRayleighImage);
      mFluorescenceProjector->SetGeometry( oneProjGeometry.GetPointer() );
      mFluorescenceProjector->GetProjectedValueAccumulation().SetEnergyAndWeight( mSingleInteractionEnergy,
                                                                                   1. );
      mFluorescenceProjector->GetProjectedValueAccumulation().SetDirection( direction );
      TRY_AND_EXIT_ON_ITK_EXCEPTION(mFluorescenceProjector->Update());
      mSingleInteractionImage = mFluorescenceProjector->GetOutput();
      mSingleInteractionImage->DisconnectPipeline();
      mFluorescenceProjector->InPlaceOn();
    }
  }

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Event
void GateHybridForcedDetectionActor::BeginOfEventAction(const G4Event*itkNotUsed(e))
{
  mNumberOfEventsInRun++;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Track
/*void GateHybridForcedDetectionActor::PreUserTrackingAction(const GateVVolume * v, const G4Track*t)
{
}*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callbacks
void GateHybridForcedDetectionActor::UserSteppingAction(const GateVVolume * v,
                                                        const G4Step * step)
{
  GateVActor::UserSteppingAction(v, step);
  /* Get interaction point from step
     Retrieve : 
     - type of limiting process (Compton Rayleigh Fluorescence)
     - no Fluo yet, wait for bug fix in next G4 release (4.6 ?)
     - coordinate of interaction, convert if needed into world coordinate system
     - Get Energy
     - -> generate adequate forward projections towards detector
  */

  // We are only interested in EM processes. One should check post-step to know
  // what is going to happen, but pre-step is used afterward to get direction
  // and position.
  const G4VProcess *pr = step->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VEmProcess *process = dynamic_cast<const G4VEmProcess*>(pr);
  if(!process) return;

  // We need the position, direction and energy at point where Compton and Rayleigh occur.
  G4ThreeVector p = step->GetPostStepPoint()->GetPosition();
  G4ThreeVector d = step->GetPreStepPoint()->GetMomentumDirection();
  double energy = step->GetPreStepPoint()->GetKineticEnergy();
  double weight = step->GetPostStepPoint()->GetWeight();

  GateScatterOrderTrackInformation * info = dynamic_cast<GateScatterOrderTrackInformation *>(step->GetTrack()->GetUserInformation());

  // d and p are in World coordinates and they must be in CT coordinates
  d = m_WorldToCT.TransformAxis(d);
  p = m_WorldToCT.TransformPoint(p);

  //Convert to ITK
  PointType point;
  VectorType direction;
  for(unsigned int i=0; i<3; i++) {
    point[i] = p[i];
    direction[i] = d[i];
  }

  //FIXME: do we prefer this solution or computing the scattering function for the material?
  const G4MaterialCutsCouple *couple = step->GetPreStepPoint()->GetMaterialCutsCouple();
  const G4ParticleDefinition *particle = step->GetTrack()->GetParticleDefinition();
  G4VEmModel* model = const_cast<G4VEmProcess*>(process)->Model();
  const G4Element* elm = model->SelectRandomAtom(couple,particle,energy);
  G4int Z = elm->GetZ();

  if(process->GetProcessName() == G4String("Compton") || process->GetProcessName() == G4String("compt")) {
    mComptonProbe.Start();
    GeometryType::Pointer oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddReg23Projection(point,
                                        mDetectorPosition,
                                        mDetectorRowVector,
                                        mDetectorColVector);
    mComptonProjector->SetInput(mComptonImage);
    mComptonProjector->SetGeometry( oneProjGeometry.GetPointer() );
    mComptonProjector->GetProjectedValueAccumulation().SetEnergyZAndWeight( energy, Z, weight );
    mComptonProjector->GetProjectedValueAccumulation().SetDirection( direction );
    TRY_AND_EXIT_ON_ITK_EXCEPTION(mComptonProjector->Update());
    mComptonImage = mComptonProjector->GetOutput();
    mComptonImage->DisconnectPipeline();
    mComptonProbe.Stop();

    // Scatter order
    if(info)
    {
      unsigned int order = info->GetScatterOrder();
      while(order>=mComptonPerOrderImages.size())
        mComptonPerOrderImages.push_back( CreateVoidProjectionImage() );
      mComptonProjector->SetInput(mComptonPerOrderImages[order]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(mComptonProjector->Update());
      mComptonPerOrderImages[order] = mComptonProjector->GetOutput();
      mComptonPerOrderImages[order]->DisconnectPipeline();
    }


  }
  else if(process->GetProcessName() == G4String("RayleighScattering") || process->GetProcessName() == G4String("Rayl")) {
    mRayleighProbe.Start();
    GeometryType::Pointer oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddReg23Projection(point,
                                        mDetectorPosition,
                                        mDetectorRowVector,
                                        mDetectorColVector);
    mRayleighProjector->SetInput(mRayleighImage);
    mRayleighProjector->SetGeometry( oneProjGeometry.GetPointer() );
    mRayleighProjector->GetProjectedValueAccumulation().SetEnergyZAndWeight( energy, Z, weight );
    mRayleighProjector->GetProjectedValueAccumulation().SetDirection( direction );
    TRY_AND_EXIT_ON_ITK_EXCEPTION(mRayleighProjector->Update());
    mRayleighImage = mRayleighProjector->GetOutput();
    mRayleighImage->DisconnectPipeline();
    mRayleighProbe.Stop();

    // Scatter order
    if(info)
    {
      unsigned int order = info->GetScatterOrder();
        while(order>=mRayleighPerOrderImages.size())
          mRayleighPerOrderImages.push_back( CreateVoidProjectionImage() );
        mRayleighProjector->SetInput(mRayleighPerOrderImages[order]);
        TRY_AND_EXIT_ON_ITK_EXCEPTION(mRayleighProjector->Update());
        mRayleighPerOrderImages[order] = mRayleighProjector->GetOutput();
        mRayleighPerOrderImages[order]->DisconnectPipeline();
    }
  }
  else if(process->GetProcessName() == G4String("PhotoElectric") || process->GetProcessName() == G4String("phot")) {
    // List of secondary particles
    const G4TrackVector * list = step->GetSecondary();
    G4String nameSecondary = G4String("0");
    G4double energySecondary = 0;
    VectorType directionSecondary;

    for(unsigned int i = 0; i<(*list).size(); i++) {
      nameSecondary = (*list)[i]->GetDefinition()->GetParticleName();

      // Check if photon has been emitted
      if(nameSecondary==G4String("gamma")) {

        GateScatterOrderTrackInformation * infoSecondary = dynamic_cast<GateScatterOrderTrackInformation *>((*list)[i]->GetUserInformation());

        // Update direction and energy for secondary photon
        energySecondary = (*list)[i]->GetKineticEnergy();
        for(unsigned int j=0; j<3; j++)
          directionSecondary[j] = (*list)[i]->GetMomentumDirection()[j];
        mFluorescenceProbe.Start();
        GeometryType::Pointer oneProjGeometry = GeometryType::New();
        oneProjGeometry->AddReg23Projection(point,
                                            mDetectorPosition,
                                            mDetectorRowVector,
                                            mDetectorColVector);
        mFluorescenceProjector->SetInput(mFluorescenceImage);
        mFluorescenceProjector->SetGeometry( oneProjGeometry.GetPointer() );
        mFluorescenceProjector->GetProjectedValueAccumulation().SetEnergyAndWeight( energySecondary, weight );
        mFluorescenceProjector->GetProjectedValueAccumulation().SetDirection( directionSecondary );
        TRY_AND_EXIT_ON_ITK_EXCEPTION(mFluorescenceProjector->Update());
        mFluorescenceImage = mFluorescenceProjector->GetOutput();
        mFluorescenceImage->DisconnectPipeline();
        mFluorescenceProbe.Stop();

        // Scatter order
        if(infoSecondary)
        {
          unsigned int order = infoSecondary->GetScatterOrder();
          while(order>=mFluorescencePerOrderImages.size())
            mFluorescencePerOrderImages.push_back( CreateVoidProjectionImage() );
          mFluorescenceProjector->SetInput(mFluorescencePerOrderImages[order]);
          TRY_AND_EXIT_ON_ITK_EXCEPTION(mFluorescenceProjector->Update());
          mFluorescencePerOrderImages[order] = mFluorescenceProjector->GetOutput();
          mFluorescencePerOrderImages[order]->DisconnectPipeline();
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateHybridForcedDetectionActor::SaveData()
{
  GateVActor::SaveData();
  // Geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer geoWriter =
      rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  geoWriter->SetObject(mGeometry);
  geoWriter->SetFilename(mGeometryFilename);
  geoWriter->WriteFile();

  itk::ImageFileWriter<InputImageType>::Pointer imgWriter;
  imgWriter = itk::ImageFileWriter<InputImageType>::New();
  char filename[1024];
  G4int rID = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();

  if(mPrimaryFilename != "") {
    // Normalize by the number of particles
    // FIXME: it assumes here that all particles hit the detector and every point
    // of the detector has the same probability to be hit.
    typedef itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyType;
    MultiplyType::Pointer mult = MultiplyType::New();
    mult->SetInput(mPrimaryImage);
    InputImageType::SizeType size = mPrimaryImage->GetLargestPossibleRegion().GetSize();
    mult->SetConstant(mNumberOfEventsInRun / double(size[0] * size[1]));
    mult->InPlaceOff();

    // Write the image of primary radiation
    sprintf(filename, mPrimaryFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(mult->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
  }

  if(mMaterialMuFilename != "") {
    AccumulationType::MaterialMuImageType *map;
    map = mComptonProjector->GetProjectedValueAccumulation().GetMaterialMu();

    // Change spacing to keV
    AccumulationType::MaterialMuImageType::SpacingType spacing = map->GetSpacing();
    spacing[1] /= keV;

    typedef itk::ChangeInformationImageFilter<AccumulationType::MaterialMuImageType> CIType;
    CIType::Pointer ci = CIType::New();
    ci->SetInput(map);
    ci->SetOutputSpacing(spacing);
    ci->ChangeSpacingOn();
    ci->Update();

    typedef itk::ImageFileWriter<AccumulationType::MaterialMuImageType> TwoDWriter;
    TwoDWriter::Pointer w = TwoDWriter::New();
    w->SetInput( ci->GetOutput() );
    w->SetFileName(mMaterialMuFilename);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(w->Update());
  }

  if(mAttenuationFilename != "") {
    //Attenuation Functor -> atten
    typedef itk::BinaryFunctorImageFilter< InputImageType, InputImageType, InputImageType,
                                           GateHybridForcedDetectionFunctor::Attenuation<InputImageType::PixelType> > attenFunctor;
    attenFunctor::Pointer atten = attenFunctor::New();
    atten->SetInput1(mPrimaryImage);
    atten->SetInput2(mFlatFieldImage);
    atten->InPlaceOff();

    sprintf(filename, mAttenuationFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(atten->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
  }

  if(mFlatFieldFilename != "")
  {
    sprintf(filename, mFlatFieldFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(mFlatFieldImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
  }

  if(mComptonFilename != "")
  {

    sprintf(filename, mComptonFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(mComptonImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());

    for(unsigned int k = 0; k<mComptonPerOrderImages.size(); k++)
    {
      sprintf(filename, "output/compton%04d_%04d.mha", rID, k+1);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mComptonPerOrderImages[k]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }
  }


  if(mRayleighFilename != "")
  {
    sprintf(filename, mRayleighFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(mRayleighImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());

    for(unsigned int k = 0; k<mRayleighPerOrderImages.size(); k++)
    {
      sprintf(filename, "output/rayleigh%04d_%04d.mha", rID, k+1);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mRayleighPerOrderImages[k]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }
  }

  if(mFluorescenceFilename != "")
  {
    sprintf(filename, mFluorescenceFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(mFluorescenceImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());

    for(unsigned int k = 0; k<mFluorescencePerOrderImages.size(); k++)
    {
      sprintf(filename, "output/fluorescence%04d_%04d.mha", rID, k+1);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mFluorescencePerOrderImages[k]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }
  }

  if(mSingleInteractionFilename!="") {
    imgWriter->SetFileName(mSingleInteractionFilename);
    imgWriter->SetInput(mSingleInteractionImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
  }
//  G4cout << "Computation of the primary took "
//         << mPrimaryProbe.GetTotal()
//         << ' '
//         << mPrimaryProbe.GetUnit()
//         << G4endl;

//  G4cout << "Computation of Compton took "
//         << mComptonProbe.GetTotal()
//         << ' '
//         << mComptonProbe.GetUnit()
//         << G4endl;

//  G4cout << "Computation of Rayleigh took "
//         << mRayleighProbe.GetTotal()
//         << ' '
//         << mRayleighProbe.GetUnit()
//         << G4endl;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActor::ResetData() 
{
  mGeometry = GeometryType::New();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActor::ComputeGeometryInfoInImageCoordinateSystem(
        GateVImageVolume *ct,
        GateVVolume *detector,
        GateVSource *src,
        PointType &primarySourcePosition,
        PointType &detectorPosition,
        VectorType &detectorRowVector,
        VectorType &detectorColVector)
{ 
  // The placement of a volume relative to its mother's coordinate system is not
  // very well explained in Geant4's doc but the code follows what's done in
  // source/geometry/volumes/src/G4PVPlacement.cc.
  //
  // One must be extremely careful with the multiplication order. It is not
  // intuitive in Geant4, i.e., G4AffineTransform.Product(A, B) means
  // B*A in matrix notations.

  // Detector to world
  GateVVolume * v = detector;
  G4VPhysicalVolume * phys = v->GetPhysicalVolume();
  G4AffineTransform detectorToWorld(phys->GetRotation(), phys->GetTranslation());

  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    detectorToWorld = detectorToWorld * x;
  }

  // CT to world
  v = ct;
  phys = v->GetPhysicalVolume();
  G4AffineTransform ctToWorld(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    ctToWorld = ctToWorld * x;
  }
  m_WorldToCT = ctToWorld.Inverse();

  // Source to world
  G4String volname = src->GetRelativePlacementVolume();
  v = GateObjectStore::GetInstance()->FindVolumeCreator(volname);
  phys = v->GetPhysicalVolume();
  G4AffineTransform sourceToWorld(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    sourceToWorld = sourceToWorld * x;
  }

  // Detector parameters
  G4AffineTransform detectorToCT(detectorToWorld *  m_WorldToCT);

  // TODO: check where to get the two directions of the detector.
  // Probably the dimension that has lowest size in one of the three directions. 
  G4ThreeVector du = detectorToCT.TransformAxis(G4ThreeVector(1,0,0));
  G4ThreeVector dv = detectorToCT.TransformAxis(G4ThreeVector(0,1,0));
  G4ThreeVector dp = detectorToCT.TransformPoint(G4ThreeVector(0,0,0));

  // Source
  G4ThreeVector s = src->GetAngDist()->GetFocusPointCopy();

  G4AffineTransform sourceToCT( sourceToWorld *  m_WorldToCT);
  s = sourceToCT.TransformPoint(s);

  // Copy in ITK vectors
  for(int i=0; i<3; i++) {
    detectorRowVector[i] = du[i];
    detectorColVector[i] = dv[i];
    detectorPosition[i] = dp[i];
    primarySourcePosition[i] = s[i];
  }


}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHybridForcedDetectionActor::InputImageType::Pointer
GateHybridForcedDetectionActor::ConvertGateImageToITKImage(GateVImageVolume * gateImgVol)
{
  GateImage *gateImg = gateImgVol->GetImage();

  // The direction is not accounted for in Gate.
  InputImageType::SizeType size;
  InputImageType::PointType origin;
  InputImageType::RegionType region;
  InputImageType::SpacingType spacing;
  for(unsigned int i=0; i<3; i++) {
    size[i] = gateImg->GetResolution()[i];
    spacing[i] = gateImg->GetVoxelSize()[i];
    origin[i] = -gateImg->GetHalfSize()[i]+0.5*spacing[i];
  }
  region.SetSize(size);

  itk::ImportImageFilter<InputPixelType, Dimension>::Pointer import;
  import = itk::ImportImageFilter<InputPixelType, Dimension>::New();
  import->SetRegion(region);
  import->SetImportPointer(&*(gateImg->begin()), gateImg->GetNumberOfValues(), false);
  import->SetSpacing(spacing);
  import->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(import->Update());

  // Get world material
  std::vector<G4Material*> mat;
  gateImgVol->BuildLabelToG4MaterialVector( mat );
  InputPixelType worldMat = mat.size();

  // Pad 1 pixel with world material because interpolation will cut out half a voxel around
  itk::ConstantPadImageFilter<InputImageType, InputImageType>::Pointer pad;
  pad = itk::ConstantPadImageFilter<InputImageType, InputImageType>::New();
  InputImageType::SizeType border;
  border.Fill(1);
  pad->SetPadBound(border);
  pad->SetConstant(worldMat);
  pad->SetInput(import->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(pad->Update());

  InputImageType::Pointer output = pad->GetOutput();
  output->DisconnectPipeline();
  return output;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHybridForcedDetectionActor::InputImageType::Pointer
GateHybridForcedDetectionActor::CreateVoidProjectionImage()
{
  mDetector = GateObjectStore::GetInstance()->FindVolumeCreator(mDetectorName);

  InputImageType::SizeType size;
  size[0] = GetDetectorResolution()[0];
  size[1] = GetDetectorResolution()[1];
  size[2] = 1;

  InputImageType::SpacingType spacing;
  spacing[0] = mDetector->GetHalfDimension(0)*2.0/size[0];
  spacing[1] = mDetector->GetHalfDimension(1)*2.0/size[1];
  spacing[2] = 1.0;

  InputImageType::PointType origin;
  origin[0] = -mDetector->GetHalfDimension(0)+0.5*spacing[0];
  origin[1] = -mDetector->GetHalfDimension(1)+0.5*spacing[1];
  origin[2] = 0.0;

  rtk::ConstantImageSource<InputImageType>::Pointer source;
  source = rtk::ConstantImageSource<InputImageType>::New();
  source->SetSpacing(spacing);
  source->SetOrigin(origin);
  source->SetSize(size);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(source->Update());

  GateHybridForcedDetectionActor::InputImageType::Pointer output;
  output = source->GetOutput();
  output->DisconnectPipeline();

  return output;
}
//-----------------------------------------------------------------------------

#endif
