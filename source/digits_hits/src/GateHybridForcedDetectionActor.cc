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
#include "GateHybridForcedDetectionFunctors.hh"

// G4
#include <G4Event.hh>
#include <G4MaterialTable.hh>
#include <G4ParticleTable.hh>
#include <G4VEmProcess.hh>
#include <G4TransportationManager.hh>
#include <G4LivermoreComptonModel.hh>
#include <G4SteppingManager.hh>
#include <G4NistManager.hh>

// rtk
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>
#include <rtkMacro.h>

// itk
#include <itkImportImageFilter.h>
#include <itkChangeInformationImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkBinaryFunctorImageFilter.h>
#include <itkAddImageFilter.h>

//-----------------------------------------------------------------------------
/// Constructors
GateHybridForcedDetectionActor::GateHybridForcedDetectionActor(G4String name, G4int depth):
  GateVActor(name,depth),
  mIsSecondarySquaredImageEnabled(false),
  mIsSecondaryUncertaintyImageEnabled(false)
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
  EnableEndOfEventAction(true);
  //   EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);
  ResetData();
  mEMCalculator = new G4EmCalculator;

  mPhaseSpaceFile = NULL;
  if(mPhaseSpaceFilename != "")
    mPhaseSpaceFile = new TFile(mPhaseSpaceFilename,"RECREATE","ROOT file for phase space",9);
  mPhaseSpace = new TTree("PhaseSpace","Phase space tree of hybrid forced detection actor");

  mPhaseSpace->Branch("Ekine",  &mInteractionEnergy, "Ekine/D");
  mPhaseSpace->Branch("Weight", &mInteractionWeight, "Weight/D");
  mPhaseSpace->Branch("X", &(mInteractionPosition[0]), "X/D");
  mPhaseSpace->Branch("Y", &(mInteractionPosition[1]), "Y/D");
  mPhaseSpace->Branch("Z", &(mInteractionPosition[2]), "Z/D");
  mPhaseSpace->Branch("dX", &(mInteractionDirection[0]), "dX/D");
  mPhaseSpace->Branch("dY", &(mInteractionDirection[1]), "dY/D");
  mPhaseSpace->Branch("dZ", &(mInteractionDirection[2]), "dZ/D");
  mPhaseSpace->Branch("ProductionVolume", mInteractionProductionVolume, "ProductionVolume/C");
  mPhaseSpace->Branch("TrackID",&mInteractionTrackId, "TrackID/I");
  mPhaseSpace->Branch("EventID",&mInteractionEventId, "EventID/I");
  mPhaseSpace->Branch("RunID",&mInteractionRunId, "RunID/I");
  mPhaseSpace->Branch("ProductionProcessTrack", mInteractionProductionProcessTrack, "ProductionProcessTrack/C");
  mPhaseSpace->Branch("ProductionProcessStep", mInteractionProductionProcessStep, "ProductionProcessStep/C");
  mPhaseSpace->Branch("TotalContribution", &mInteractionTotalContribution, "TotalContribution/D");
  mPhaseSpace->Branch("InteractionVolume", mInteractionVolume, "Volume/C");
  mPhaseSpace->Branch("Material", mInteractionMaterial, "Material/C");
  mPhaseSpace->Branch("MaterialZ", &mInteractionZ, "MaterialZ/I");
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
  if(mSource->GetPosDist()->GetPosDisType() == "Point") {
    if(mSource->GetAngDist()->GetDistType() != "iso") {
      GateError("Forced detection only supports iso distributions with Point source.");
    }
  }
  else if(mSource->GetPosDist()->GetPosDisType() == "Plane" ||
     mSource->GetPosDist()->GetPosDisType() == "UserFluenceImage") {
    if(mSource->GetAngDist()->GetDistType() != "focused") {
      GateError("Forced detection only supports focused distributions for Plane and UserFluenceImage sources.");
    }
  }
  else
    GateError("Forced detection only supports Point, Plane or UserFluenceImage distributions.");

  // Read the response detector curve from an external file
  mEnergyResponseDetector.ReadResponseDetectorFile(mResponseFilename);

  // Create list of energies
  double energyMax = 0.;
  std::vector<double> energyList;
  std::vector<double> energyWeightList;
  G4String st = mSource->GetEneDist()->GetEnergyDisType();
  if (st == "Mono") {
    energyList.push_back(mSource->GetEneDist()->GetMonoEnergy());
    energyWeightList.push_back(mEnergyResponseDetector(energyList.back()));
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
      energyWeightList.back() *= mEnergyResponseDetector(energyList.back());
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
  mSecondarySquared = CreateVoidProjectionImage();
  mComptonImage = CreateVoidProjectionImage();
  mRayleighImage = CreateVoidProjectionImage();
  mFluorescenceImage = CreateVoidProjectionImage();

  mComptonPerOrderImages.clear();
  mRayleighPerOrderImages.clear();
  mFluorescencePerOrderImages.clear();

  // Create geometry and param of output image
  ComputeGeometryInfoInImageCoordinateSystem(gate_image_volume,
                                             mDetector,
                                             mSource,
                                             mPrimarySourcePosition,
                                             mDetectorPosition,
                                             mDetectorRowVector,
                                             mDetectorColVector);

  // There are two geometry objects. One stores all projection images
  // (one per run) and the other contains the geometry of one projection
  // image.
  mGeometry->AddReg23Projection(mPrimarySourcePosition,
                                mDetectorPosition,
                                mDetectorRowVector,
                                mDetectorColVector);
  GeometryType::Pointer oneProjGeometry = GeometryType::New();
  oneProjGeometry->AddReg23Projection(mPrimarySourcePosition,
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
  if(mAttenuationFilename != "" || mFlatFieldFilename != "")
  {
    // Constant image source of 1x1x1 voxel of world material
    typedef rtk::ConstantImageSource< InputImageType > ConstantImageSourceType;
    ConstantImageSourceType::PointType origin;
    ConstantImageSourceType::SizeType dim;
    ConstantImageSourceType::Pointer flatFieldSource  = ConstantImageSourceType::New();
    origin[0] = 0.;
    origin[1] = 0.;
    origin[2] = 0.;
    dim[0] = 1;
    dim[1] = 1;
    dim[2] = 1;
    flatFieldSource->SetOrigin( origin );
    flatFieldSource->SetSpacing( mGateVolumeImage->GetSpacing() );
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
  mComptonProjector->GetProjectedValueAccumulation().SetResponseDetector( &mEnergyResponseDetector );
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
      mComptonProjector->GetProjectedValueAccumulation().SetResponseDetector( &mEnergyResponseDetector );
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
                                                                               mEnergyResponseDetector(mSingleInteractionEnergy) );
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
                                                                                  mEnergyResponseDetector(mSingleInteractionEnergy) );
      TRY_AND_EXIT_ON_ITK_EXCEPTION(mFluorescenceProjector->Update());
      mSingleInteractionImage = mFluorescenceProjector->GetOutput();
      mSingleInteractionImage->DisconnectPipeline();
      mFluorescenceProjector->InPlaceOn();
    }
  }

  if(mWaterLUTFilename != "")
    CreateWaterLUT(energyList, energyWeightList);

  if(mIsSecondarySquaredImageEnabled || mIsSecondaryUncertaintyImageEnabled) {
    mEventComptonImage = CreateVoidProjectionImage();
    mEventRayleighImage = CreateVoidProjectionImage();
    mEventFluorescenceImage = CreateVoidProjectionImage();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActor::BeginOfEventAction(const G4Event *itkNotUsed(e))
{
  mNumberOfEventsInRun++;

  if(mIsSecondarySquaredImageEnabled || mIsSecondaryUncertaintyImageEnabled) {
    // The event contribution are put in new images which at this point are in the
    // mEventComptonImage / mEventRayleighImage / mEventFluorescenceImage. We therefore
    // swap the two and they will be swapped back in EndOfEventAction.
    std::swap(mEventComptonImage, mComptonImage);
    std::swap(mEventRayleighImage, mRayleighImage);
    std::swap(mEventFluorescenceImage, mFluorescenceImage);

    // Make sure the time stamps of the mEvent images are more recent to detect if one
    // image has been modified during the event.
    mEventComptonImage->Modified();
    mEventRayleighImage->Modified();
    mEventFluorescenceImage->Modified();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActor::EndOfEventAction(const G4Event *e)
{
  typedef itk::AddImageFilter <OutputImageType, OutputImageType, OutputImageType> AddImageFilterType;
  AddImageFilterType::Pointer addFilter = AddImageFilterType::New();

  if(mIsSecondarySquaredImageEnabled || mIsSecondaryUncertaintyImageEnabled) {
    // First: accumulate contribution to event, square and add to total squared
    InputImageType::Pointer totalContribEvent(NULL);
    if( mEventComptonImage->GetTimeStamp() < mComptonImage->GetTimeStamp() ) {
      totalContribEvent = mComptonImage;
    }
    if( mEventRayleighImage->GetTimeStamp() < mRayleighImage->GetTimeStamp() ) {
      if(totalContribEvent.GetPointer()) {
        addFilter->SetInput1(totalContribEvent);
        addFilter->SetInput2(mRayleighImage);
        addFilter->InPlaceOff();
        TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
        totalContribEvent = addFilter->GetOutput();
        totalContribEvent->DisconnectPipeline();
      }
      else {
        totalContribEvent = mRayleighImage;
      }
    }
    if( mEventFluorescenceImage->GetTimeStamp() < mFluorescenceImage->GetTimeStamp() ) {
      if(totalContribEvent.GetPointer()) {
        addFilter->SetInput1(totalContribEvent);
        addFilter->SetInput2(mFluorescenceImage);
        addFilter->InPlaceOff();
        TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
        totalContribEvent = addFilter->GetOutput();
        totalContribEvent->DisconnectPipeline();
      }
      else {
        totalContribEvent = mFluorescenceImage;
      }
    }
    if(totalContribEvent.GetPointer()) {
      typedef itk::MultiplyImageFilter<OutputImageType, OutputImageType, OutputImageType> MultiplyImageFilterType;
      MultiplyImageFilterType::Pointer multFilter = MultiplyImageFilterType::New();
      multFilter->InPlaceOff();
      multFilter->SetInput1(totalContribEvent);
      multFilter->SetInput2(totalContribEvent);
      addFilter->SetInput1(mSecondarySquared);
      addFilter->SetInput2(multFilter->GetOutput());
      addFilter->InPlaceOn();
      TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
      mSecondarySquared = addFilter->GetOutput();
      mSecondarySquared->DisconnectPipeline();
    }

    // Second: accumulate non squared images and reset mEvent images
    if( mEventComptonImage->GetTimeStamp() < mComptonImage->GetTimeStamp() ) {
      addFilter->SetInput1(mEventComptonImage);
      addFilter->SetInput2(mComptonImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
      mComptonImage = addFilter->GetOutput();
      mComptonImage->DisconnectPipeline();
      mEventComptonImage = CreateVoidProjectionImage();
    }
    else
      std::swap(mEventComptonImage, mComptonImage);
    if( mEventRayleighImage->GetTimeStamp() < mRayleighImage->GetTimeStamp() ) {
      mRayleighImage->DisconnectPipeline();
      addFilter->SetInput1(mEventRayleighImage);
      addFilter->SetInput2(mRayleighImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
      mRayleighImage = addFilter->GetOutput();
      mRayleighImage->DisconnectPipeline();
      mEventRayleighImage = CreateVoidProjectionImage();
    }
    else
      std::swap(mEventRayleighImage, mRayleighImage);
    if( mEventFluorescenceImage->GetTimeStamp() < mFluorescenceImage->GetTimeStamp() ) {
      addFilter->SetInput1(mEventFluorescenceImage);
      addFilter->SetInput2(mFluorescenceImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
      mFluorescenceImage = addFilter->GetOutput();
      mFluorescenceImage->DisconnectPipeline();
      mEventFluorescenceImage = CreateVoidProjectionImage();
    }
    else
      std::swap(mEventFluorescenceImage, mFluorescenceImage);
  }

  GateVActor::EndOfEventAction(e);
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
  mInteractionPosition = step->GetPostStepPoint()->GetPosition();
  mInteractionDirection = step->GetPreStepPoint()->GetMomentumDirection();
  mInteractionEnergy = step->GetPreStepPoint()->GetKineticEnergy();
  mInteractionWeight = step->GetPostStepPoint()->GetWeight();

  // Other information for phase space
  mInteractionTrackId = step->GetTrack()->GetTrackID();
  mInteractionEventId = GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
  mInteractionRunId   = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();
  // mInteractionProductionVolume
  G4String st = "";
  if(step->GetTrack()->GetLogicalVolumeAtVertex())
    st = step->GetTrack()->GetLogicalVolumeAtVertex()->GetName();
  strcpy(mInteractionProductionVolume, st.c_str());
  // mInteractionProductionProcessTrack
  st = "";
  if(step->GetTrack()->GetCreatorProcess() )
    st =  step->GetTrack()->GetCreatorProcess()->GetProcessName();
  strcpy(mInteractionProductionProcessTrack, st.c_str());
  // mInteractionProductionProcessStep
  st = process->GetProcessName();
  strcpy(mInteractionProductionProcessStep, st.c_str());
  // mInteractionVolume
  st = step->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume()->GetName();
  strcpy(mInteractionVolume, st.c_str());
  // mInteractionVolume
  st = step->GetPreStepPoint()->GetMaterial()->GetName();
  strcpy(mInteractionMaterial, st.c_str());


  GateScatterOrderTrackInformation * info = dynamic_cast<GateScatterOrderTrackInformation *>(step->GetTrack()->GetUserInformation());

  // d and p are in World coordinates and they must be in CT coordinates
  G4ThreeVector d = m_WorldToCT.TransformAxis(mInteractionDirection);
  G4ThreeVector p = m_WorldToCT.TransformPoint(mInteractionPosition);

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
  const G4Element* elm = model->SelectRandomAtom(couple,particle,mInteractionEnergy);
  mInteractionZ = elm->GetZ();

  if(process->GetProcessName() == G4String("Compton") || process->GetProcessName() == G4String("compt")) {
    mComptonProbe.Start();
    GeometryType::Pointer oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddReg23Projection(point,
                                        mDetectorPosition,
                                        mDetectorRowVector,
                                        mDetectorColVector);
    mComptonProjector->SetInput(mComptonImage);
    mComptonProjector->SetGeometry( oneProjGeometry.GetPointer() );
    mComptonProjector->GetProjectedValueAccumulation().SetEnergyZAndWeight( mInteractionEnergy, mInteractionZ, mInteractionWeight );
    mComptonProjector->GetProjectedValueAccumulation().SetDirection( direction );
    TRY_AND_EXIT_ON_ITK_EXCEPTION(mComptonProjector->Update());
    mComptonImage = mComptonProjector->GetOutput();
    mComptonImage->DisconnectPipeline();
    mComptonProbe.Stop();
    mInteractionTotalContribution = mComptonProjector->GetProjectedValueAccumulation().GetIntegralOverDetectorAndReset();
    if(mPhaseSpaceFile) mPhaseSpace->Fill();

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
    mInteractionWeight = mEnergyResponseDetector(mInteractionEnergy)*mInteractionWeight;
    GeometryType::Pointer oneProjGeometry = GeometryType::New();
    oneProjGeometry->AddReg23Projection(point,
                                        mDetectorPosition,
                                        mDetectorRowVector,
                                        mDetectorColVector);
    mRayleighProjector->SetInput(mRayleighImage);
    mRayleighProjector->SetGeometry( oneProjGeometry.GetPointer() );
    mRayleighProjector->GetProjectedValueAccumulation().SetEnergyZAndWeight( mInteractionEnergy, mInteractionZ, mInteractionWeight );
    mRayleighProjector->GetProjectedValueAccumulation().SetDirection( direction );
    TRY_AND_EXIT_ON_ITK_EXCEPTION(mRayleighProjector->Update());
    mRayleighImage = mRayleighProjector->GetOutput();
    mRayleighImage->DisconnectPipeline();
    mRayleighProbe.Stop();
    mInteractionTotalContribution = mRayleighProjector->GetProjectedValueAccumulation().GetIntegralOverDetectorAndReset();
    if(mPhaseSpaceFile) mPhaseSpace->Fill();

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
    VectorType directionSecondary;

    for(unsigned int i = 0; i<(*list).size(); i++) {
      nameSecondary = (*list)[i]->GetDefinition()->GetParticleName();

      // Check if photon has been emitted
      if(nameSecondary==G4String("gamma")) {

        GateScatterOrderTrackInformation * infoSecondary = dynamic_cast<GateScatterOrderTrackInformation *>((*list)[i]->GetUserInformation());

        // Update direction and energy for secondary photon
        mInteractionEnergy = (*list)[i]->GetKineticEnergy();
        mInteractionWeight = ((*list)[i]->GetWeight())*mEnergyResponseDetector(mInteractionEnergy);
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
        mFluorescenceProjector->GetProjectedValueAccumulation().SetEnergyAndWeight( mInteractionEnergy, mInteractionWeight );
        TRY_AND_EXIT_ON_ITK_EXCEPTION(mFluorescenceProjector->Update());
        mFluorescenceImage = mFluorescenceProjector->GetOutput();
        mFluorescenceImage->DisconnectPipeline();
        mFluorescenceProbe.Stop();
        mInteractionTotalContribution = mFluorescenceProjector->GetProjectedValueAccumulation().GetIntegralOverDetectorAndReset();
        if(mPhaseSpaceFile) mPhaseSpace->Fill();

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
    // Write the image of primary radiation accounting for the fluence of the
    // primary source.
    sprintf(filename, mPrimaryFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput( PrimaryFluenceWeighting(mPrimaryImage) );
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

    // In the attenuation, we assume that the whole detector is irradiated.
    // Otherwise we would have a division by 0.
    atten->SetInput1(mPrimaryImage);
    atten->SetInput2(mFlatFieldImage);
    atten->InPlaceOff();

    sprintf(filename, mAttenuationFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(atten->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
  }

  if(mFlatFieldFilename != "") {
    // Write the image of the flat field accounting for the fluence of the
    // primary source.
    sprintf(filename, mFlatFieldFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput( PrimaryFluenceWeighting(mFlatFieldImage) );
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
  }

  if(mComptonFilename != "") {
    sprintf(filename, mComptonFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(mComptonImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());

    for(unsigned int k = 1; k<mComptonPerOrderImages.size(); k++)
    {
      sprintf(filename, "output/compton%04d_order%02d.mha", rID, k);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mComptonPerOrderImages[k]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }
  }

  if(mRayleighFilename != "") {
    sprintf(filename, mRayleighFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(mRayleighImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());

    for(unsigned int k = 1; k<mRayleighPerOrderImages.size(); k++)
    {
      sprintf(filename, "output/rayleigh%04d_order%02d.mha", rID, k);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mRayleighPerOrderImages[k]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }
  }

  if(mFluorescenceFilename != "") {
    sprintf(filename, mFluorescenceFilename.c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(mFluorescenceImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());

    for(unsigned int k = 1; k<mFluorescencePerOrderImages.size(); k++)
    {
      sprintf(filename, "output/fluorescence%04d_order%02d.mha", rID, k);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mFluorescencePerOrderImages[k]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }
  }

  if(mSecondaryFilename != "" ||
     mIsSecondarySquaredImageEnabled ||
     mIsSecondaryUncertaintyImageEnabled ||
     mTotalFilename != "") {
    // The secondary image contains all calculated scatterings
    // (Compton, Rayleigh and/or Fluorescence)
    // Create projections image
    InputImageType::Pointer mSecondaryImage;
    mSecondaryImage = CreateVoidProjectionImage();

    // Add Image Filter used to sum the different figures obtained on each process
    typedef itk::AddImageFilter <OutputImageType, OutputImageType, OutputImageType> AddImageFilterType;
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    addFilter->InPlaceOn();
    addFilter->SetInput1(mSecondaryImage);

    // Compton
    addFilter->SetInput2(mComptonImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
    mSecondaryImage = addFilter->GetOutput();

    // Rayleigh
    mSecondaryImage->DisconnectPipeline();
    addFilter->SetInput1(mSecondaryImage);
    addFilter->SetInput2(mRayleighImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
    mSecondaryImage = addFilter->GetOutput();

    // Fluorescence
    mSecondaryImage->DisconnectPipeline();
    addFilter->SetInput1(mSecondaryImage);
    addFilter->SetInput2(mFluorescenceImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
    mSecondaryImage = addFilter->GetOutput();

    // Write scatter image
    if(mSecondaryFilename != "") {
      sprintf(filename, mSecondaryFilename.c_str(), rID);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mSecondaryImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());

      // Write scatter squared image
      if(mIsSecondarySquaredImageEnabled) {
        imgWriter->SetFileName(G4String(removeExtension(filename)) +
                               "-Squared." +
                               G4String(getExtension(filename)));
        imgWriter->SetInput(mSecondarySquared);
        TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
      }

      // Write scatter uncertainty image
      if(mIsSecondaryUncertaintyImageEnabled) {
        //Attenuation Functor -> atten
        typedef itk::BinaryFunctorImageFilter< InputImageType, InputImageType, InputImageType,
                                               GateHybridForcedDetectionFunctor::Chetty<InputImageType::PixelType> > ChettyType;
        ChettyType::Pointer chetty = ChettyType::New();
        chetty->GetFunctor().SetN(mNumberOfEventsInRun);

        // In the attenuation, we assume that the whole detector is irradiated.
        // Otherwise we would have a division by 0.
        chetty->SetInput1(mSecondaryImage);
        chetty->SetInput2(mSecondarySquared);
        chetty->InPlaceOff();

        imgWriter->SetFileName(G4String(removeExtension(filename)) +
                               "-Uncertainty." +
                               G4String(getExtension(filename)));
        imgWriter->SetInput(chetty->GetOutput());

        TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
      }
    }

    if(mTotalFilename != "") {
      // Primary
      mSecondaryImage->DisconnectPipeline();
      addFilter->SetInput1(mSecondaryImage);
      addFilter->SetInput2(mPrimaryImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION( addFilter->Update() );
      mSecondaryImage = addFilter->GetOutput();

      // Write Total Image
      sprintf(filename, mTotalFilename.c_str(), rID);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mSecondaryImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }
  }

  if(mSingleInteractionFilename!="") {
    imgWriter->SetFileName(mSingleInteractionFilename);
    imgWriter->SetInput(mSingleInteractionImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
  }

  if(mPhaseSpaceFile)
    mPhaseSpace->GetCurrentFile()->Write();
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
  if(src->GetPosDist()->GetPosDisType() == "Point")
    s = src->GetPosDist()->GetCentreCoords(); // point

  m_SourceToCT = sourceToWorld *  m_WorldToCT;
  s = m_SourceToCT.TransformPoint(s);

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

//-----------------------------------------------------------------------------
void
GateHybridForcedDetectionActor::CreateWaterLUT(const std::vector<double> &energyList,
                                               const std::vector<double> &energyWeightList)
{
  // Get the list of involved processes (Rayleigh, Compton, PhotoElectric)
  G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle("gamma");
  G4ProcessVector* plist = particle->GetProcessManager()->GetProcessList();
  std::vector<G4String> processNameVector;
  for (G4int j = 0; j < plist->size(); j++) {
    G4ProcessType type = (*plist)[j]->GetProcessType();
    std::string name = (*plist)[j]->GetProcessName();
    if ((type == fElectromagnetic) && (name != "msc")) {
      processNameVector.push_back(name);
    }
  }

  // Create mu data
  std::vector<double> mu(energyList.size(), 0.);
  G4Material * mat = G4NistManager::Instance()->FindOrBuildMaterial("G4_WATER");
  double energyWeightDetRespSum = 0.;
  std::vector<double> energyWeightDetResp(energyList.size(), 0.);
  for(unsigned int e=0; e<energyList.size(); e++) {
    for (unsigned int j = 0; j < processNameVector.size(); j++) {
      mu[e] +=
          mEMCalculator->ComputeCrossSectionPerVolume(energyList[e],
                                                      "gamma",
                                                      processNameVector[j],
                                                      mat->GetName());
    }
    energyWeightDetResp[e] = energyWeightList[e] * mEnergyResponseDetector(energyList[e]);
    energyWeightDetRespSum += energyWeightDetResp[e];
  }
  for(unsigned int e=0; e<energyList.size(); e++)
    energyWeightDetResp[e] /= energyWeightDetRespSum;

  const double spacing = 0.1;
  unsigned int n = (unsigned int)floor(1000./spacing);
  typedef itk::Image<double, 1> LUTType;
  LUTType::RegionType region;
  region.SetSize(0, n);
  LUTType::Pointer lengthToAttenuationLUT = LUTType::New();
  lengthToAttenuationLUT->SetRegions(region);
  lengthToAttenuationLUT->Allocate();
  lengthToAttenuationLUT->SetSpacing(&spacing);
  itk::ImageRegionIterator<LUTType> it(lengthToAttenuationLUT, region);
  double prev = itk::NumericTraits<double>::NonpositiveMin();
  double deltaMin = itk::NumericTraits<double>::max();
  for(unsigned int i=0; i<n; i++, ++it) {
    const double length = i*spacing*CLHEP::mm;
    double value = 0.;
    for(unsigned int e=0; e<energyList.size(); e++) {
      value += energyWeightDetResp[e] * exp(-1.*mu[e]*length);
    }
    value = -1. * log(value);
    it.Set(value);
    deltaMin = std::min(deltaMin, value-prev);
    prev = value;
  }
  --it;

  // Take the inverse
  LUTType::Pointer attenuationToLengthLUT = LUTType::New();
  n = (unsigned int)floor(it.Get()/deltaMin);
  region.SetSize(0, n);
  attenuationToLengthLUT->SetRegions(region);
  attenuationToLengthLUT->Allocate();
  attenuationToLengthLUT->SetSpacing(&deltaMin);

  itk::ImageRegionIterator<LUTType> itInv(attenuationToLengthLUT, region);
  itInv.Set(0.);
  ++itInv;
  double currAtt = deltaMin;
  double lengthLeft = 0.;
  double attLeft = 0.;
  it.GoToBegin();
  ++it;
  double attRight = it.Get();
  while(!itInv.IsAtEnd()) {
    while(attRight<currAtt) {
      attLeft = it.Get();
      lengthLeft += spacing;
      ++it;
      attRight = it.Get();
    }
    itInv.Set( ((currAtt-attLeft) * (lengthLeft+spacing) + (attRight-currAtt) * (lengthLeft)) / (attRight-attLeft) );

    // Next
    ++itInv;
    currAtt += deltaMin;
  }

  // Write result
  typedef itk::ImageFileWriter<LUTType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(attenuationToLengthLUT);
  writer->SetFileName(mWaterLUTFilename);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(writer->Update());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHybridForcedDetectionActor::InputImageType::Pointer
GateHybridForcedDetectionActor::PrimaryFluenceWeighting(const InputImageType::Pointer input)
{
  InputImageType::Pointer output = input;
  if(mSource->GetPosDist()->GetPosDisType() == "Point") {
    GateWarning("Primary fluence is not accounted for with a Point source distribution");
  }
  else if(mSource->GetPosDist()->GetPosDisType() == "Plane") {
    // Check plane source projection probability
    G4ThreeVector sourceCorner1 = mSource->GetPosDist()->GetCentreCoords();
    sourceCorner1[0] -= mSource->GetPosDist()->GetHalfX();
    sourceCorner1[1] -= mSource->GetPosDist()->GetHalfY();
    sourceCorner1 = m_SourceToCT.TransformPoint(sourceCorner1);

    G4ThreeVector sourceCorner2 = mSource->GetPosDist()->GetCentreCoords();
    sourceCorner2[0] += mSource->GetPosDist()->GetHalfX();
    sourceCorner2[1] += mSource->GetPosDist()->GetHalfY();
    sourceCorner2 = m_SourceToCT.TransformPoint(sourceCorner2);

    // Compute source plane corner positions in homogeneous coordinates
    itk::Vector<double, 4> corner1Hom, corner2Hom;
    corner1Hom[0] = sourceCorner1[0];
    corner1Hom[1] = sourceCorner1[1];
    corner1Hom[2] = sourceCorner1[2];
    corner1Hom[3] = 1.;
    corner2Hom[0] = sourceCorner2[0];
    corner2Hom[1] = sourceCorner2[1];
    corner2Hom[2] = sourceCorner2[2];
    corner2Hom[3] = 1.;

    // Project onto detector
    itk::Vector<double, 3> corner1ProjHom, corner2ProjHom;
    corner1ProjHom.SetVnlVector(mGeometry->GetMatrices().back().GetVnlMatrix() * corner1Hom.GetVnlVector());
    corner2ProjHom.SetVnlVector(mGeometry->GetMatrices().back().GetVnlMatrix() * corner2Hom.GetVnlVector());
    corner1ProjHom /= corner1ProjHom[2];
    corner2ProjHom /= corner2ProjHom[2];

    // Convert to non homogeneous coordinates
    InputImageType::PointType corner1Proj, corner2Proj;
    corner1Proj[0] = corner1ProjHom[0];
    corner1Proj[1] = corner1ProjHom[1];
    corner1Proj[2] = 0.;
    corner2Proj[0] = corner2ProjHom[0];
    corner2Proj[1] = corner2ProjHom[1];
    corner2Proj[2] = 0.;

    // Convert to projection indices
    itk::ContinuousIndex<double, 3> corner1Idx, corner2Idx;
    input->TransformPhysicalPointToContinuousIndex<double>(corner1Proj, corner1Idx);
    input->TransformPhysicalPointToContinuousIndex<double>(corner2Proj, corner2Idx);
    if(corner1Idx[0]>corner2Idx[0])
      std::swap(corner1Idx[0], corner2Idx[0]);
    if(corner1Idx[1]>corner2Idx[1])
      std::swap(corner1Idx[1], corner2Idx[1]);

    // Create copy of image normalized by the number of particles and the ratio
    // between source size on the detector and the detector size in pixels
    typedef itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyType;
    MultiplyType::Pointer mult = MultiplyType::New();
    mult->SetInput(input);
    InputImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();
    mult->SetConstant(mNumberOfEventsInRun /
                      ((corner2Idx[1] - corner1Idx[1]) *
                       (corner2Idx[0] - corner1Idx[0])));
    mult->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(mult->Update());

    // Multiply by pixel fraction
    itk::ImageRegionIterator<InputImageType> it(mult->GetOutput(),
                                                mult->GetOutput()->GetLargestPossibleRegion());
    InputImageType::IndexType idx = input->GetLargestPossibleRegion().GetIndex();
    for(unsigned int j=idx[1]; j<size[1]; j++)
      for(unsigned int i=idx[0]; i<size[0]; i++) {
        double maxInfX = std::max<double>(i-0.5, corner1Idx[0]);
        double maxInfY = std::max<double>(j-0.5, corner1Idx[1]);
        double minSupX = std::min<double>(i+0.5, corner2Idx[0]);
        double minSupY = std::min<double>(j+0.5, corner2Idx[1]);
        it.Set(it.Get() *
               std::max<double>(0., minSupX-maxInfX) *
               std::max<double>(0., minSupY-maxInfY));
        ++it;
      }
    output = mult->GetOutput();
  }
  return output;
}
//-----------------------------------------------------------------------------
#endif
