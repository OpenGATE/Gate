/*----------------------
 GATE version name: gate_v6

 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See GATE/LICENSE.txt for further details
 ----------------------*/

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

/* Gate */
#include "GateFixedForcedDetectionActor.hh"
#include "GateMiscFunctions.hh"
#include "GateFixedForcedDetectionFunctors.hh"

/* G4 */
#include <G4Event.hh>
#include <G4MaterialTable.hh>
#include <G4ParticleTable.hh>
#include <G4VEmProcess.hh>
#include <G4TransportationManager.hh>
#include <G4LivermoreComptonModel.hh>
#include <G4SteppingManager.hh>
#include <G4NistManager.hh>

/* rtk */
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>
#include <rtkMacro.h>

/* itk */
#include <itkChangeInformationImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkImageFileWriter.h>
#include <itkBinaryFunctorImageFilter.h>
#include <itkAddImageFilter.h>
#include <itksys/SystemTools.hxx>

/*  Constructors */
GateFixedForcedDetectionActor::GateFixedForcedDetectionActor(G4String name, G4int depth) :
    GateVActor(name, depth),
    mIsSecondarySquaredImageEnabled(false),
    mIsSecondaryUncertaintyImageEnabled(false),
    mNoisePrimary(0),
    mInputRTKGeometryFilename(""),
    mEnergyResolvedBinSize(0),
    mSourceType("plane"),
    mGeneratePhotons(false)
  {
  GateDebugMessageInc("Actor",4,"GateFixedForcedDetectionActor() -- begin"<<G4endl);
  pActorMessenger = new GateFixedForcedDetectionActorMessenger(this);
  mDetectorResolution[0] = mDetectorResolution[1] = mDetectorResolution[2] = 1;
  GateDebugMessageDec("Actor",4,"GateFixedForcedDetectionActor() -- end"<<G4endl);

  mMapProcessNameWithType["Compton"] = COMPTON;
  mMapProcessNameWithType["compt"] = COMPTON;
  mMapProcessNameWithType["RayleighScattering"] = RAYLEIGH;
  mMapProcessNameWithType["Rayl"] = RAYLEIGH;
  mMapProcessNameWithType["PhotoElectric"] = PHOTOELECTRIC;
  mMapProcessNameWithType["phot"] = PHOTOELECTRIC;
  mMapProcessNameWithType["IsotropicPrimary"] = ISOTROPICPRIMARY;
  mMapTypeWithProcessName[COMPTON] = "Compton";
  mMapTypeWithProcessName[RAYLEIGH] = "Rayleigh";
  mMapTypeWithProcessName[PHOTOELECTRIC] = "PhotoElectric";
  mMapTypeWithProcessName[ISOTROPICPRIMARY] = "IsotropicPrimary";
  }

/* Destructor */
GateFixedForcedDetectionActor::~GateFixedForcedDetectionActor()
  {
  delete pActorMessenger;
  }

/* Construct */
void GateFixedForcedDetectionActor::Construct()
  {
  GateVActor::Construct();
  /*  Callbacks */
  EnableBeginOfRunAction(true);
  EnableEndOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(true);
  /*   EnablePreUserTrackingAction(true); */
  EnableUserSteppingAction(true);
  ResetData();
  mEMCalculator = new G4EmCalculator;

  CreatePhaseSpace(mPhaseSpaceFilename, mPhaseSpaceFile, mPhaseSpace);
  }

void GateFixedForcedDetectionActor::TestSource(GateSourceMgr * sm)
  {
  if (sm->GetNumberOfSources() == 0)
    {
    GateError("Error: no source set.");
    }
  if (sm->GetNumberOfSources() != 1)
    {
    GateWarning("Several sources found, we consider the first one.");
    }
  mSource = sm->GetSource(0);

  if (mSourceType == "")
    {
    GateError("Error: type of source not defined.");
    }
  /* Check Point: dosen't work if rot1 != (1,0,0) or  rot2 != (0,1,0) */
  if (mSourceType == "plane" || mSourceType == "point")
    {
    if (mSource->getRotX().getX() != 1
        || mSource->getRotX().getY() != 0
        || mSource->getRotX().getZ() != 0
        || mSource->getRotY().getX() != 0
        || mSource->getRotY().getY() != 1
        || mSource->getRotY().getZ() != 0)
      {
      GateError("Error: forced detection only supports plane source without rotations");
      }
    if (mSource->GetPosDist()->GetPosDisType() == "Point")
      {
      if (mSource->GetAngDist()->GetDistType() != "iso")
        {
        GateError("Error: forced detection only supports iso distributions with point source.");
        }
      }
    else if (mSource->GetPosDist()->GetPosDisType() == "Plane")
      {
      if (mSource->GetAngDist()->GetDistType() != "focused")
        {
        GateError("Error: forced detection only supports focused distributions for plane sources.");
        }
      }
    }
  else if (mSourceType == "isotropic")
    {

    }
  else
    {
    GateError("Error: forced detection only supports point,plane or isotropic distributions.");
    }
  }

void GateFixedForcedDetectionActor::GetEnergyList(std::vector<double> & energyList,
                                                  std::vector<double> & energyWeightList)
  {
  mMaxPrimaryEnergy = 0.;
  mMinPrimaryEnergy = itk::NumericTraits<double>::max();
  G4String energyDistributionType = mSource->GetEneDist()->GetEnergyDisType();
  if (energyDistributionType == "Mono")
    {
    energyList.push_back(mSource->GetEneDist()->GetMonoEnergy());
    energyWeightList.push_back(mEnergyResponseDetector(energyList.back()));
    mMaxPrimaryEnergy = std::max(mMaxPrimaryEnergy, energyList.back());
    mMinPrimaryEnergy = std::min(mMinPrimaryEnergy, energyList.back());
    }
  else if (energyDistributionType == "User")
    { /* histo */
    G4PhysicsOrderedFreeVector energyHistogram = mSource->GetEneDist()->GetUserDefinedEnergyHisto();
    double weightSum = 0.;
    double energy = 0;
    for (unsigned int i = 0; i < energyHistogram.GetVectorLength(); i++)
      {
      energy = energyHistogram.Energy(i);
      energyList.push_back(energy);
      energyWeightList.push_back(energyHistogram.Value(energy));
      weightSum += energyWeightList.back();
      /* noise is desactivated */
      if (mNoisePrimary == 0)
        {
        energyWeightList.back() *= mEnergyResponseDetector(energyList.back());
        }
      mMaxPrimaryEnergy = std::max(mMaxPrimaryEnergy, energyList.back());
      mMinPrimaryEnergy = std::min(mMinPrimaryEnergy, energyList.back());
      }
    for (unsigned int i = 0; i < energyHistogram.GetVectorLength(); i++)
      {
      energyWeightList[i] /= weightSum;
      }
    }
  else
    {
    GateError("Error: source type should be Mono or User.");
    }
  }

GateVImageVolume* GateFixedForcedDetectionActor::SearchForVoxelisedVolume()
  {
  GateVImageVolume* gate_image_volume = NULL;
  for (std::map<G4String, GateVVolume*>::const_iterator it = GateObjectStore::GetInstance()->begin();
      it != GateObjectStore::GetInstance()->end(); it++)
    {
    if (dynamic_cast<GateVImageVolume*>(it->second))
      {
      /* Loop on volumes to check that they contain world material only */
      if (it->second->GetMaterialName() != mVolume->GetMaterialName()
          && it->second->GetMaterialName() != "")
        {
        GateError("Error: additionnal volumes should share the world's material -> "<<mVolume->GetMaterialName()<<" : ("<<it->first << " -> " << it->second->GetMaterialName()<<")");
        }
      if (gate_image_volume != NULL)
        {
        GateError("Error: forced detection manages only one voxelised volume.");
        }
      else
        {
        gate_image_volume = dynamic_cast<GateVImageVolume*>(it->second);
        }
      }
    }
  if (!gate_image_volume)
    {
    GateError("Error: you need one voxelized volume in your scene.");
    }
  return gate_image_volume;
  }

void GateFixedForcedDetectionActor::CreateProjectionImages()
  {
  mPrimaryImage = CreateVoidProjectionImage();
  for (unsigned int i = 0; i < PRIMARY; i++)
    {
    const ProcessType pt = ProcessType(i);
    mDoFFDForThisProcess[pt] = (mProcessImageFilenames[pt] != ""
                                || mTotalFilename != ""
                                || mSecondaryFilename != "");
    mProcessImage[pt] = CreateVoidProjectionImage();
    mSquaredImage[pt] = CreateVoidProjectionImage();
    mPerOrderImages[pt].clear();
    }
  mSecondarySquaredImage = CreateVoidProjectionImage();
  }

/* Callback Begin of Run */
void GateFixedForcedDetectionActor::BeginOfRunAction(const G4Run*r)
  {
  GateVActor::BeginOfRunAction(r);
  mNumberOfEventsInRun = 0;
  /* Get information on the source */
  GateSourceMgr * sourceMessenger = GateSourceMgr::GetInstance();
  TestSource(sourceMessenger);
  /* Read the response detector curve from an external file */
  mEnergyResponseDetector.ReadResponseDetectorFile(mResponseFilename);
  /* Create list of energies */
  std::vector<double> energyList;
  std::vector<double> energyWeightList;
  GetEnergyList(energyList, energyWeightList);
  /* Search for voxelized volume. If more than one, crash (yet). */
  GateVImageVolume* gateImageVolume = SearchForVoxelisedVolume();
  /* Conversion of CT to ITK and to int values */
  mGateVolumeImage = ConvertGateImageToITKImage(gateImageVolume);
  /* Create projection images */
  CreateProjectionImages();
  /* Set geometry from RTK geometry file */
  if (mInputRTKGeometryFilename != "")
    {
    SetGeometryFromInputRTKGeometryFile(mSource, mDetector, gateImageVolume, r);
    }
  /* Create geometry and param of output image */
  ComputeGeometryInfoInImageCoordinateSystem(gateImageVolume,
                                             mDetector,
                                             mSource,
                                             mPrimarySourcePosition,
                                             mDetectorPosition,
                                             mDetectorRowVector,
                                             mDetectorColVector);

  /* There are two geometry objects. One stores all projection images
   (one per run) and the other contains the geometry of one projection
   image. */
  mGeometry->AddReg23Projection(mPrimarySourcePosition,
                                mDetectorPosition,
                                mDetectorRowVector,
                                mDetectorColVector);
  GeometryType::Pointer oneProjGeometry = GeometryType::New();
  oneProjGeometry->AddReg23Projection(mPrimarySourcePosition,
                                      mDetectorPosition,
                                      mDetectorRowVector,
                                      mDetectorColVector);

  /* Create primary projector and compute primary */
  unsigned int nPixOneSlice = mPrimaryImage->GetLargestPossibleRegion().GetNumberOfPixels()
                              / mPrimaryImage->GetLargestPossibleRegion().GetSize(2);
  mProcessTimeProbe[PRIMARY].Start();
  PreparePrimaryProjector(oneProjGeometry,
                          energyList,
                          energyWeightList,
                          gateImageVolume,
                          nPixOneSlice);

  /* Compute flat field if required */
  if (mAttenuationFilename != "" || mFlatFieldFilename != "")
    {
    ComputeFlatField(energyList, energyWeightList);
    }
  mProcessTimeProbe[PRIMARY].Stop();

  PrepareComptonProjector(gateImageVolume, nPixOneSlice, oneProjGeometry);
  PrepareRayleighProjector(gateImageVolume, nPixOneSlice, oneProjGeometry);
  PrepareFluorescenceProjector(gateImageVolume, nPixOneSlice, oneProjGeometry);
  PrepareIsotropicPrimaryProjector(gateImageVolume, nPixOneSlice, oneProjGeometry);

  if (mIsSecondarySquaredImageEnabled || mIsSecondaryUncertaintyImageEnabled)
    {
    for (unsigned int i = 0; i < PRIMARY; i++)
      mEventImage[ProcessType(i)] = CreateVoidProjectionImage();
    }
  }

void GateFixedForcedDetectionActor::ComputeFlatField(std::vector<double> & energyList,
                                                     std::vector<double> & energyWeightList)
  {

  /* Constant image source of 1x1x1 voxel of world material */
  typedef rtk::ConstantImageSource<InputImageType> ConstantImageSourceType;
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SizeType dim;
  ConstantImageSourceType::Pointer flatFieldSource = ConstantImageSourceType::New();
  origin[0] = 0.;
  origin[1] = 0.;
  origin[2] = 0.;
  dim[0] = 1;
  dim[1] = 1;
  dim[2] = 1;
  flatFieldSource->SetOrigin(origin);
  flatFieldSource->SetSpacing(mGateVolumeImage->GetSpacing());
  flatFieldSource->SetSize(dim);
  flatFieldSource->SetConstant(mPrimaryProjector->GetProjectedValueAccumulation().GetMaterialMuMap()->GetLargestPossibleRegion().GetSize()[0]
                               - 1);
  mFlatFieldImage = CreateVoidProjectionImage();
  mPrimaryProjector->SetInput(FirstSliceProjection(mFlatFieldImage));
  mPrimaryProjector->SetInput(1, flatFieldSource->GetOutput());
  /* Remove noise from I0. */
  if (mNoisePrimary != 0)
    {
    for (unsigned int i = 0; i < energyWeightList.size(); i++)
      {
      energyWeightList[i] *= mEnergyResponseDetector(energyList[i]);
      }
    mPrimaryProjector->GetProjectedValueAccumulation().SetNumberOfPrimaries(0.);
    }

  TRY_AND_EXIT_ON_ITK_EXCEPTION(mPrimaryProjector->Update());

  }

void GateFixedForcedDetectionActor::PreparePrimaryProjector(GeometryType::Pointer oneProjGeometry,
                                                            std::vector<double> & energyList,
                                                            std::vector<double> & energyWeightList,
                                                            GateVImageVolume* gate_image_volume,
                                                            unsigned int & nPixOneSlice)
  {
  mPrimaryProjector = PrimaryProjectionType::New();
  mPrimaryProjector->InPlaceOn();
  mPrimaryProjector->SetInput(FirstSliceProjection(mPrimaryImage));
  mPrimaryProjector->SetInput(1, mGateVolumeImage);
  mPrimaryProjector->SetGeometry(oneProjGeometry.GetPointer());
  mPrimaryProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mPrimaryImage,
                                                                             mDetectorRowVector,
                                                                             mDetectorColVector);
  mPrimaryProjector->GetProjectedValueAccumulation().SetVolumeSpacing(mGateVolumeImage->GetSpacing());
  mPrimaryProjector->GetProjectedValueAccumulation().SetInterpolationWeights(mPrimaryProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights());
  mPrimaryProjector->GetProjectedValueAccumulation().SetEnergyWeightList(&energyWeightList);
  mPrimaryProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                         energyList,
                                                                         gate_image_volume);
  mPrimaryProjector->GetProjectedValueAccumulation().Init(mPrimaryProjector->GetNumberOfThreads());
  mPrimaryProjector->GetProjectedValueAccumulation().SetNumberOfPrimaries(mNoisePrimary);
  mPrimaryProjector->GetProjectedValueAccumulation().SetResponseDetector(&mEnergyResponseDetector);
  mPrimaryProjector->GetProjectedValueAccumulation().SetEnergyResolvedParameters(mEnergyResolvedBinSize,
                                                                                 nPixOneSlice);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(mPrimaryProjector->Update());
  }

void GateFixedForcedDetectionActor::PrepareComptonProjector(GateVImageVolume* gate_image_volume,
                                                            unsigned int & nPixOneSlice,
                                                            GeometryType::Pointer oneProjGeometry)
  {
  mComptonProjector = ComptonProjectionType::New();
  mComptonProjector->InPlaceOn();
  mComptonProjector->SetInput(1, mGateVolumeImage);
  mComptonProjector->SetGeometry(oneProjGeometry.GetPointer());
  mComptonProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mProcessImage[COMPTON],
                                                                             mDetectorRowVector,
                                                                             mDetectorColVector);
  mComptonProjector->GetProjectedValueAccumulation().SetVolumeSpacing(mGateVolumeImage->GetSpacing());
  mComptonProjector->GetProjectedValueAccumulation().SetInterpolationWeights(mComptonProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights());
  mComptonProjector->GetProjectedValueAccumulation().SetResponseDetector(&mEnergyResponseDetector);
  mComptonProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                         1. * keV,
                                                                         mMaxPrimaryEnergy,
                                                                         gate_image_volume);
  mComptonProjector->GetProjectedValueAccumulation().Init(mComptonProjector->GetNumberOfThreads());
  mComptonProjector->GetProjectedValueAccumulation().SetEnergyResolvedParameters(mEnergyResolvedBinSize,
                                                                                 nPixOneSlice);
  }

void GateFixedForcedDetectionActor::PrepareRayleighProjector(GateVImageVolume* gate_image_volume,
                                                             unsigned int & nPixOneSlice,
                                                             GeometryType::Pointer oneProjGeometry)
  {
  mRayleighProjector = RayleighProjectionType::New();
  mRayleighProjector->InPlaceOn();
  mRayleighProjector->SetInput(1, mGateVolumeImage);
  mRayleighProjector->SetGeometry(oneProjGeometry.GetPointer());
  mRayleighProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mProcessImage[RAYLEIGH],
                                                                              mDetectorRowVector,
                                                                              mDetectorColVector);
  mRayleighProjector->GetProjectedValueAccumulation().SetVolumeSpacing(mGateVolumeImage->GetSpacing());
  mRayleighProjector->GetProjectedValueAccumulation().SetInterpolationWeights(mRayleighProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights());
  mRayleighProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                          1. * keV,
                                                                          mMaxPrimaryEnergy,
                                                                          gate_image_volume);
  mRayleighProjector->GetProjectedValueAccumulation().Init(mRayleighProjector->GetNumberOfThreads());
  mRayleighProjector->GetProjectedValueAccumulation().SetEnergyResolvedParameters(mEnergyResolvedBinSize,
                                                                                  nPixOneSlice);

  }

void GateFixedForcedDetectionActor::PrepareFluorescenceProjector(GateVImageVolume* gate_image_volume,
                                                                 unsigned int & nPixOneSlice,
                                                                 GeometryType::Pointer oneProjGeometry)
  {
  mFluorescenceProjector = FluorescenceProjectionType::New();
  mFluorescenceProjector->InPlaceOn();
  mFluorescenceProjector->SetInput(1, mGateVolumeImage);
  mFluorescenceProjector->SetGeometry(oneProjGeometry.GetPointer());
  mFluorescenceProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mProcessImage[PHOTOELECTRIC],
                                                                                  mDetectorRowVector,
                                                                                  mDetectorColVector);
  mFluorescenceProjector->GetProjectedValueAccumulation().SetVolumeSpacing(mGateVolumeImage->GetSpacing());
  mFluorescenceProjector->GetProjectedValueAccumulation().SetInterpolationWeights(mFluorescenceProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights());
  mFluorescenceProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                              1. * keV,
                                                                              mMaxPrimaryEnergy,
                                                                              gate_image_volume);
  mFluorescenceProjector->GetProjectedValueAccumulation().Init(mFluorescenceProjector->GetNumberOfThreads());
  mFluorescenceProjector->GetProjectedValueAccumulation().SetEnergyResolvedParameters(mEnergyResolvedBinSize,
                                                                                      nPixOneSlice);
  }

void GateFixedForcedDetectionActor::PrepareIsotropicPrimaryProjector(GateVImageVolume* gateImageVolume,
                                                                     unsigned int & nPixOneSlice,
                                                                     GeometryType::Pointer oneProjGeometry)
  {
  mIsotropicPrimaryProjector = IsotropicPrimaryProjectionType::New();
  mIsotropicPrimaryProjector->GetProjectedValueAccumulation().setGeneratePhotons(mGeneratePhotons);
  mIsotropicPrimaryProjector->GetProjectedValueAccumulation().PreparePhotonsList(mIsotropicPrimaryProjector->GetNumberOfThreads());
  mIsotropicPrimaryProjector->InPlaceOn();
  mIsotropicPrimaryProjector->SetInput(1, mGateVolumeImage);
  mIsotropicPrimaryProjector->SetGeometry(oneProjGeometry.GetPointer());
  mIsotropicPrimaryProjector->GetProjectedValueAccumulation().SetSolidAngleParameters(mProcessImage[ISOTROPICPRIMARY],
                                                                                      mDetectorRowVector,
                                                                                      mDetectorColVector);
  mIsotropicPrimaryProjector->GetProjectedValueAccumulation().SetVolumeSpacing(mGateVolumeImage->GetSpacing());
  mIsotropicPrimaryProjector->GetProjectedValueAccumulation().SetInterpolationWeights(mIsotropicPrimaryProjector->GetInterpolationWeightMultiplication().GetInterpolationWeights());
  mIsotropicPrimaryProjector->GetProjectedValueAccumulation().CreateMaterialMuMap(mEMCalculator,
                                                                                  1. * keV,
                                                                                  mMaxPrimaryEnergy,
                                                                                  gateImageVolume);

  mIsotropicPrimaryProjector->GetProjectedValueAccumulation().Init(mIsotropicPrimaryProjector->GetNumberOfThreads());

  mIsotropicPrimaryProjector->GetProjectedValueAccumulation().SetEnergyResolvedParameters(mEnergyResolvedBinSize,
                                                                                          nPixOneSlice);
  }

void GateFixedForcedDetectionActor::BeginOfEventAction(const G4Event *e)
  {
  mNumberOfEventsInRun++;
  if (e)
    {
    mInteractionOrder = 0;
    }

  if (mIsSecondarySquaredImageEnabled || mIsSecondaryUncertaintyImageEnabled)
    {
    /* The event contribution are put in new images which at this point are in the
     mEventComptonImage / mEventRayleighImage / mEventFluorescenceImage. We therefore
     swap the two and they will be swapped back in EndOfEventAction. */
    for (unsigned int i = 0; i < PRIMARY; i++)
      {
      std::swap(mEventImage[ProcessType(i)], mProcessImage[ProcessType(i)]);
      /*  Make sure the time stamps of the mEvent images are more recent to detect if one
       image has been modified during the event. */
      mEventImage[ProcessType(i)]->Modified();
      }
    }
  }

void GateFixedForcedDetectionActor::EndOfEventAction(const G4Event *e)
  {
  if (mIsSecondarySquaredImageEnabled || mIsSecondaryUncertaintyImageEnabled)
    {
    typedef itk::AddImageFilter<OutputImageType, OutputImageType, OutputImageType> AddImageFilterType;
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
    typedef itk::MultiplyImageFilter<OutputImageType, OutputImageType, OutputImageType> MultiplyImageFilterType;
    MultiplyImageFilterType::Pointer multFilter = MultiplyImageFilterType::New();

    /* First: accumulate contribution to event, square and add to total squared */
    InputImageType::Pointer totalContribEvent(NULL);
    for (unsigned int i = 0; i < PRIMARY; i++)
      {
      if (mEventImage[ProcessType(i)]->GetTimeStamp()
          < mProcessImage[ProcessType(i)]->GetTimeStamp())
        {
        /* First: accumulate contribution to event, square and add to total squared */
        multFilter->SetInput1(mProcessImage[ProcessType(i)]);
        multFilter->SetInput2(mProcessImage[ProcessType(i)]);
        multFilter->InPlaceOff();
        addFilter->SetInput1(mSquaredImage[ProcessType(i)]);
        addFilter->SetInput2(multFilter->GetOutput());
        addFilter->InPlaceOff();
        TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());
        mSquaredImage[ProcessType(i)] = addFilter->GetOutput();
        mSquaredImage[ProcessType(i)]->DisconnectPipeline();

        /* Second: accumulate in total event for the global secondary image */
        if (totalContribEvent.GetPointer())
          {
          addFilter->SetInput1(totalContribEvent);
          addFilter->SetInput2(mProcessImage[ProcessType(i)]);
          TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());
          totalContribEvent = addFilter->GetOutput();
          totalContribEvent->DisconnectPipeline();
          }
        else
          {
          totalContribEvent = mProcessImage[ProcessType(i)];
          }
        }
      }

    if (totalContribEvent.GetPointer())
      {
      multFilter->SetInput1(totalContribEvent);
      multFilter->SetInput2(totalContribEvent);
      multFilter->InPlaceOff();
      addFilter->SetInput1(mSecondarySquaredImage);
      addFilter->SetInput2(multFilter->GetOutput());
      TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());
      mSecondarySquaredImage = addFilter->GetOutput();
      mSecondarySquaredImage->DisconnectPipeline();
      }

    for (unsigned int i = 0; i < PRIMARY; i++)
      {
      if (mEventImage[ProcessType(i)]->GetTimeStamp()
          < mProcessImage[ProcessType(i)]->GetTimeStamp())
        {
        /* Accumulate non squared images and reset mEvent images */
        addFilter->SetInput1(mEventImage[ProcessType(i)]);
        addFilter->SetInput2(mProcessImage[ProcessType(i)]);
        TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());
        mProcessImage[ProcessType(i)] = addFilter->GetOutput();
        mProcessImage[ProcessType(i)]->DisconnectPipeline();
        mEventImage[ProcessType(i)] = CreateVoidProjectionImage();
        }
      else
        {
        std::swap(mEventImage[ProcessType(i)], mProcessImage[ProcessType(i)]);
        }
      }
    }
  if (e)
    {
    GateVActor::EndOfEventAction(e);
    }
  }

/* Callbacks */
void GateFixedForcedDetectionActor::UserSteppingAction(const GateVVolume * v, const G4Step * step)
  {
  GateVActor::UserSteppingAction(v, step);
  /* Get interaction point from step
   Retrieve :
   - type of limiting process (Compton Rayleigh Fluorescence)
   - coordinate of interaction, convert if needed into world coordinate system
   - Get Energy
   - -> generate adequate forward projections towards detector
   */

  /* We are only interested in EM processes. One should check post-step to know
   what is going to happen, but pre-step is used afterward to get direction
   and position. */
  const G4VProcess *pr = step->GetPostStepPoint()->GetProcessDefinedStep();
  const G4VEmProcess *process = dynamic_cast<const G4VEmProcess*>(pr);
  if (!process)
    {
    /* See if we have to place it in BeginOfEvent */
    if (mSourceType == "isotropic"
        && GateRunManager::GetRunManager()->GetCurrentEvent()->GetPrimaryVertex()->GetPosition()
           == step->GetPreStepPoint()->GetPosition())
      {

      ForceDetectionOfInteraction(GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID(),
                                  G4String("IsotropicPrimary"),
                                  step->GetPreStepPoint()->GetPosition(),
                                  step->GetPreStepPoint()->GetMomentumDirection(),
                                  step->GetPreStepPoint()->GetKineticEnergy(),
                                  step->GetPreStepPoint()->GetWeight(),
                                  0,
                                  step->GetPostStepPoint()->GetProperTime());
      /* Kill photons going outside of phantom */
      /* For now, put weight=0 but see if other method is more appropriate and check secondaries */
      step->GetPostStepPoint()->SetWeight(0.0);

      }

    return;
    }
  /* FIXME: do we prefer this solution or computing the scattering function for the material? */
  const G4MaterialCutsCouple *couple = step->GetPreStepPoint()->GetMaterialCutsCouple();
  const G4ParticleDefinition *particle = step->GetTrack()->GetParticleDefinition();
#if G4VERSION_MAJOR<10 && G4VERSION_MINOR==5
  G4VEmModel* model = const_cast<G4VEmProcess*>(process)->Model();
#else
  G4VEmModel* model = const_cast<G4VEmProcess*>(process)->EmModel();
#endif
  const G4Element* elm = model->SelectRandomAtom(couple,
                                                 particle,
                                                 step->GetPreStepPoint()->GetKineticEnergy());

  if (process->GetProcessName() == G4String("PhotoElectric")
      || process->GetProcessName() == G4String("phot"))
    {
    /* List of secondary particles */
    const G4TrackVector * list = step->GetSecondary();
    for (unsigned int i = 0; i < (*list).size(); i++)
      {
      G4String nameSecondary = (*list)[i]->GetDefinition()->GetParticleName();
      /* Check if photon has been emitted */
      if (nameSecondary == G4String("gamma"))
        {
        ForceDetectionOfInteraction(GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID(),
                                    process->GetProcessName(),
                                    step->GetPostStepPoint()->GetPosition(),
                                    (*list)[i]->GetMomentumDirection(),
                                    (*list)[i]->GetKineticEnergy(),
                                    (*list)[i]->GetWeight(),
                                    elm->GetZ());
        }
      }
    }
  else
    {
    ForceDetectionOfInteraction(GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID(),
                                process->GetProcessName(),
                                step->GetPostStepPoint()->GetPosition(),
                                step->GetPreStepPoint()->GetMomentumDirection(),
                                step->GetPreStepPoint()->GetKineticEnergy(),
                                step->GetPostStepPoint()->GetWeight(),
                                elm->GetZ());
    }

  }

/* Save data */
void GateFixedForcedDetectionActor::ForceDetectionOfInteraction(G4int eventID,
                                                                G4String processName,
                                                                G4ThreeVector interactionPosition,
                                                                G4ThreeVector interactionDirection,
                                                                double energy,
                                                                double weight,
                                                                int Z,
                                                                const double & properTime)
  {
  /* In case a root file is created, copy values to branched variables */
  mInteractionPosition = interactionPosition;
  mInteractionDirection = interactionDirection;
  mInteractionEnergy = energy;
  mInteractionWeight = weight;
  mInteractionZ = Z;
  mInteractionEventId = eventID;
  strcpy(mInteractionProductionProcessStep, processName.c_str());
  if (mMapProcessNameWithType.find(processName) == mMapProcessNameWithType.end())
    {
    GateWarning("Unhandled gamma interaction in GateFixedForcedDetectionActor. Process name is " << processName << ".\n");
    return;
    }
  else
    {
    mInteractionOrder++;
    switch (mMapProcessNameWithType[processName])
      {
      case COMPTON:
        this->ForceDetectionOfInteraction<COMPTON>(mComptonProjector.GetPointer(),
                                                   mProcessImage[COMPTON],
                                                   properTime);
        break;

      case RAYLEIGH:
        mInteractionWeight = mEnergyResponseDetector(mInteractionEnergy) * mInteractionWeight;
        this->ForceDetectionOfInteraction<RAYLEIGH>(mRayleighProjector.GetPointer(),
                                                    mProcessImage[RAYLEIGH],
                                                    properTime);
        break;

      case PHOTOELECTRIC:
        mInteractionWeight = mEnergyResponseDetector(mInteractionEnergy) * mInteractionWeight;
        this->ForceDetectionOfInteraction<PHOTOELECTRIC>(mFluorescenceProjector.GetPointer(),
                                                         mProcessImage[PHOTOELECTRIC],
                                                         properTime);
        break;

      case ISOTROPICPRIMARY:
        mInteractionWeight = mEnergyResponseDetector(mInteractionEnergy) * mInteractionWeight;
        this->ForceDetectionOfInteraction<ISOTROPICPRIMARY>(mIsotropicPrimaryProjector.GetPointer(),
                                                            mProcessImage[ISOTROPICPRIMARY],
                                                            properTime);
        break;

      default:
        GateError("Error: implementation problem, unexpected process type reached.");
      }
    }
  if (mPhaseSpaceFile)
    {
    mPhaseSpace->Fill();
    }
  }

void GateFixedForcedDetectionActor::GeneratePhotons(const unsigned int & numberOfThreads,
                                                    const std::vector<std::vector<newPhoton> > & photonList,
                                                    const double & properTime)
  {
  static G4EventManager * em = G4EventManager::GetEventManager();
  G4StackManager * sm = em->GetStackManager();
  G4ThreeVector position;
  for (unsigned int thread = 0; thread < numberOfThreads; thread++)
    {
    for (unsigned int photonId = 0; photonId < photonList[thread].size(); photonId++)
      {
      for (int i = 0; i < 3; i++)
        {
        position[i] = photonList[thread][photonId].position[i]
                      + mGateToITKImageFilter->GetOrigin()[i];
        }
      G4DynamicParticle * newPhoton = new G4DynamicParticle(G4Gamma::Gamma(),
                                                            photonList[thread][photonId].direction,
                                                            photonList[thread][photonId].energy);
      G4Track * newTrack = new G4Track(newPhoton, properTime, position);
      newTrack->SetParentID(666);
      newTrack->SetWeight(photonList[thread][photonId].weight);
      sm->PushOneTrack(newTrack);
      }
    }
  }

template<ProcessType VProcess, class TProjectorType>
void GateFixedForcedDetectionActor::ForceDetectionOfInteraction(TProjectorType *projector,
                                                                InputImageType::Pointer & input,
                                                                const double & properTime)
  {
  if (!mDoFFDForThisProcess[VProcess])
    {
    return;
    }
  /* direction and position are in World coordinates and they must be in CT coordinates */
  G4ThreeVector interactionPositionInCT = m_WorldToCT.TransformPoint(mInteractionPosition);
  G4ThreeVector interactionDirectionInCT = m_WorldToCT.TransformAxis(mInteractionDirection);

  /* Convert to ITK */
  VectorType direction;
  for (unsigned int i = 0; i < 3; i++)
    {
    mInteractionITKPosition[i] = interactionPositionInCT[i];
    direction[i] = interactionDirectionInCT[i];
    }

  /* Create interaction geometry */
  GeometryType::Pointer oneProjGeometry = GeometryType::New();
  oneProjGeometry->AddReg23Projection(mInteractionITKPosition,
                                      mDetectorPosition,
                                      mDetectorRowVector,
                                      mDetectorColVector);
  mProcessTimeProbe[VProcess].Start();
  projector->SetInput(FirstSliceProjection(input));
  projector->SetGeometry(oneProjGeometry.GetPointer());
  projector->GetProjectedValueAccumulation().SetEnergyZAndWeight(mInteractionEnergy,
                                                                 mInteractionZ,
                                                                 mInteractionWeight);
  projector->GetProjectedValueAccumulation().ClearPhotonList();
  projector->GetProjectedValueAccumulation().SetDirection(direction);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(projector->Update());
  mProcessTimeProbe[VProcess].Stop();
  mInteractionTotalContribution = projector->GetProjectedValueAccumulation().GetIntegralOverDetectorAndReset();

  if (mGeneratePhotons)
    {
    GeneratePhotons(projector->GetNumberOfThreads(),
                    projector->GetProjectedValueAccumulation().GetPhotonList(),
                    properTime);
    }

  /* Scatter order */
  if (mPerOrderImagesBaseName != "")
    {
    while (mInteractionOrder > (int) mPerOrderImages[VProcess].size())
      {
      mPerOrderImages[VProcess].push_back(CreateVoidProjectionImage());
      }
    projector->SetInput(FirstSliceProjection(mPerOrderImages[VProcess][mInteractionOrder - 1]));
    TRY_AND_EXIT_ON_ITK_EXCEPTION(projector->Update());
    }
  }

void GateFixedForcedDetectionActor::SaveData()
  {
  SaveData("");
  }

void GateFixedForcedDetectionActor::SaveData(const G4String prefix)
  {
  typedef itk::BinaryFunctorImageFilter<InputImageType, InputImageType, InputImageType,
      GateFixedForcedDetectionFunctor::Chetty<InputImageType::PixelType> > ChettyType;

  GateVActor::SaveData();

  /* Geometry */
  if (mGeometryFilename != "")
    {
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer geoWriter = rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
    geoWriter->SetObject(mGeometry);
    geoWriter->SetFilename(AddPrefix(prefix, mGeometryFilename));
    geoWriter->WriteFile();
    }

  itk::ImageFileWriter<InputImageType>::Pointer imgWriter;
  imgWriter = itk::ImageFileWriter<InputImageType>::New();
  char filename[1024];
  G4int rID = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();

  if (mPrimaryFilename != "")
    {
    /* Write the image of primary radiation accounting for the fluence of the
     primary source. */
    sprintf(filename, AddPrefix(prefix, mPrimaryFilename).c_str(), rID);
    imgWriter->SetFileName(filename);
    if (mSourceType == "isotropic")
      {
      imgWriter->SetInput(PrimaryFluenceWeighting(mProcessImage[ISOTROPICPRIMARY]));
      }
    else
      {
      imgWriter->SetInput(PrimaryFluenceWeighting(mPrimaryImage));
      }
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }

  if (mMaterialMuFilename != "")
    {
    AccumulationType::MaterialMuImageType *map;
    map = mComptonProjector->GetProjectedValueAccumulation().GetMaterialMu();

    /* Change spacing to keV */
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
    w->SetInput(ci->GetOutput());
    w->SetFileName(AddPrefix(prefix, mMaterialMuFilename));
    TRY_AND_EXIT_ON_ITK_EXCEPTION(w->Update());
    }

  if (mAttenuationFilename != "")
    {
    /* Attenuation Functor -> atten */
    typedef itk::BinaryFunctorImageFilter<InputImageType, InputImageType, InputImageType,
        GateFixedForcedDetectionFunctor::Attenuation<InputImageType::PixelType> > attenFunctor;
    attenFunctor::Pointer atten = attenFunctor::New();

    /* In the attenuation, we assume that the whole detector is irradiated.
     Otherwise we would have a division by 0. */
    atten->SetInput1(mPrimaryImage);
    atten->SetInput2(mFlatFieldImage);
    atten->InPlaceOff();

    sprintf(filename, AddPrefix(prefix, mAttenuationFilename).c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(atten->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }

  if (mFlatFieldFilename != "")
    {
    /*  Write the image of the flat field accounting for the fluence of the
     primary source. */
    sprintf(filename, AddPrefix(prefix, mFlatFieldFilename).c_str(), rID);
    imgWriter->SetFileName(filename);
    imgWriter->SetInput(PrimaryFluenceWeighting(mFlatFieldImage));
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
    }

  for (unsigned int i = 0; i < PRIMARY; i++)
    {
    ProcessType pt = ProcessType(i);
    if (mProcessImageFilenames[pt] != "")
      {
      sprintf(filename, AddPrefix(prefix, mProcessImageFilenames[pt]).c_str(), rID);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mProcessImage[pt]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
      if (mIsSecondarySquaredImageEnabled)
        {
        imgWriter->SetFileName(G4String(removeExtension(filename))
                               + "-Squared."
                               + G4String(getExtension(filename)));
        imgWriter->SetInput(mSquaredImage[pt]);
        TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
        }
      if (mIsSecondaryUncertaintyImageEnabled)
        {
        ChettyType::Pointer chetty = ChettyType::New();
        chetty->GetFunctor().SetN(mNumberOfEventsInRun);
        chetty->SetInput1(mProcessImage[pt]);
        chetty->SetInput2(mSquaredImage[pt]);
        chetty->InPlaceOff();

        imgWriter->SetFileName(G4String(removeExtension(filename))
                               + "-Uncertainty."
                               + G4String(getExtension(filename)));
        imgWriter->SetInput(chetty->GetOutput());

        TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
        }
      for (unsigned int k = 0; k < mPerOrderImages[pt].size(); k++)
        {
        sprintf(filename,
                AddPrefix(prefix,
                          mPerOrderImagesBaseName
                          + mMapTypeWithProcessName[pt]
                          + "%04d_order%02d.mha").c_str(),
                rID,
                k);
        imgWriter->SetFileName(filename);
        imgWriter->SetInput(mPerOrderImages[pt][k]);
        TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
        }
      }
    }

  if (mSecondaryFilename != ""
      || mIsSecondarySquaredImageEnabled
      || mIsSecondaryUncertaintyImageEnabled
      || mTotalFilename != "")
    {
    typedef itk::AddImageFilter<OutputImageType, OutputImageType, OutputImageType> AddImageFilterType;
    AddImageFilterType::Pointer addFilter = AddImageFilterType::New();

    /*  The secondary image contains all calculated scatterings
     (Compton, Rayleigh and/or Fluorescence)
     Create projections image */
    InputImageType::Pointer mSecondaryImage = CreateVoidProjectionImage();
    for (unsigned int i = 0; i < PRIMARY - 1; i++)
      {
      /*  Add Image Filter used to sum the different figures obtained on each process */
      addFilter->InPlaceOn();
      addFilter->SetInput1(mSecondaryImage);
      addFilter->SetInput2(mProcessImage[ProcessType(i)]);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());
      mSecondaryImage = addFilter->GetOutput();
      mSecondaryImage->DisconnectPipeline();
      }

    /*  Write scatter image */
    if (mSecondaryFilename != "")
      {
      sprintf(filename, AddPrefix(prefix, mSecondaryFilename).c_str(), rID);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mSecondaryImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
      }

    if (mIsSecondarySquaredImageEnabled)
      {
      imgWriter->SetFileName(G4String(removeExtension(filename))
                             + "-Squared."
                             + G4String(getExtension(filename)));
      imgWriter->SetInput(mSecondarySquaredImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
      }

    /*  Write scatter uncertainty image */
    if (mIsSecondaryUncertaintyImageEnabled)
      {
      ChettyType::Pointer chetty = ChettyType::New();
      chetty->GetFunctor().SetN(mNumberOfEventsInRun);
      chetty->SetInput1(mSecondaryImage);
      chetty->SetInput2(mSecondarySquaredImage);
      chetty->InPlaceOff();

      imgWriter->SetFileName(G4String(removeExtension(filename))
                             + "-Uncertainty."
                             + G4String(getExtension(filename)));
      imgWriter->SetInput(chetty->GetOutput());

      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
      }

    if (mTotalFilename != "")
      {
      /*  Primary */
      mSecondaryImage->DisconnectPipeline();
      addFilter->SetInput1(mSecondaryImage);
      addFilter->SetInput2(PrimaryFluenceWeighting(mPrimaryImage));
      TRY_AND_EXIT_ON_ITK_EXCEPTION(addFilter->Update());
      mSecondaryImage = addFilter->GetOutput();

      /*  Write Total Image */
      sprintf(filename, AddPrefix(prefix, mTotalFilename).c_str(), rID);
      imgWriter->SetFileName(filename);
      imgWriter->SetInput(mSecondaryImage);
      TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());
      }
    }

  if (mPhaseSpaceFile)
    {
    mPhaseSpace->GetCurrentFile()->Write();
    }
  }

void GateFixedForcedDetectionActor::ResetData()
  {
  mGeometry = GeometryType::New();
  }

void GateFixedForcedDetectionActor::SetGeometryFromInputRTKGeometryFile(GateVSource *source,
                                                                        GateVVolume *detector,
                                                                        GateVImageVolume *ct,
                                                                        const G4Run *run)
  {
  if (!mInputGeometry.GetPointer())
    {
    rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
    geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
    geometryReader->SetFilename(mInputRTKGeometryFilename);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(geometryReader->GenerateOutputInformation());
    mInputGeometry = geometryReader->GetGeometry();
    }

  if (run->GetRunID() >= (int) mInputGeometry->GetGantryAngles().size())
    {
    GateError("Error: SetGeometryFromInputRTKGeometryFile: you have more runs than what file " << mInputRTKGeometryFilename << " describes.");
    }

  /*  Source */
  if (source->GetRelativePlacementVolume() != "world")
    {
    GateError("Error: SetGeometryFromInputRTKGeometryFile" << "expects a source attached to the world.");
    }
  G4ThreeVector srcTrans;
  srcTrans[0] = mInputGeometry->GetSourceOffsetsX()[run->GetRunID()];
  srcTrans[1] = mInputGeometry->GetSourceOffsetsY()[run->GetRunID()];
  srcTrans[2] = mInputGeometry->GetSourceToIsocenterDistances()[run->GetRunID()];
  if (source->GetPosDist()->GetPosDisType() == "Point")
    {
    source->GetPosDist()->SetCentreCoords(srcTrans);
    } /*  point */
  else
    {
    G4ThreeVector offset = source->GetPosDist()->GetCentreCoords()
                           - source->GetAngDist()->GetFocusPointCopy();
    source->GetAngDist()->SetFocusPoint(srcTrans);
    source->GetAngDist()->SetFocusPointCopy(srcTrans);
    source->GetPosDist()->SetCentreCoords(srcTrans + offset);
    }

  /*  Detector */
  if (detector->GetParentVolume()->GetLogicalVolume()->GetName() != "world_log")
    {
    GateError("SetGeometryFromInputRTKGeometryFile" << " expects a detector attached to the world.");
    }
  G4ThreeVector detTrans;
  /* FIXME: detector->GetOrigin()? */
  detTrans[0] = mInputGeometry->GetProjectionOffsetsX()[run->GetRunID()];
  detTrans[1] = mInputGeometry->GetProjectionOffsetsY()[run->GetRunID()];
  detTrans[2] = srcTrans[2] - mInputGeometry->GetSourceToDetectorDistances()[run->GetRunID()];
  detector->GetPhysicalVolume()->SetTranslation(detTrans);

  /*  Create rotation matrix and rotate CT */
  if (ct->GetParentVolume()->GetLogicalVolume()->GetName() != "world_log")
    {
    GateError("Error: SetGeometryFromInputRTKGeometryFile" << " expects a voxelized volume attached to the world.");
    }
  CLHEP::Hep3Vector rows[3];
  for (unsigned int j = 0; j < 3; j++)
    {
    for (unsigned int i = 0; i < 3; i++)
      {
      rows[j][i] = mInputGeometry->GetRotationMatrices()[run->GetRunID()](i, j);
      }
    }
  if (!ct->GetPhysicalVolume()->GetRotation())
    {
    ct->GetPhysicalVolume()->SetRotation(new G4RotationMatrix);
    }

  // Read (once and for all, before any transformation) the CT position
  static G4AffineTransform ctToWorld(ct->GetPhysicalVolume()->GetRotation(),
                                     ct->GetPhysicalVolume()->GetTranslation());

  // Rotation of the CT according to geometry
  G4RotationMatrix worldRot;
  worldRot.setRows(rows[0], rows[1], rows[2]);

  G4AffineTransform ctToRotWorld = ctToWorld * worldRot;

  ct->GetPhysicalVolume()->GetRotation()->setRows(ctToRotWorld.NetRotation().row1(),
                                                  ctToRotWorld.NetRotation().row2(),
                                                  ctToRotWorld.NetRotation().row3());
  ct->GetPhysicalVolume()->SetTranslation(ctToRotWorld.NetTranslation());

  /*  According to BookForAppliDev.pdf section 3.4.4.3, we are allowed to change
   the geometry in BeginOfRunAction provided that we call this: */
  GateRunManager::GetRunManager()->GeometryHasBeenModified();
  }

void GateFixedForcedDetectionActor::ComputeGeometryInfoInImageCoordinateSystem(GateVImageVolume *ct,
                                                                               GateVVolume *detector,
                                                                               GateVSource *src,
                                                                               PointType & primarySourcePosition,
                                                                               PointType & detectorPosition,
                                                                               VectorType & detectorRowVector,
                                                                               VectorType & detectorColVector)
  {
  /*  The placement of a volume relative to its mother's coordinate system is not
   very well explained in Geant4's doc but the code follows what's done in
   source/geometry/volumes/src/G4PVPlacement.cc.

   One must be extremely careful with the multiplication order. It is not
   intuitive in Geant4, i.e., G4AffineTransform.Product(A, B) means
   B*A in matrix notations. */

  /*  Detector to world */
  GateVVolume * v = detector;
  G4VPhysicalVolume * phys = v->GetPhysicalVolume();
  G4AffineTransform detectorToWorld(phys->GetRotation(), phys->GetTranslation());

  while (v->GetLogicalVolumeName() != "world_log")
    {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    detectorToWorld = detectorToWorld * x;
    }

  /*  CT to world */
  v = ct;
  phys = v->GetPhysicalVolume();
  G4AffineTransform ctToWorld(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log")
    {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    ctToWorld = ctToWorld * x;
    }
  m_WorldToCT = ctToWorld.Inverse();

  /*  Source to world */
  G4String volname = src->GetRelativePlacementVolume();
  v = GateObjectStore::GetInstance()->FindVolumeCreator(volname);
  phys = v->GetPhysicalVolume();
  G4AffineTransform sourceToWorld(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log")
    {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    sourceToWorld = sourceToWorld * x;
    }

  /*  Detector parameters */
  G4AffineTransform detectorToCT(detectorToWorld * m_WorldToCT);
  /*  check where to get the two directions of the detector.
   Probably the dimension that has lowest size in one of the three directions. */
  G4ThreeVector du;
  G4ThreeVector dv;
  if (detector->GetHalfDimension(0) > 0.00000055
      && detector->GetHalfDimension(1) > 0.00000055
      && detector->GetHalfDimension(2) > 0.00000055)
    {
    GateError("Error: Only plane detectors have been implemented yet -> one dimension of the box should be at 1 nm.  (x = "<<2.0*detector->GetHalfDimension(0)<<"mm, y = "<<2.0*detector->GetHalfDimension(1)<<"mm, z = "<<2.0*detector->GetHalfDimension(2)<<"mm)");
    }
  else
    {
    if (v->GetHalfDimension(0) < 0.00000055)
      {
      du = detectorToCT.TransformAxis(G4ThreeVector(0, 1, 0));
      dv = detectorToCT.TransformAxis(G4ThreeVector(0, 0, 1));
      }
    else if (v->GetHalfDimension(0) < 0.00000055)
      {
      du = detectorToCT.TransformAxis(G4ThreeVector(1, 0, 0));
      dv = detectorToCT.TransformAxis(G4ThreeVector(0, 0, 1));
      }
    else
      {
      du = detectorToCT.TransformAxis(G4ThreeVector(1, 0, 0));
      dv = detectorToCT.TransformAxis(G4ThreeVector(0, 1, 0));
      }

    }
  G4ThreeVector dp = detectorToCT.TransformPoint(G4ThreeVector(0, 0, 0));

  /*  Source */
  G4ThreeVector s = src->GetAngDist()->GetFocusPointCopy();
  if (src->GetPosDist()->GetPosDisType() == "Point")
    {
    s = src->GetPosDist()->GetCentreCoords();
    } /*  point */

  m_SourceToCT = sourceToWorld * m_WorldToCT;
  s = m_SourceToCT.TransformPoint(s);

  /*  Copy in ITK vectors */
  for (int i = 0; i < 3; i++)
    {
    detectorRowVector[i] = du[i];
    detectorColVector[i] = dv[i];
    detectorPosition[i] = dp[i];
    primarySourcePosition[i] = s[i];
    }
  }

GateFixedForcedDetectionActor::InputImageType::Pointer GateFixedForcedDetectionActor::ConvertGateImageToITKImage(GateVImageVolume * gateImgVol)
  {
  GateImage *gateImg = gateImgVol->GetImage();

  /*  The direction is not accounted for in Gate. */
  InputImageType::SizeType size;
  InputImageType::PointType origin;
  InputImageType::RegionType region;
  InputImageType::SpacingType spacing;
  for (unsigned int i = 0; i < 3; i++)
    {
    size[i] = gateImg->GetResolution()[i];
    spacing[i] = gateImg->GetVoxelSize()[i];
    origin[i] = -gateImg->GetHalfSize()[i] + 0.5 * spacing[i];
    }
  region.SetSize(size);

  if (mGateToITKImageFilter.GetPointer() == NULL)
    {
    mGateToITKImageFilter = rtk::ImportImageFilter<InputImageType>::New();
    }
  mGateToITKImageFilter->SetRegion(region);
  mGateToITKImageFilter->SetImportPointer(&*(gateImg->begin()),
                                          gateImg->GetNumberOfValues(),
                                          false);
  mGateToITKImageFilter->SetSpacing(spacing);
  mGateToITKImageFilter->SetOrigin(origin);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(mGateToITKImageFilter->Update());

  /*  Get world material */
  std::vector<G4Material*> mat;
  gateImgVol->BuildLabelToG4MaterialVector(mat);
  InputPixelType worldMat = mat.size();

  /*  Pad 1 pixel with world material because interpolation will cut out half a voxel around */
  itk::ConstantPadImageFilter<InputImageType, InputImageType>::Pointer pad;
  pad = itk::ConstantPadImageFilter<InputImageType, InputImageType>::New();
  InputImageType::SizeType border;
  border.Fill(1);
  pad->SetPadBound(border);
  pad->SetConstant(worldMat);
  pad->SetInput(mGateToITKImageFilter->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(pad->Update());

  InputImageType::Pointer output = pad->GetOutput();
  return output;
  }

GateFixedForcedDetectionActor::InputImageType::Pointer GateFixedForcedDetectionActor::CreateVoidProjectionImage()
  {
  mDetector = GateObjectStore::GetInstance()->FindVolumeCreator(mDetectorName);
  InputImageType::SizeType size;
  size[0] = GetDetectorResolution()[0];
  size[1] = GetDetectorResolution()[1];
  if (mEnergyResolvedBinSize == 0.)
    {
    size[2] = 1;
    }
  else
    {
    size[2] = 1 + itk::Math::Floor<unsigned int>(mMaxPrimaryEnergy / mEnergyResolvedBinSize + 0.5);
    }

  InputImageType::SpacingType spacing;
  spacing[0] = mDetector->GetHalfDimension(0) * 2.0 / size[0];
  spacing[1] = mDetector->GetHalfDimension(1) * 2.0 / size[1];
  spacing[2] = (mEnergyResolvedBinSize == 0.) ? 1.0 : mEnergyResolvedBinSize / CLHEP::keV;

  InputImageType::PointType origin;
  origin[0] = -mDetector->GetHalfDimension(0) + 0.5 * spacing[0];
  origin[1] = -mDetector->GetHalfDimension(1) + 0.5 * spacing[1];
  origin[2] = 0.0;

  rtk::ConstantImageSource<InputImageType>::Pointer source;
  source = rtk::ConstantImageSource<InputImageType>::New();
  source->SetSpacing(spacing);
  source->SetOrigin(origin);
  source->SetSize(size);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(source->Update());

  GateFixedForcedDetectionActor::InputImageType::Pointer output;
  output = source->GetOutput();
  output->DisconnectPipeline();
  return output;
  }

/*  This function is used for energy resolved outputs. We create an image that
 has one (energy) slice only so that itk will only iterate on one slice.
 However, the pointer points to a full image and we move the pointer around
 to the correct energy slice in the functors (see the Accumulate function). */
GateFixedForcedDetectionActor::InputImageType::Pointer GateFixedForcedDetectionActor::FirstSliceProjection(InputImageType::Pointer & input)
  {
  input->Modified();
  GateFixedForcedDetectionActor::InputImageType::RegionType region;
  region = input->GetLargestPossibleRegion();
  rtk::ImportImageFilter<InputImageType>::Pointer sliceFilter = rtk::ImportImageFilter<
      InputImageType>::New();
  region.SetSize(2, 1);
  sliceFilter->SetRegion(region);
  sliceFilter->SetImportPointer(input->GetBufferPointer(), region.GetNumberOfPixels(), false);
  sliceFilter->SetSpacing(input->GetSpacing());
  sliceFilter->SetOrigin(input->GetOrigin());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(sliceFilter->Update());
  return sliceFilter->GetOutput();
  }

GateFixedForcedDetectionActor::InputImageType::Pointer GateFixedForcedDetectionActor::PrimaryFluenceWeighting(const InputImageType::Pointer input)
  {
  InputImageType::Pointer output = input;
  if (mSource->GetPosDist()->GetPosDisType() == "Point")
    {
    GateWarning("Primary fluence is not accounted for with a Point source distribution");
    }
  else if (mSource->GetPosDist()->GetPosDisType() == "Plane")
    {
    /*  Check plane source projection probability */
    G4ThreeVector sourceCorner1 = mSource->GetPosDist()->GetCentreCoords();
    sourceCorner1[0] -= mSource->GetPosDist()->GetHalfX();
    sourceCorner1[1] -= mSource->GetPosDist()->GetHalfY();
    sourceCorner1 = m_SourceToCT.TransformPoint(sourceCorner1);

    G4ThreeVector sourceCorner2 = mSource->GetPosDist()->GetCentreCoords();
    sourceCorner2[0] += mSource->GetPosDist()->GetHalfX();
    sourceCorner2[1] += mSource->GetPosDist()->GetHalfY();
    sourceCorner2 = m_SourceToCT.TransformPoint(sourceCorner2);

    /* Compute source plane corner positions in homogeneous coordinates */
    itk::Vector<double, 4> corner1Hom, corner2Hom;
    corner1Hom[0] = sourceCorner1[0];
    corner1Hom[1] = sourceCorner1[1];
    corner1Hom[2] = sourceCorner1[2];
    corner1Hom[3] = 1.;
    corner2Hom[0] = sourceCorner2[0];
    corner2Hom[1] = sourceCorner2[1];
    corner2Hom[2] = sourceCorner2[2];
    corner2Hom[3] = 1.;

    /* Project onto detector */
    itk::Vector<double, 3> corner1ProjHom, corner2ProjHom;
    corner1ProjHom.SetVnlVector(mGeometry->GetMatrices().back().GetVnlMatrix()
                                * corner1Hom.GetVnlVector());
    corner2ProjHom.SetVnlVector(mGeometry->GetMatrices().back().GetVnlMatrix()
                                * corner2Hom.GetVnlVector());
    corner1ProjHom /= corner1ProjHom[2];
    corner2ProjHom /= corner2ProjHom[2];

    /* Convert to non homogeneous coordinates */
    InputImageType::PointType corner1Proj, corner2Proj;
    corner1Proj[0] = corner1ProjHom[0];
    corner1Proj[1] = corner1ProjHom[1];
    corner1Proj[2] = 0.;
    corner2Proj[0] = corner2ProjHom[0];
    corner2Proj[1] = corner2ProjHom[1];
    corner2Proj[2] = 0.;

    /* Convert to projection indices */
    itk::ContinuousIndex<double, 3> corner1Idx, corner2Idx;
    input->TransformPhysicalPointToContinuousIndex<double>(corner1Proj, corner1Idx);
    input->TransformPhysicalPointToContinuousIndex<double>(corner2Proj, corner2Idx);
    if (corner1Idx[0] > corner2Idx[0])
      {
      std::swap(corner1Idx[0], corner2Idx[0]);
      }
    if (corner1Idx[1] > corner2Idx[1])
      {
      std::swap(corner1Idx[1], corner2Idx[1]);
      }

    /* Create copy of image normalized by the number of particles and the ratio
     between source size on the detector and the detector size in pixels */
    typedef itk::MultiplyImageFilter<InputImageType, InputImageType> MultiplyType;
    MultiplyType::Pointer mult = MultiplyType::New();
    mult->SetInput(input);
    InputImageType::SizeType size = input->GetLargestPossibleRegion().GetSize();
    mult->SetConstant(mNumberOfEventsInRun
                      / ((corner2Idx[1] - corner1Idx[1]) * (corner2Idx[0] - corner1Idx[0])));
    mult->InPlaceOff();
    TRY_AND_EXIT_ON_ITK_EXCEPTION(mult->Update());

    /* Multiply by pixel fraction */
    itk::ImageRegionIterator<InputImageType> it(mult->GetOutput(),
                                                mult->GetOutput()->GetLargestPossibleRegion());
    InputImageType::IndexType idx = input->GetLargestPossibleRegion().GetIndex();
    for (unsigned int k = idx[2]; k < size[2]; k++)
      {
      for (unsigned int j = idx[1]; j < size[1]; j++)
        {
        for (unsigned int i = idx[0]; i < size[0]; i++)
          {
          double maxInfX = std::max<double>(i - 0.5, corner1Idx[0]);
          double maxInfY = std::max<double>(j - 0.5, corner1Idx[1]);
          double minSupX = std::min<double>(i + 0.5, corner2Idx[0]);
          double minSupY = std::min<double>(j + 0.5, corner2Idx[1]);
          it.Set(it.Get()
                 * std::max<double>(0., minSupX - maxInfX)
                 * std::max<double>(0., minSupY - maxInfY));
          ++it;
          }
        }
      }
    output = mult->GetOutput();
    }
  return output;
  }

G4String GateFixedForcedDetectionActor::AddPrefix(G4String prefix, G4String filename)
  {
  G4String path = itksys::SystemTools::GetFilenamePath(filename);
  G4String name = itksys::SystemTools::GetFilenameName(filename);
  if (path == "")
    {
    path = ".";
    }
  return path + '/' + prefix + name;
  }

void GateFixedForcedDetectionActor::CreatePhaseSpace(const G4String phaseSpaceFilename,
                                                     TFile *&phaseSpaceFile,
                                                     TTree *&phaseSpace)
  {
  phaseSpaceFile = NULL;
  if (phaseSpaceFilename != "")
    {
    phaseSpaceFile = new TFile(phaseSpaceFilename, "RECREATE", "ROOT file for phase space", 9);
    }

  phaseSpace = new TTree("PhaseSpace", "Phase space tree of fixed forced detection actor");
  phaseSpace->Branch("Ekine", &mInteractionEnergy, "Ekine/D");
  phaseSpace->Branch("Weight", &mInteractionWeight, "Weight/D");
  phaseSpace->Branch("X", &(mInteractionPosition[0]), "X/D");
  phaseSpace->Branch("Y", &(mInteractionPosition[1]), "Y/D");
  phaseSpace->Branch("Z", &(mInteractionPosition[2]), "Z/D");
  phaseSpace->Branch("dX", &(mInteractionDirection[0]), "dX/D");
  phaseSpace->Branch("dY", &(mInteractionDirection[1]), "dY/D");
  phaseSpace->Branch("dZ", &(mInteractionDirection[2]), "dZ/D");
  phaseSpace->Branch("EventID", &mInteractionEventId, "EventID/I");
  phaseSpace->Branch("ProductionProcessStep",
                     mInteractionProductionProcessStep,
                     "ProductionProcessStep/C");
  phaseSpace->Branch("TotalContribution", &mInteractionTotalContribution, "TotalContribution/D");
  phaseSpace->Branch("MaterialZ", &mInteractionZ, "MaterialZ/I");
  phaseSpace->Branch("Order", &mInteractionOrder, "Order/I");
  }

#endif
