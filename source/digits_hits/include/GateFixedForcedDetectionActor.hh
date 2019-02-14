/*----------------------
 GATE version name: gate_v6

 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ----------------------*/

/* \class GateFixedForcedDetectionActor */

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#ifndef GATEFIXEDFORCEDDECTECTIONACTOR_HH
#define GATEFIXEDFORCEDDECTECTIONACTOR_HH

#include "globals.hh"
#include "G4String.hh"
#include <iomanip>
#include <vector>

/* Gate */
#include "GateVActor.hh"
#include "GateFixedForcedDetectionActorMessenger.hh"
#include "GateImage.hh"
#include "GateSourceMgr.hh"
#include "GateVImageVolume.hh"
#include "GateFixedForcedDetectionFunctors.hh"
#include "GateEnergyResponseFunctor.hh"
#include "GateFixedForcedDetectionProjector.h"
#include "GateFixedForcedDetectionProcessType.hh"
#include "GateARFSD.hh"

/* itk */
#include <itkTimeProbe.h>
#include <itkBinShrinkImageFilter.h>
#include <itkMultiplyImageFilter.h>

/* rtk */
#include <rtkConstantImageSource.h>
#include <rtkThreeDCircularProjectionGeometry.h>
#include <rtkImportImageFilter.h>

class GateFixedForcedDetectionActorMessenger;
class GateFixedForcedDetectionActor: public GateVActor
  {
public:

  /* This macro initialize the CreatePrototype and CreateInstance */
  FCT_FOR_AUTO_CREATOR_ACTOR(GateFixedForcedDetectionActor)

  GateFixedForcedDetectionActor(G4String name, G4int depth = 0);
  virtual ~GateFixedForcedDetectionActor();

  /*Constructs the actor */
  virtual void Construct();

  /* Callbacks */
  virtual void BeginOfRunAction(const G4Run*);
  virtual void BeginOfEventAction(const G4Event*e = NULL);
  virtual void EndOfEventAction(const G4Event*e = NULL);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

  /* Saves the data collected to the file */
  virtual void SaveData();
  virtual void SaveData(const G4String prefix);
  virtual void ResetData();

  /* Typedef for rtk */
  static const unsigned int Dimension = 3;
  typedef float InputPixelType;
  typedef itk::Image<InputPixelType, Dimension> InputImageType;
  typedef itk::Image<int, Dimension> IntegerImageType;
  typedef itk::Image<double, Dimension> DoubleImageType;
  typedef itk::Image<std::complex<InputPixelType>, Dimension> ComplexImageType;
  typedef float OutputPixelType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  typedef GeometryType::PointType PointType;
  typedef GeometryType::VectorType VectorType;
  typedef rtk::ConstantImageSource<OutputImageType> ConstantImageSourceType;
  typedef itk::BinShrinkImageFilter<InputImageType, OutputImageType> BinShrinkFilterType;

  /* Resolution of the detector plane (2D only, z=1); */
  const G4ThreeVector & GetDetectorResolution() const
    {
    return mDetectorResolution;
    }
  void SetDetectorResolution(int x, int y)
    {
    mDetectorResolution[0] = x;
    mDetectorResolution[1] = y;
    }
  void SetDetectorVolumeName(G4String name)
    {
    mDetectorName = name;
    }
  void SetGeometryFilename(G4String name)
    {
    mGeometryFilename = name;
    }
  void SetPrimaryFilename(G4String name)
    {
    mPrimaryFilename = name;
    }
  void SetMaterialMuFilename(G4String name)
    {
    mMaterialMuFilename = name;
    }
  void SetAttenuationFilename(G4String name)
    {
    mAttenuationFilename = name;
    }
  void SetMaterialDeltaFilename(G4String name)
    {
    mMaterialDeltaFilename = name;
    }
  void SetFresnelFilename(G4String name)
    {
    mFresnelFilename = name;
    }
  void SetResponseDetectorFilename(G4String name)
    {
    mResponseFilename = name;
    }
  void SetFlatFieldFilename(G4String name)
    {
    mFlatFieldFilename = name;
    }
  void SetComptonFilename(G4String name)
    {
    mProcessImageFilenames[COMPTON] = name;
    }
  void SetRayleighFilename(G4String name)
    {
    mProcessImageFilenames[RAYLEIGH] = name;
    }
  void SetFluorescenceFilename(G4String name)
    {
    mProcessImageFilenames[PHOTOELECTRIC] = name;
    }
  void SetIsotropicPrimaryFilename(G4String name)
    {
    mProcessImageFilenames[ISOTROPICPRIMARY] = name;
    }
  void SetSourceType(G4String name)
    {
    mSourceType = name;
    }
  void SetGeneratePhotons(G4String name)
    {
    if (name == "true" || name == "True")
      {
      mGeneratePhotons = true;
      }
    }

  void SetARF(G4String name)
    {
    if (name == "true" || name == "True")
      {
      mARF = true;
      }
    }
  void SetSecondaryFilename(G4String name)
    {
    mSecondaryFilename = name;
    }
  void EnableSecondarySquaredImage(bool b)
    {
    mIsSecondarySquaredImageEnabled = b;
    }
  void EnableSecondaryUncertaintyImage(bool b)
    {
    mIsSecondaryUncertaintyImageEnabled = b;
    }
  void SetTotalFilename(G4String name)
    {
    mTotalFilename = name;
    }
  void SetPhaseSpaceFilename(G4String name)
    {
    mPhaseSpaceFilename = name;
    }
  void SetNoisePrimary(G4int n)
    {
    mNoisePrimary = n;
    }
  const BinShrinkFilterType::ShrinkFactorsType & GetBinningFactor() const
    {
    return mBinningFactor;
    }
  void SetBinningFactor(G4int x, G4int y)
    {
    mBinningFactor[0] = x;
    mBinningFactor[1] = y;
    mBinShrinkFilter->SetShrinkFactors(mBinningFactor);
    }
  void SetInputRTKGeometryFilename(G4String name)
    {
    mInputRTKGeometryFilename = name;
    }
  void SetEnergyResolvedBinSize(const double e)
    {
    mEnergyResolvedBinSize = e;
    }

  void SetGeometryFromInputRTKGeometryFile(GateVSource *source,
                                           GateVVolume *detector,
                                           GateVImageVolume *ct,
                                           const G4Run *run);
  void ComputeGeometryInfoInImageCoordinateSystem(GateVImageVolume *image,
                                                  GateVVolume *detector,
                                                  GateVSource *src,
                                                  PointType &primarySourcePosition,
                                                  PointType &detectorPosition,
                                                  VectorType &detectorRowVector,
                                                  VectorType &detectorColVector);
  InputImageType::Pointer ConvertGateImageToITKImage(GateVImageVolume * gateImgVol);
  InputImageType::Pointer CreateVoidProjectionImage();
  InputImageType::Pointer FirstSliceProjection(InputImageType::Pointer &input);
  virtual void CreatePhaseSpace(const G4String phaseSpaceFilename,
                                TFile *&phaseSpaceFile,
                                TTree *&phaseSpace);

  /* The actual forced detection functions */
  virtual void ForceDetectionOfInteraction(G4int eventID,
                                           G4String processName,
                                           G4ThreeVector pt,
                                           G4ThreeVector dir,
                                           double energy,
                                           double weight,
                                           int Z);

  template<ProcessType VProcess, class TProjectorType>
  void ForceDetectionOfInteraction(TProjectorType *projector,
                                   InputImageType::Pointer &input);
  void TestSource(GateSourceMgr * sm);
  void GetEnergyList(std::vector<double> & energyList, std::vector<double> & energyWeightList);
  GateVImageVolume* SearchForVoxelisedVolume();

  void PrepareIsotropicPrimaryProjector(GateVImageVolume* gate_image_volume,
                                        unsigned int & nPixOneSlice,
                                        GeometryType::Pointer oneProjGeometry);

  void PrepareComptonProjector(GateVImageVolume* gate_image_volume,
                               unsigned int & nPixOneSlice,
                               GeometryType::Pointer oneProjGeometry);

  void PrepareRayleighProjector(GateVImageVolume* gate_image_volume,
                                unsigned int & nPixOneSlice,
                                GeometryType::Pointer oneProjGeometry);

  void PrepareFluorescenceProjector(GateVImageVolume* gate_image_volume,
                                    unsigned int & nPixOneSlice,
                                    GeometryType::Pointer oneProjGeometry);

  void PreparePrimaryProjector(GeometryType::Pointer oneProjGeometry,
                               std::vector<double> & energyList,
                               std::vector<double> & energyWeightList,
                               GateVImageVolume* gate_image_volume,
                               unsigned int & nPixOneSlice);

  void CalculatePropagatorImage(const double D, double magnification, std::vector<double> & energyList);

  void CreateProjectionImages();
  void GeneratePhotons(const unsigned int & numberOfThreads,
                       const std::vector<std::vector<newPhoton> > & photonList);

  void ConnectARF(const unsigned int & numberOfThreads,
                  const std::vector<std::vector<newPhoton> > & photonList,
                  unsigned int newHead = 1);

  void ComputeFlatField(std::vector<double> & energyList, std::vector<double> & energyWeightList);
protected:
  GateFixedForcedDetectionActorMessenger * pActorMessenger;

  G4String mDetectorName;
  G4EmCalculator * mEMCalculator;
  GateVVolume * mDetector;
  GateVSource * mSource;
  G4String mGeometryFilename;
  G4String mPrimaryFilename;
  G4String mMaterialMuFilename;
  G4String mAttenuationFilename;
  G4String mMaterialDeltaFilename;
  G4String mFresnelFilename;
  G4String mResponseFilename;
  G4String mFlatFieldFilename;
  G4String mSecondaryFilename;
  G4String mPerOrderImagesBaseName;
  bool mIsSecondarySquaredImageEnabled;
  bool mIsSecondaryUncertaintyImageEnabled;
  G4String mTotalFilename;

  /* Parameter for statistical noise */
  G4int mNoisePrimary;
  /* Parameter for modeling pixel-binning */
  BinShrinkFilterType::ShrinkFactorsType mBinningFactor;

  G4double mMinPrimaryEnergy;
  G4double mMaxPrimaryEnergy;

  G4ThreeVector mDetectorResolution;
  GateEnergyResponseFunctor mEnergyResponseDetector;

  G4String mInputRTKGeometryFilename;
  G4double mEnergyResolvedBinSize;
  GeometryType::Pointer mInputGeometry;
  GeometryType::Pointer mGeometry;
  InputImageType::Pointer mGateVolumeImage;
  rtk::ImportImageFilter<InputImageType>::Pointer mGateToITKImageFilter;
  InputImageType::Pointer mPrimaryImage;
  InputImageType::Pointer mDeltaImage;
  ComplexImageType::Pointer mPropagatorImage;
  InputImageType::Pointer mFlatFieldImage;
  InputImageType::Pointer mFlatFieldDeltaImage;
  std::map<ProcessType, InputImageType::Pointer> mProcessImage;
  std::map<ProcessType, InputImageType::Pointer> mSquaredImage;
  std::map<ProcessType, InputImageType::Pointer> mEventImage;
  InputImageType::Pointer mSecondarySquaredImage;

  std::map<G4String, ProcessType> mMapProcessNameWithType;
  std::map<ProcessType, G4String> mMapTypeWithProcessName;

  /* Geometry information initialized at the beginning of the run */
  G4AffineTransform m_WorldToCT;
  G4AffineTransform m_WorldToDetector;
  G4AffineTransform m_SourceToCT;
  G4AffineTransform m_SourceToDetector;
  PointType mPrimarySourcePosition;
  PointType mDetectorPosition;
  VectorType mDetectorRowVector;
  VectorType mDetectorColVector;

  /* Accumulation type */
  typedef GateFixedForcedDetectionFunctor::VAccumulation AccumulationType;

  /* Primary stuff */
  unsigned int mNumberOfEventsInRun;
  typedef GateFixedForcedDetectionProjector<
      GateFixedForcedDetectionFunctor::PrimaryValueAccumulation> PrimaryProjectionType;
  PrimaryProjectionType::Pointer mPrimaryProjector;
  /* Per process members */
  std::map<ProcessType, bool> mDoFFDForThisProcess;
  std::map<ProcessType, itk::TimeProbe> mProcessTimeProbe;
  std::map<ProcessType, std::vector<InputImageType::Pointer> > mPerOrderImages;
  std::map<ProcessType, G4String> mProcessImageFilenames;

  /* Compton stuff */
  typedef GateFixedForcedDetectionProjector<
      GateFixedForcedDetectionFunctor::ComptonValueAccumulation> ComptonProjectionType;
  ComptonProjectionType::Pointer mComptonProjector;

  /* Rayleigh stuff */
  typedef GateFixedForcedDetectionProjector<
      GateFixedForcedDetectionFunctor::RayleighValueAccumulation> RayleighProjectionType;
  RayleighProjectionType::Pointer mRayleighProjector;

  /* Fluorescence stuff */
  typedef GateFixedForcedDetectionProjector<
      GateFixedForcedDetectionFunctor::FluorescenceValueAccumulation> FluorescenceProjectionType;
  FluorescenceProjectionType::Pointer mFluorescenceProjector;

  /* Isotropic Primaries stuff */
  typedef GateFixedForcedDetectionProjector<
      GateFixedForcedDetectionFunctor::IsotropicPrimaryValueAccumulation> IsotropicPrimaryProjectionType;
  IsotropicPrimaryProjectionType::Pointer mIsotropicPrimaryProjector;

  /* Pixel-binning stuff */
  BinShrinkFilterType::Pointer mBinShrinkFilter;
  typedef itk::MultiplyImageFilter<OutputImageType, OutputImageType, OutputImageType> BinMultiplyFilterType;
  BinMultiplyFilterType::Pointer mBinMultiplyFilter;

  /* Phase space variables */
  G4String mPhaseSpaceFilename;
  TFile *mPhaseSpaceFile;
  TTree *mPhaseSpace;
  G4ThreeVector mInteractionDirection;
  G4ThreeVector mInteractionPosition;
  PointType mInteractionITKPosition;
  double mInteractionEnergy;
  double mInteractionWeight;
  Char_t mInteractionProductionProcessStep[256];
  int mInteractionEventId;
  double mInteractionTotalContribution;
  double mInteractionSquaredContribution;
  int mInteractionZ;
  int mInteractionOrder;
  int mInteractionChainCode;
  double mInteractionSquaredIntegralOverDetector;
  G4String mSourceType;
  bool mGeneratePhotons;

  bool mARF;
  unsigned int mNumberOfProcessedPrimaries;
  unsigned int mNumberOfProcessedSecondaries;
  unsigned int mNumberOfProcessedCompton;
  unsigned int mNumberOfProcessedRayleigh;
  unsigned int mNumberOfProcessedPE;

  /* Account for primary fluence weighting */
  InputImageType::Pointer PrimaryFluenceWeighting(const InputImageType::Pointer input);

  /* Account for pixel-binning */
  InputImageType::Pointer PixelBinning(const InputImageType::Pointer input, bool bSum = true, bool bSQRT = false);

  G4String AddPrefix(G4String prefix, G4String filename);
  };

MAKE_AUTO_CREATOR_ACTOR(FixedForcedDetectionActor, GateFixedForcedDetectionActor)

#endif /* end #define GATEFIXEDFORCEDDECTECTIONACTOR_HH */

#endif /* GATE_USE_RTK */
