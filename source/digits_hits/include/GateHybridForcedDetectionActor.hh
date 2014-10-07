/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

//-----------------------------------------------------------------------------
/// \class GateHybridForcedDetectionActor
//-----------------------------------------------------------------------------

#include "GateConfiguration.h"
#ifdef GATE_USE_RTK

#ifndef GATEHYBRIDFORCEDDECTECTIONACTOR_HH
#define GATEHYBRIDFORCEDDECTECTIONACTOR_HH

#include "globals.hh"
#include "G4String.hh"
#include <iomanip>
#include <vector>

// Gate
#include "GateVActor.hh"
#include "GateHybridForcedDetectionActorMessenger.hh"
#include "GateImage.hh"
#include "GateSourceMgr.hh"
#include "GateVImageVolume.hh"
#include "GateHybridForcedDetectionFunctors.hh"
#include "GateEnergyResponseFunctor.hh"
#include "GateHybridForcedDetectionProjector.h"
#include "GateHybridForcedDetectionProcessType.hh"

// itk
#include <itkTimeProbe.h>

// rtk
#include <rtkConstantImageSource.h>
#include <rtkReg23ProjectionGeometry.h>

//-----------------------------------------------------------------------------

class GateHybridForcedDetectionActorMessenger;
class GateHybridForcedDetectionActor : public GateVActor
{
public:

  //-----------------------------------------------------------------------------
  // This macro initialize the CreatePrototype and CreateInstance
  FCT_FOR_AUTO_CREATOR_ACTOR(GateHybridForcedDetectionActor)

  GateHybridForcedDetectionActor(G4String name, G4int depth=0);
  virtual ~GateHybridForcedDetectionActor();

  // Constructs the actor
  virtual void Construct();

  // Callbacks
  virtual void BeginOfRunAction(const G4Run*);
  virtual void EndOfRunAction(const G4Run*);
  virtual void BeginOfEventAction(const G4Event*e=NULL);
  virtual void EndOfEventAction(const G4Event*e=NULL);
  // virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*);

  // Saves the data collected to the file
  virtual void SaveData() { SaveData(""); }
  virtual void SaveData(const G4String prefix);
  virtual void ResetData();

  // Resolution of the detector plane (2D only, z=1);
  const G4ThreeVector & GetDetectorResolution() const { return mDetectorResolution; }
  void SetDetectorResolution(int x, int y) { mDetectorResolution[0] = x; mDetectorResolution[1] = y; }
  void SetDetectorVolumeName(G4String name) { mDetectorName = name; }
  void SetGeometryFilename(G4String name) { mGeometryFilename = name; }
  void SetPrimaryFilename(G4String name) { mPrimaryFilename = name; }
  void SetMaterialMuFilename(G4String name) { mMaterialMuFilename = name; }
  void SetAttenuationFilename(G4String name) { mAttenuationFilename = name; }
  void SetResponseDetectorFilename(G4String name) { mResponseFilename = name; }
  void SetFlatFieldFilename(G4String name) { mFlatFieldFilename = name; }
  void SetComptonFilename(G4String name) { mProcessImageFilenames[COMPTON] = name; }
  void SetRayleighFilename(G4String name) { mProcessImageFilenames[RAYLEIGH] = name; }
  void SetFluorescenceFilename(G4String name) { mProcessImageFilenames[PHOTOELECTRIC] = name; }
  void SetSecondaryFilename(G4String name) { mSecondaryFilename = name; }
  void EnableSecondarySquaredImage(bool b) { mIsSecondarySquaredImageEnabled = b; }
  void EnableSecondaryUncertaintyImage(bool b) { mIsSecondaryUncertaintyImageEnabled = b; }
  void SetTotalFilename(G4String name) { mTotalFilename = name; }
  void SetSingleInteractionFilename(G4String name) { mSingleInteractionFilename = name; }
  void SetSingleInteractionType(G4String type) { mSingleInteractionType = type; }
  void SetSingleInteractionPosition(G4ThreeVector pos) { mSingleInteractionPosition = pos; }
  void SetSingleInteractionDirection(G4ThreeVector dir) { mSingleInteractionDirection = dir; }
  void SetSingleInteractionEnergy(G4double e) { mSingleInteractionEnergy = e; }
  void SetSingleInteractionZ(G4int z) { mSingleInteractionZ = z; }
  void SetPhaseSpaceFilename(G4String name) { mPhaseSpaceFilename = name; }
  void SetWaterLUTFilename(G4String name) { mWaterLUTFilename = name; }
  void SetWaterLUTMaterial(G4String name) { mWaterLUTMaterial = name; }
  void SetNoisePrimary(G4int n) { mNoisePrimary = n; }
  void SetSecondPassPrefix(G4String name) { mSecondPassPrefix = name; }
  void SetSecondPassDetectorResolution(int x, int y) { mSecondPassDetectorResolution[0] = x; mSecondPassDetectorResolution[1] = y; }
  void SetRussianRouletteFilename(G4String name) { mRussianRouletteFilename = name; }
  void SetRussianRouletteSpacing(G4double s) { mRussianRouletteSpacing = s; }
  void SetRussianRouletteMinimumCountInRegion(G4double p) { mRussianRouletteMinimumCountInRegion = std::max(2.,p); }
  void SetRussianRouletteMinimumProbability(G4double p) { mRussianRouletteMinimumProbability = p; }
  void SetInputRTKGeometryFilename(G4String name) { mInputRTKGeometryFilename = name; }

  // Typedef for rtk
  static const unsigned int Dimension = 3;
  typedef float                                       InputPixelType;
  typedef itk::Image<InputPixelType, Dimension>       InputImageType;
  typedef itk::Image<int, Dimension>                  IntegerImageType;
  typedef itk::Image<double, Dimension>               DoubleImageType;
  typedef float                                       OutputPixelType;
  typedef itk::Image<OutputPixelType, Dimension>      OutputImageType;
  typedef rtk::Reg23ProjectionGeometry                GeometryType;
  typedef rtk::Reg23ProjectionGeometry::PointType     PointType;
  typedef rtk::Reg23ProjectionGeometry::VectorType    VectorType;
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;

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
  void CreatePhaseSpace(const G4String phaseSpaceFilename, TFile *&phaseSpaceFile, TTree *&phaseSpace);

  // The actual forced detection functions
  void ForceDetectionOfInteraction(G4int runID, G4int eventID, G4int trackID,
                                   G4String prodVol, G4String creatorProc,
                                   G4String processName, G4String interVol,
                                   G4ThreeVector pt, G4ThreeVector dir,
                                   double energy, double weight,
                                   G4String material, int Z, int order);
  template <ProcessType VProcess, class TProjectorType>
  void ForceDetectionOfInteraction(TProjectorType *projector,
                                   InputImageType::Pointer &input);

protected:
  GateHybridForcedDetectionActorMessenger * pActorMessenger;

  G4String mDetectorName;
  G4EmCalculator * mEMCalculator;
  GateVVolume * mDetector;
  GateVSource * mSource;
  G4String mGeometryFilename;
  G4String mPrimaryFilename;
  G4String mMaterialMuFilename;
  G4String mAttenuationFilename;
  G4String mResponseFilename;
  G4String mFlatFieldFilename;
  G4String mSecondaryFilename;
  bool mIsSecondarySquaredImageEnabled;
  bool mIsSecondaryUncertaintyImageEnabled;
  G4String mTotalFilename;
  G4String mWaterLUTFilename;
  G4String mWaterLUTMaterial;

  //parameter for statistical noise
  G4int   mNoisePrimary; 

  G4ThreeVector mDetectorResolution;
  GateEnergyResponseFunctor mEnergyResponseDetector;

  G4String mInputRTKGeometryFilename;
  rtk::ThreeDCircularProjectionGeometry::Pointer mInputGeometry;
  GeometryType::Pointer mGeometry;
  InputImageType::Pointer mGateVolumeImage;
  InputImageType::Pointer mPrimaryImage;
  InputImageType::Pointer mFlatFieldImage;
  std::map<ProcessType, InputImageType::Pointer> mProcessImage;
  std::map<ProcessType, InputImageType::Pointer> mSquaredImage;
  std::map<ProcessType, InputImageType::Pointer> mEventImage;
  InputImageType::Pointer mSecondarySquaredImage;

  std::map<G4String, ProcessType> mMapProcessNameWithType;
  std::map<ProcessType, G4String> mMapTypeWithProcessName;

  // Geometry information initialized at the beginning of the run
  G4AffineTransform m_WorldToCT;
  G4AffineTransform m_SourceToCT;
  PointType mPrimarySourcePosition;
  PointType mDetectorPosition;
  VectorType mDetectorRowVector;
  VectorType mDetectorColVector;

  // Accumulation type
  typedef GateHybridForcedDetectionFunctor::VAccumulation AccumulationType;

  // Primary stuff
  unsigned int mNumberOfEventsInRun;
  typedef rtk::JosephForwardProjectionImageFilter<
                 InputImageType,
                 InputImageType,
                 GateHybridForcedDetectionFunctor::InterpolationWeightMultiplication,
                 GateHybridForcedDetectionFunctor::PrimaryValueAccumulation>
                   PrimaryProjectionType;

  // Per process members
  std::map<ProcessType, bool> mDoFFDForThisProcess;
  std::map<ProcessType, itk::TimeProbe> mProcessTimeProbe;
  std::map<ProcessType, std::vector<InputImageType::Pointer> > mPerOrderImages;
  std::map<ProcessType, G4String> mProcessImageFilenames;

  // Compton stuff
  typedef GateHybridForcedDetectionProjector<GateHybridForcedDetectionFunctor::ComptonValueAccumulation> ComptonProjectionType;
  ComptonProjectionType::Pointer mComptonProjector;

  // Rayleigh stuff
  typedef GateHybridForcedDetectionProjector<GateHybridForcedDetectionFunctor::RayleighValueAccumulation> RayleighProjectionType;
  RayleighProjectionType::Pointer mRayleighProjector;

  // Fluorescence stuff
  typedef GateHybridForcedDetectionProjector<GateHybridForcedDetectionFunctor::FluorescenceValueAccumulation> FluorescenceProjectionType;
  FluorescenceProjectionType::Pointer mFluorescenceProjector;

  // Parameters for single event output
  InputImageType::Pointer mSingleInteractionImage;
  G4String                mSingleInteractionFilename;
  G4String                mSingleInteractionType;
  G4ThreeVector           mSingleInteractionPosition;
  G4ThreeVector           mSingleInteractionDirection;
  G4double                mSingleInteractionEnergy;
  G4int                   mSingleInteractionZ;

  // Phase space variables
  G4String mPhaseSpaceFilename;
  TFile   *mPhaseSpaceFile;
  TTree   *mPhaseSpace;
  G4ThreeVector mInteractionDirection;
  G4ThreeVector mInteractionPosition;
  double        mInteractionEnergy;
  double        mInteractionWeight;
  Char_t        mInteractionProductionVolume[256];
  Char_t        mInteractionProductionProcessTrack[256];
  Char_t        mInteractionProductionProcessStep[256];
  int           mInteractionTrackId;
  int           mInteractionEventId;
  int           mInteractionRunId;
  double        mInteractionTotalContribution;
  double        mInteractionSquaredContribution;
  Char_t        mInteractionVolume[256];
  Char_t        mInteractionMaterial[256];
  int           mInteractionZ;
  int           mInteractionOrder;

  // Water equivalent conversion
  void CreateWaterLUT(const std::vector<double> &energyList,
                      const std::vector<double> &energyWeightList);

  // Account for primary fluence weighting
  InputImageType::Pointer PrimaryFluenceWeighting(const InputImageType::Pointer input);

  // Second pass members
  G4String AddPrefix(G4String prefix, G4String filename);
  OutputImageType::Pointer CreateRussianRouletteVoidImage();
  G4String mSecondPassPrefix;
  G4String mRussianRouletteFilename;
  G4ThreeVector mSecondPassDetectorResolution;
  std::map<ProcessType, OutputImageType::Pointer> mRussianRouletteImages;
  std::map<ProcessType, OutputImageType::Pointer> mRussianRouletteSquaredImages;
  std::map<ProcessType, OutputImageType::Pointer> mRussianRouletteCountImages;
  std::map<ProcessType, OutputImageType::Pointer> mRussianRouletteProbabilityImages;
  G4double mRussianRouletteSpacing;
  G4int mRussianRouletteMinimumCountInRegion;
  G4double mRussianRouletteMinimumProbability;
  TFile   *mSecondPassPhaseSpaceFile;
  TTree   *mSecondPassPhaseSpace;
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(HybridForcedDetectionActor, GateHybridForcedDetectionActor)


#endif /* end #define GATEHYBRIDFORCEDDECTECTIONACTOR_HH */

#endif // GATE_USE_RTK
