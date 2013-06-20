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

// itk
#include <itkTimeProbe.h>

// rtk
#include <rtkConstantImageSource.h>
#include <rtkReg23ProjectionGeometry.h>
#include <rtkJosephForwardProjectionImageFilter.h>

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
  virtual void BeginOfEventAction(const G4Event*);
  // virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*);
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*); 

  /// Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  // Resolution of the detector plane (2D only, z=1);
  const G4ThreeVector & GetDetectorResolution() const { return mDetectorResolution; }
  void SetDetectorResolution(int x, int y) { mDetectorResolution[0] = x; mDetectorResolution[1] = y; }
  void SetDetectorVolumeName(G4String name) { mDetectorName = name; }
  void SetGeometryFilename(G4String name) { mGeometryFilename = name; }
  void SetPrimaryFilename(G4String name) { mPrimaryFilename = name; }
  void SetMaterialMuFilename(G4String name) { mMaterialMuFilename = name; }
  void SetAttenuationFilename(G4String name) { mAttenuationFilename = name; }
  void SetFlatFieldFilename(G4String name) { mFlatFieldFilename = name; }
  void SetComptonFilename(G4String name) { mComptonFilename = name; }
  void SetRayleighFilename(G4String name) { mRayleighFilename = name; }
  void SetFluorescenceFilename(G4String name) { mFluorescenceFilename = name; }
  void SetSingleInteractionFilename(G4String name) { mSingleInteractionFilename = name; }
  void SetSingleInteractionType(G4String type) { mSingleInteractionType = type; }
  void SetSingleInteractionPosition(G4ThreeVector pos) { mSingleInteractionPosition = pos; }
  void SetSingleInteractionDirection(G4ThreeVector dir) { mSingleInteractionDirection = dir; }
  void SetSingleInteractionEnergy(G4double e) { mSingleInteractionEnergy = e; }
  void SetSingleInteractionZ(G4int z) { mSingleInteractionZ = z; }
  void SetPhaseSpaceFilename(G4String name) { mPhaseSpaceFilename = name; }

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
  
  void ComputeGeometryInfoInImageCoordinateSystem(GateVImageVolume *image,
                                                  GateVVolume *detector,
                                                  GateVSource *src,
                                                  PointType &primarySourcePosition,
                                                  PointType &detectorPosition,
                                                  VectorType &detectorRowVector,
                                                  VectorType &detectorColVector);
  InputImageType::Pointer ConvertGateImageToITKImage(GateVImageVolume * gateImgVol);
  InputImageType::Pointer CreateVoidProjectionImage();

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
  G4String mFlatFieldFilename;
  G4String mComptonFilename;
  G4String mRayleighFilename;
  G4String mFluorescenceFilename;

  G4ThreeVector mDetectorResolution;

  GeometryType::Pointer mGeometry;
  InputImageType::Pointer mGateVolumeImage;
  InputImageType::Pointer mPrimaryImage;
  InputImageType::Pointer mFlatFieldImage;
  InputImageType::Pointer mComptonImage;
  InputImageType::Pointer mRayleighImage;
  InputImageType::Pointer mFluorescenceImage;
  std::vector<InputImageType::Pointer> mComptonPerOrderImages;
  std::vector<InputImageType::Pointer> mRayleighPerOrderImages;
  std::vector<InputImageType::Pointer> mFluorescencePerOrderImages;

  // Geometry information initialized at the beginning of the run
  G4AffineTransform m_WorldToCT;
  PointType mDetectorPosition;
  VectorType mDetectorRowVector;
  VectorType mDetectorColVector;

  // Accumulation type
  typedef GateHybridForcedDetectionFunctor::VAccumulation AccumulationType;

  // Primary stuff
  itk::TimeProbe mPrimaryProbe;
  unsigned int mNumberOfEventsInRun;
  typedef rtk::JosephForwardProjectionImageFilter<
                 InputImageType,
                 InputImageType,
                 GateHybridForcedDetectionFunctor::InterpolationWeightMultiplication,
                 GateHybridForcedDetectionFunctor::PrimaryValueAccumulation>
                   PrimaryProjectionType;

  // Compton stuff
  itk::TimeProbe mComptonProbe;
  typedef rtk::JosephForwardProjectionImageFilter<
                 InputImageType,
                 InputImageType,
                 GateHybridForcedDetectionFunctor::InterpolationWeightMultiplication,
                 GateHybridForcedDetectionFunctor::ComptonValueAccumulation>
                   ComptonProjectionType;
  ComptonProjectionType::Pointer mComptonProjector;

  // Rayleigh stuff
  itk::TimeProbe mRayleighProbe;
  typedef rtk::JosephForwardProjectionImageFilter<
                 InputImageType,
                 InputImageType,
                 GateHybridForcedDetectionFunctor::InterpolationWeightMultiplication,
                 GateHybridForcedDetectionFunctor::RayleighValueAccumulation>
                   RayleighProjectionType;
  RayleighProjectionType::Pointer mRayleighProjector;

  // Fluorescence stuff
  itk::TimeProbe mFluorescenceProbe;
  typedef rtk::JosephForwardProjectionImageFilter< 
                 InputImageType,
                 InputImageType,
                 GateHybridForcedDetectionFunctor::InterpolationWeightMultiplication,
                 GateHybridForcedDetectionFunctor::FluorescenceValueAccumulation> 
                   FluorescenceProjectionType;
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
  G4int         mInteractionZ;
  Char_t        mInteractionProductionVolume[256];
  Char_t        mInteractionProductionProcessTrack[256];
  Char_t        mInteractionProductionProcessStep[256];
  int           mInteractionTrackId;
  int           mInteractionEventId;
  int           mInteractionRunId;
  double        mInteractionTotalContribution;
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(HybridForcedDetectionActor, GateHybridForcedDetectionActor)


#endif /* end #define GATEHYBRIDFORCEDDECTECTIONACTOR_HH */

#endif // GATE_USE_RTK
