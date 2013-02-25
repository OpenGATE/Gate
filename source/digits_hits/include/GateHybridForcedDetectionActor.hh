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

// rtk
#include "rtkProjectionGeometry.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkReg23ProjectionGeometry.h"

// itk
#include "itkImageFileWriter.h"

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
  // virtual void BeginOfEventAction(const G4Event*);
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
  void CreateLabelToMuConversion(const std::vector<double> &E,
                                 GateVImageVolume * gate_image_volume);
  InputImageType::Pointer ConvertGateImageToITKImage(GateImage * gateImg);
  InputImageType::Pointer CreateVoidProjectionImage();

protected:
  GateHybridForcedDetectionActorMessenger * pActorMessenger;
  
  G4String mDetectorName;
  GateVVolume * mDetector;
  GateVSource * mSource;
  G4String mGeometryFilename;
  G4String mPrimaryFilename;
  G4String mMaterialMuFilename;

  G4ThreeVector mDetectorResolution;

  GeometryType::Pointer mGeometry;
  InputImageType::Pointer mPrimaryImage;
  itk::Image<double, 2>::Pointer mMaterialMu;

  // Geometry information initialized at the beginning of the run
  PointType mDetectorPosition;
  VectorType mDetectorRowVector;
  VectorType mDetectorColVector;

  // Callback classes for forward projection
  class PrimaryInterpolationWeightMultiplication;
  class PrimaryValueAccumulation;
};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(HybridForcedDetectionActor, GateHybridForcedDetectionActor)


#endif /* end #define GATEHYBRIDFORCEDDECTECTIONACTOR_HH */

#endif // GATE_USE_RTK
