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

// rtk
#include "rtkProjectionGeometry.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkConstantImageSource.h"

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

  // Typedef for rtk
  static const unsigned int Dimension = 3;
  typedef float                                  InputPixelType;
  typedef itk::Image<InputPixelType, Dimension>  InputImageType;
  typedef float                                  OutputPixelType;
  typedef itk::Image<OutputPixelType, Dimension> OutputImageType;
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  
  void CreateMuImage(const std::vector<double> & label2mu, 
                     const GateImage * gate_image, 
                     InputImageType * input);
  void GenerateDRR(const InputImageType * input, 
                   const OutputImageType * projInput, 
                   GeometryType * geometry,
                   OutputImageType * output);
  OutputImageType::Pointer CreateGeometry(GateVVolume * detector,
                                          GateVSource * src, 
                                          GeometryType * geometry);
  void CreateLabelToMuConversion(const double E, std::vector<double> & label2mu);
    
protected:
  GateHybridForcedDetectionActorMessenger * pActorMessenger;
  
  G4String mDetectorName;
  GateVVolume * mDetector;
  GateVSource * mSource;
  
  G4ThreeVector mDetectorResolution;
  std::vector<double> mEnergyList;

};
//-----------------------------------------------------------------------------

MAKE_AUTO_CREATOR_ACTOR(HybridForcedDetectionActor,GateHybridForcedDetectionActor)


#endif /* end #define GATEHYBRIDFORCEDDECTECTIONACTOR_HH */

#endif // GATE_USE_RTK
