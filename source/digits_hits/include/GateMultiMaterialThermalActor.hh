/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \class GateMultiMaterialThermalActor
  \author fsmekens@gmail.com
  \brief Class GateMultiMaterialThermalActor : This actor produces voxelised images of the heat diffusion in tissue.

*/

#ifndef GATEMULTIMATERIALTHERMALACTOR_HH
#define GATEMULTIMATERIALTHERMALACTOR_HH

#include <G4NistManager.hh>
#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "G4UnitsTable.hh"
#include "GateMultiMaterialThermalActorMessenger.hh"
#include "GateImageWithStatistic.hh"
#include "GateVImageVolume.hh"

#include "G4Event.hh"
#include <time.h>

// itk
#include "GateConfiguration.h"
#ifdef GATE_USE_ITK
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageIterator.h"
#include "itkRecursiveGaussianImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkExpImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkImportImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkPasteImageFilter.h"

#endif

class GateMultiMaterialThermalActor : public GateVImageActor
{
public:
  //-----------------------------------------------------------------------------
  // Actor name
  virtual ~GateMultiMaterialThermalActor();

  FCT_FOR_AUTO_CREATOR_ACTOR(GateMultiMaterialThermalActor)

  // typedef for itk
  typedef itk::Image<float, 3> FloatImageType;
  typedef itk::Image<double, 3> DoubleImageType;
  typedef itk::ImageRegionIterator<FloatImageType> FloatIteratorType;
  typedef itk::ImageRegionIterator<DoubleImageType> DoubleIteratorType;
  typedef itk::ImageDuplicator<FloatImageType> FloatDuplicatorType;
  typedef itk::ImageDuplicator<DoubleImageType> DoubleDuplicatorType;
  typedef itk::ImageFileReader<DoubleImageType> DoubleReaderType;
  typedef itk::MultiplyImageFilter<DoubleImageType, FloatImageType, DoubleImageType> MultiplyFilterType;
  typedef itk::ExpImageFilter<DoubleImageType, FloatImageType> ExpFilterType;
  typedef itk::AddImageFilter<DoubleImageType, DoubleImageType, DoubleImageType> AddImageFilterType;
  typedef itk::SubtractImageFilter<DoubleImageType, DoubleImageType> SubtractImageFilterType;
  typedef itk::BinaryThresholdImageFilter<FloatImageType, FloatImageType> BinaryThresholdFilterType;
  typedef itk::RecursiveGaussianImageFilter<DoubleImageType, DoubleImageType> GaussianFilterType;
  typedef itk::ImportImageFilter<float, 3> ImportFilterType;
  typedef itk::RegionOfInterestImageFilter<FloatImageType, FloatImageType> FloatToFloatROIFilterType;
  typedef itk::RegionOfInterestImageFilter<DoubleImageType, DoubleImageType> DoubleToDoubleROIFilterType;
  typedef itk::PasteImageFilter<DoubleImageType> PasteImageFilterType;
  
  //-----------------------------------------------------------------------------
  // Defines ROI for thermal diffusion and corresponding thermal diffusion periodicity
  struct DiffusionStruct {
    double diffusivity;
    double period;
    double sigma;
    bool isROIused;
    FloatImageType::RegionType regionOfInterest;
    FloatImageType::Pointer mask;
    double timer;
    double currentTimeStep;
    double totalTime;
    int diffusionNumber;
    
    DiffusionStruct(double c, double s, bool useROI, FloatImageType::RegionType r, FloatImageType::Pointer m) {
      diffusivity = c;
      sigma = s;
      isROIused = useROI;
      regionOfInterest = r;
      mask = m;
      period = sigma * sigma / (2.0 * diffusivity);
      timer = 0.0;
      currentTimeStep = 0.0;
      totalTime = 0.0;
      diffusionNumber = 0;
    }
    
    bool CheckDiffusionTime(double stepTime, bool forced) {
      timer += stepTime;
      totalTime += stepTime;
      sigma = sqrt(2.0 * timer * diffusivity);
      if((timer >= period or forced) and sigma > 0.0) {
        currentTimeStep = timer;
        timer = 0.0;
        return true;
      }
      else { return false; }
    }
  };
  
  //-----------------------------------------------------------------------------
  // Defines rectangle ROI and corresponding measurement periodicity 
  struct MeasurementStruct {
    int label;
    double period;
    double timer;
    double totalTime;
    std::vector<DoubleImageType::IndexType> indexList;
    std::vector<double> timeList;
    std::vector<double> measList;
    
    MeasurementStruct(int l, double t) {
      label = l;
      period = t;
      timer = 0.0;
      totalTime = 0.0;
      indexList.clear();
      timeList.clear();
      measList.clear();
    }
    
    bool CheckMeasurementTime(double stepTime, bool forced) {
      timer += stepTime;
      totalTime += stepTime;
      if(timer >= period or forced)
      {
        timer = 0.0;
        return true;
      }
      else { return false; }
    }
    
    void SetValue(double t, double m) {
      timeList.push_back(t);
      measList.push_back(m);
    }
  };
  
  //-----------------------------------------------------------------------------
  // Constructs the sensor
  virtual void Construct();

  virtual void BeginOfRunAction(const G4Run *r);
  virtual void EndOfRunAction(const G4Run *);
  virtual void BeginOfEventAction(const G4Event *event);
  virtual void EndOfEventAction(const G4Event *event);
  virtual void UserSteppingActionInVoxel(const int index, const G4Step *step);
  virtual void UserPreTrackActionInVoxel(const int, const G4Track *track);
  virtual void UserPostTrackActionInVoxel(const int, const G4Track *) {}

  //  Saves the data collected to the file
  virtual void SaveData();
  virtual void ResetData();

  // Scorer related
  virtual void Initialize(G4HCofThisEvent*) {}
  virtual void EndOfEvent(G4HCofThisEvent*) {}

  // Main functions for applying the thermal diffusion and perfussion
  void ConstructRegionMasks(GateVImageVolume *);
  void ConstructPerfusionMap();
  void ReadMeasurementFile(DoubleImageType::Pointer);
  double GetPropertyFromMaterial(const G4Material *, G4String, G4double);
  void ApplyStepPerfusion(double, bool);
  void ApplyStepDiffusion(double, bool);
  void ApplyStepMeasurement(double, bool);
  void ApplyUserRelaxation();
  
  // images management
  FloatImageType::Pointer ConvertGateToITKImage_float(GateImage *);
  DoubleImageType::Pointer ConvertGateToITKImage_double(GateImageDouble *);
  void SaveITKimage(FloatImageType::Pointer, G4String);
  void SaveITKimage(DoubleImageType::Pointer, G4String);
  
  // Complex 'set' functions
  void SetBloodPerfusionByMaterial(G4bool);
  void SetBloodPerfusionByConstant(G4double);
  void SetBloodPerfusionByImage(G4String);
  void SetMeasurementFilename(G4String);
  // Basic 'set' functions
  void setRelaxationTime(G4double t) { mUserRelaxationTime = t; }
  void setDiffusivity(G4double d) { mUserMaterialDiffusivity = d; }
  void setBloodDensity(G4double d) { mUserBloodDensity = d; }
  void setBloodHeatCapacity(G4double c) { mUserBloodHeatCapacity = c; }
  void setTissueDensity(G4double d) { mUserTissueDensity = d; }
  void setTissueHeatCapacity(G4double c) { mUserTissueHeatCapacity = c; }
  void enableStepDiffusion(G4bool b) { mIsDiffusionActivated = b; }
  
protected:

  GateMultiMaterialThermalActor(G4String name, G4int depth=0);
  GateMultiMaterialThermalActorMessenger * pMessenger;

  int mCurrentEvent;
  StepHitType mUserStepHitType;

  // image data
  GateImageWithStatistic mAbsorptionImage;
  G4String mAbsorptionFilename;
  G4String mHeatAbsorptionFilename;
  G4String mHeatRelaxationFilename;
  DoubleImageType::Pointer mITKheatMap;
  DoubleImageType::Pointer mITKperfusionRateMap;
  DoubleImageType::Pointer mITKheatConversionMap;

  // time data
  double mCurrentTime;
  double mPreviousTime;
  double mPerfusionTimer;
  double mUserRelaxationTime;
  std::vector<G4Material *> mMaterialList;
  std::map<G4Material *, DiffusionStruct> mMaterialToDiffusionStruct;
  double mMinTimeStep;

  // activation flags
  bool mIsPerfusionActivated;
  bool mIsPerfusionByMaterial;
  bool mIsPerfusionByConstant;
  bool mIsPerfusionByImage;
  bool mIsDiffusionActivated;  
  bool mIsMeasurementActivated;
  G4String mMeasurementFilename;
  std::vector<MeasurementStruct> mMeasurementPoints;

  // constant values
  G4String mUserPerfusionImageName;
  double mUserBloodPerfusionRate;
  double mUserBloodDensity;
  double mUserBloodHeatCapacity;
  double mUserTissueDensity;
  double mUserTissueHeatCapacity;
  double mMinPerfusionCoef;
  double mMaxPerfusionCoef;
  double mPerfusionRatio;
  double mMinPerfusionTimeStep;
  double mUserMaterialDiffusivity;
  
};

MAKE_AUTO_CREATOR_ACTOR(MultiMaterialThermalActor,GateMultiMaterialThermalActor)

#endif /* end #define GATEMULTIMATERIALTHERMALACTOR_HH */
