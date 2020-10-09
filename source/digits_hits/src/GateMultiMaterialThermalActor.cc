/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

// This actor is only compiled if ITK is available
#include "GateConfiguration.h"
#ifdef  GATE_USE_ITK

/*
  \class GateMultiMaterialThermalActor
  \author fsmekens@gmail.com
  \brief Class GateMultiMaterialThermalActor : This actor produces voxelised images of the heat diffusion in tissue.
*/

#include <G4VoxelLimits.hh>
#include <G4NistManager.hh>

#include "GateMultiMaterialThermalActor.hh"
#include "GateMiscFunctions.hh"
#include "G4VProcess.hh"
#include "GateMHDImage.hh"
#include "GateImageT.hh"
#include "GateMiscFunctions.hh"
#include "GateMachine.hh"
#include "GateApplicationMgr.hh"
#include <sys/time.h>
#include <iostream>
#include <string>

//-----------------------------------------------------------------------------

GateMultiMaterialThermalActor::GateMultiMaterialThermalActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateMultiMaterialThermalActor() -- begin"<<G4endl);

  mCurrentEvent=-1;
  mUserRelaxationTime = -1.0;
  mIsDiffusionActivated = false;
  mIsPerfusionByMaterial = false;
  mIsPerfusionActivated = false;
  mIsPerfusionByMaterial = false;
  mIsPerfusionByConstant = false;
  mIsPerfusionByImage = false;
  mIsMeasurementActivated = false;
  mMeasurementFilename = "";
  mPerfusionRatio = 0.99;
  mUserPerfusionImageName = "";
  mUserBloodPerfusionRate = 0.0;
  mUserBloodDensity = -1.0;
  mUserBloodHeatCapacity = -1.0;
  mUserTissueHeatCapacity = -1.0;

  pMessenger = new GateMultiMaterialThermalActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateMultiMaterialThermalActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::SetBloodPerfusionByMaterial(G4bool b)
{
  mIsPerfusionActivated = b;
  mIsPerfusionByMaterial = b;
  mIsPerfusionByConstant = false;
  mIsPerfusionByImage = false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::SetBloodPerfusionByConstant(G4double value)
{
  mIsPerfusionActivated = true;
  mIsPerfusionByMaterial = false;
  mIsPerfusionByConstant = true;
  mIsPerfusionByImage = false;
  mUserBloodPerfusionRate = value;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::SetBloodPerfusionByImage(G4String string)
{
  mIsPerfusionActivated = true;
  mIsPerfusionByMaterial = false;
  mIsPerfusionByConstant = false;
  mIsPerfusionByImage = true;
  mUserPerfusionImageName = string;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::SetMeasurementFilename(G4String string)
{
  mIsMeasurementActivated = true;
  mMeasurementFilename = string;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Destructor
GateMultiMaterialThermalActor::~GateMultiMaterialThermalActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Constructor
void GateMultiMaterialThermalActor::Construct()
{
  GateDebugMessageInc("Actor", 4, "GateMultiMaterialThermalActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct();

  // Record the stepHitType
  mUserStepHitType = mStepHitType;

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableEndOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);

  // Output Filenames
  mAbsorptionFilename = G4String(removeExtension(mSaveFilename))+"-AbsorptionMap."+G4String(getExtension(mSaveFilename));
  mHeatAbsorptionFilename = G4String(removeExtension(mSaveFilename))+"-HeatAbsorptionMap."+G4String(getExtension(mSaveFilename));
  mHeatRelaxationFilename = G4String(removeExtension(mSaveFilename))+"-HeatRelaxationMap."+G4String(getExtension(mSaveFilename));

  // Set origin, transform, flag
  SetOriginTransformAndFlagToImage(mAbsorptionImage);

  // Resize and allocate images
  mAbsorptionImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mAbsorptionImage.Allocate();
  mAbsorptionImage.SetFilename(mAbsorptionFilename);

  // initialize ITK heat map from actor energy map
  GateImageDouble *energyMap = dynamic_cast<GateImageDouble *>(&(mAbsorptionImage.GetValueImage()));
  DoubleDuplicatorType::Pointer doubleDuplicatorFilter = DoubleDuplicatorType::New();
  doubleDuplicatorFilter->SetInputImage(ConvertGateToITKImage_double(energyMap));
  doubleDuplicatorFilter->Update();
  mITKheatMap = doubleDuplicatorFilter->GetOutput();
  mITKheatMap->DisconnectPipeline();
  
  if(mIsMeasurementActivated) { ReadMeasurementFile(mITKheatMap); }
  
  // construct diffusion masks
  GateVImageVolume *gateVoxelisedMap = dynamic_cast<GateVImageVolume *>(GetVolume());
  if(!gateVoxelisedMap) { GateError("Error: in its actual version, only voxelised volume can be used as attached volume."); }
  else { ConstructRegionMasks(gateVoxelisedMap); }

  mCurrentTime = GateApplicationMgr::GetInstance()->GetTimeStart();

  // Print information
  GateMessage("Actor", 1,
              "\tThermalActor    = '" << GetObjectName() << "'" << G4endl <<
              "\tAbsorptionFilename      = " << mAbsorptionFilename << G4endl);

  ResetData();
  GateMessageDec("Actor", 4, "GateMultiMaterialThermalActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateMultiMaterialThermalActor::SaveData()
{
  mAbsorptionImage.SaveData(mCurrentEvent+1);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::ResetData()
{
  mAbsorptionImage.Reset();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::BeginOfRunAction(const G4Run * r) {  
  GateVActor::BeginOfRunAction(r);
  
  mCurrentTime = GateApplicationMgr::GetInstance()->GetCurrentTime();

  GateDebugMessage("Actor", 3, "GateMultiMaterialThermalActor -- Begin of Run" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);

  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateMultiMaterialThermalActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::EndOfEventAction(const G4Event *)
{
  double currentTime = GateApplicationMgr::GetInstance()->GetCurrentTime();  
  double tmpTime = currentTime-mCurrentTime;
  
  if(mIsDiffusionActivated) { ApplyStepDiffusion(tmpTime, false); }
  if(mIsPerfusionActivated) { ApplyStepPerfusion(tmpTime, false); }
  if(mIsMeasurementActivated) { ApplyStepMeasurement(tmpTime, false); }
  
  mCurrentTime = currentTime;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::EndOfRunAction(const G4Run* r)
{
  GateVActor::EndOfRunAction(r);

  DD("EndOfRunAction::Begin");
  
  double currentTime = GateApplicationMgr::GetInstance()->GetCurrentTime();  
  double tmpTime = currentTime-mCurrentTime;

  if(mIsDiffusionActivated) { ApplyStepDiffusion(tmpTime, true); }
  if(mIsPerfusionActivated) { ApplyStepPerfusion(tmpTime, true); }
  if(mIsMeasurementActivated) { ApplyStepMeasurement(tmpTime, true); }
  
  mCurrentTime = currentTime;

  SaveITKimage(mITKheatMap, mHeatAbsorptionFilename);
  
  if(mUserRelaxationTime > 0.0) { ApplyUserRelaxation(); }
}

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::UserPreTrackActionInVoxel(const int /*index*/, const G4Track* track) {

  if(track->GetDefinition()->GetParticleName() == "opticalphoton") { mStepHitType = PostStepHitType; }
  else { mStepHitType = mUserStepHitType; }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {

  GateDebugMessageInc("Actor", 4, "GateMultiMaterialThermalActor -- UserSteppingActionInVoxel - begin" << G4endl);
  
  const double edep = step->GetTotalEnergyDeposit();  
  const G4String process = step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName();
  
  // if no energy is deposited or energy is deposited outside image => do nothing
  if (step->GetPostStepPoint()->GetKineticEnergy() == 0) {
    GateDebugMessage("Actor", 5, "edep == 0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateMultiMaterialThermalActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  if (index <0) {
    GateDebugMessage("Actor", 5, "index<0 : do nothing" << G4endl);
    GateDebugMessageDec("Actor", 4, "GateMultiMaterialThermalActor -- UserSteppingActionInVoxel -- end" << G4endl);
    return;
  }

  GateDebugMessage("Actor", 2, "GateMultiMaterialThermalActor -- UserSteppingActionInVoxel:\tedep = " << G4BestUnit(edep, "Energy") << G4endl);

  if ( edep > 0.0 )
  {
    // add energy in the gate image
    mAbsorptionImage.AddValue(index, edep);

    // add energy in the ITK image (for diffusion and perfusion)
    G4ThreeVector gatePixelCoordinate = mImage.GetCoordinatesFromIndex(index);
    DoubleImageType::IndexType indexITK;
    indexITK[0] = gatePixelCoordinate.getX();
    indexITK[1] = gatePixelCoordinate.getY();
    indexITK[2] = gatePixelCoordinate.getZ();
    double heatValue = mITKheatMap->GetPixel(indexITK);
    double heatConversion = mITKheatConversionMap->GetPixel(indexITK);
    mITKheatMap->SetPixel(indexITK, heatValue + edep*heatConversion);
  }
  
  GateDebugMessageDec("Actor", 4, "GateMultiMaterialThermalActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::ApplyStepDiffusion(double timeStep, bool forced)
{
  DoubleImageType::Pointer tmpHeatMap = mITKheatMap;
  tmpHeatMap->DisconnectPipeline();

  std::map<G4Material *, DiffusionStruct>::iterator itMap;
  std::map<G4Material *, DiffusionStruct>::iterator itBegin = mMaterialToDiffusionStruct.begin();
  std::map<G4Material *, DiffusionStruct>::iterator itEnd = mMaterialToDiffusionStruct.end();

  for(itMap = itBegin; itMap != itEnd; ++itMap)
  {
    bool checkDiffusion = itMap->second.CheckDiffusionTime(timeStep, forced);    
    if(checkDiffusion)
    {
      DoubleToDoubleROIFilterType::Pointer roiFilter1 = DoubleToDoubleROIFilterType::New();
      DoubleToDoubleROIFilterType::Pointer roiFilter2 = DoubleToDoubleROIFilterType::New();
      AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
      MultiplyFilterType::Pointer multiplyFilter = MultiplyFilterType::New();
      SubtractImageFilterType::Pointer subtractFilter = SubtractImageFilterType::New();
      PasteImageFilterType::Pointer pasteFilter = PasteImageFilterType::New();
      GaussianFilterType::Pointer gaussianFilterX = GaussianFilterType::New();
      GaussianFilterType::Pointer gaussianFilterY = GaussianFilterType::New();
      GaussianFilterType::Pointer gaussianFilterZ = GaussianFilterType::New();
      gaussianFilterX->SetDirection(0);
      gaussianFilterY->SetDirection(1);
      gaussianFilterZ->SetDirection(2);
      double sigma = itMap->second.sigma;
      gaussianFilterX->SetSigma(sigma);
      gaussianFilterY->SetSigma(sigma);
      gaussianFilterZ->SetSigma(sigma);
      gaussianFilterX->SetOrder(itk::GaussianOrderEnum::ZeroOrder);
      gaussianFilterY->SetOrder(itk::GaussianOrderEnum::ZeroOrder);
      gaussianFilterZ->SetOrder(itk::GaussianOrderEnum::ZeroOrder);
      gaussianFilterX->SetNormalizeAcrossScale(false);
      gaussianFilterY->SetNormalizeAcrossScale(false);
      gaussianFilterZ->SetNormalizeAcrossScale(false);
      
      G4cout << "useROI " << itMap->second.isROIused << " | mat "<< itMap->first->GetName() << " is diffusing | sigma = " << itMap->second.sigma / mm << " mm | currentTimeStep = " << itMap->second.currentTimeStep / s << " s | timeStep = " << timeStep / s << " s | currentTime = " << mCurrentTime / s << " s" << G4endl;
      itMap->second.diffusionNumber++;
      
      if(itMap->second.isROIused)
      { 
        // 1. ROIs on original (tmpHeatMap) and currently diffused (mITKheatMap) energyMaps
        roiFilter1->SetInput(tmpHeatMap);
        roiFilter1->SetRegionOfInterest(itMap->second.regionOfInterest);
        roiFilter1->Update();
        roiFilter2->SetInput(mITKheatMap);
        roiFilter2->SetRegionOfInterest(itMap->second.regionOfInterest);
        roiFilter2->Update();
        // 2. multiply original map by the mask
        multiplyFilter->SetInput1(roiFilter1->GetOutput());
        multiplyFilter->SetInput2(itMap->second.mask);
        multiplyFilter->Update();
        // 3. subtract masked original map of the diffused map
        subtractFilter->SetInput1(roiFilter2->GetOutput());
        subtractFilter->SetInput2(multiplyFilter->GetOutput());
        subtractFilter->Update();
        // 4. apply recursive gaussian filter with corresponding diffusivity
        gaussianFilterX->SetInput(multiplyFilter->GetOutput());
        gaussianFilterX->Update();
        gaussianFilterY->SetInput(gaussianFilterX->GetOutput());
        gaussianFilterY->Update();
        gaussianFilterZ->SetInput(gaussianFilterY->GetOutput());
        gaussianFilterZ->Update();
        // 5. add the result in the diffused map
        addFilter->SetInput1(subtractFilter->GetOutput());
        addFilter->SetInput2(gaussianFilterZ->GetOutput());
        addFilter->Update();
        // 6. paste the result (ROI) in the global map
        pasteFilter->SetSourceImage(addFilter->GetOutput());
        pasteFilter->SetSourceRegion(addFilter->GetOutput()->GetLargestPossibleRegion());
        pasteFilter->SetDestinationImage(mITKheatMap);
        pasteFilter->SetDestinationIndex(itMap->second.regionOfInterest.GetIndex());
        pasteFilter->Update();
        // 7. update final map
        mITKheatMap = pasteFilter->GetOutput();
        mITKheatMap->DisconnectPipeline();
      }
      else
      {
        // 1. multiply mask with energy map
        multiplyFilter->SetInput1(tmpHeatMap);
        multiplyFilter->SetInput2(itMap->second.mask);
        multiplyFilter->Update();
        // 2. subtract masked original map of the diffused map
        subtractFilter->SetInput1(mITKheatMap);
        subtractFilter->SetInput2(multiplyFilter->GetOutput());
        subtractFilter->Update();
        // 3. apply recursive gaussian filter with corresponding diffusivity
        gaussianFilterX->SetInput(multiplyFilter->GetOutput());
        gaussianFilterX->Update();
        gaussianFilterY->SetInput(gaussianFilterX->GetOutput());
        gaussianFilterY->Update();
        gaussianFilterZ->SetInput(gaussianFilterY->GetOutput());
        gaussianFilterZ->Update();
        // 4. add the result in the diffused map
        addFilter->SetInput1(subtractFilter->GetOutput());
        addFilter->SetInput2(gaussianFilterZ->GetOutput());
        addFilter->Update();
        // 5. update final map
        mITKheatMap = addFilter->GetOutput();
        mITKheatMap->DisconnectPipeline();
      }
    }
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

void GateMultiMaterialThermalActor::ApplyUserRelaxation()
{
  // -------------------------------------------------------------------------------
  // mimic time
  
  double totalTime = mCurrentTime + mUserRelaxationTime;

  if(mIsDiffusionActivated)
  {
    while(mCurrentTime < totalTime)
    {
      ApplyStepDiffusion(mMinTimeStep, false);
      if(mIsPerfusionActivated) { ApplyStepPerfusion(mMinTimeStep, false); }
      if(mIsMeasurementActivated) { ApplyStepMeasurement(mMinTimeStep, false); }

      mCurrentTime += mMinTimeStep;
    }

    std::map<G4Material *, DiffusionStruct>::iterator itMap;
    std::map<G4Material *, DiffusionStruct>::iterator itBegin = mMaterialToDiffusionStruct.begin();
    std::map<G4Material *, DiffusionStruct>::iterator itEnd = mMaterialToDiffusionStruct.end();
    for(itMap = itBegin; itMap != itEnd; ++itMap)
    {
      G4cout << "mat " << itMap->first->GetName() <<" has diffused " << itMap->second.diffusionNumber << " times | totalTime = " << itMap->second.totalTime / s << " s" << G4endl;
    }    
  }
  else if(mIsPerfusionActivated) {
    ApplyStepPerfusion(mUserRelaxationTime, true);
  }

  // save heat map and local measurement
  SaveITKimage(mITKheatMap, mHeatRelaxationFilename);
  for(unsigned int i=0; i<mMeasurementPoints.size(); i++)
  {
    std::vector<double> timeList = mMeasurementPoints[i].timeList;
    std::vector<double> measList = mMeasurementPoints[i].measList;

    G4String filename = G4String(removeExtension(mSaveFilename))+"-ROI-"+ std::to_string(mMeasurementPoints[i].label) +".txt";
    std::ofstream os(filename);
    
    os << "time(s) \tenergy(eV) \tNvoxel="<< mMeasurementPoints[i].indexList.size() << std::endl;
    for(unsigned int j=0; j<timeList.size(); j++)
    {
      os.precision(10);
      os << timeList[j] / s << "\t" << measList[j] << "\t" << std::endl;
    }
    os.close();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::ApplyStepPerfusion(double timeStep, bool forced)
{
  mPerfusionTimer += timeStep;
  if(mPerfusionTimer >= mMinPerfusionTimeStep or forced)
  {
    G4cout << "blood perfusion | currentTimeStep = " << mPerfusionTimer / s << " s | timeStep = " << timeStep / s << " s | currentTime = " << mCurrentTime / s << " s" << G4endl;
    
    MultiplyFilterType::Pointer multConst = MultiplyFilterType::New();
    MultiplyFilterType::Pointer multImage = MultiplyFilterType::New();
    ExpFilterType::Pointer expImage = ExpFilterType::New();
    
    multConst->SetInput(mITKperfusionRateMap);
    multConst->SetConstant(-mPerfusionTimer);
    expImage->SetInput(multConst->GetOutput());
    multImage->SetInput1(mITKheatMap);
    multImage->SetInput2(expImage->GetOutput());
    multImage->Update();
    
    mITKheatMap = multImage->GetOutput();
    mITKheatMap->DisconnectPipeline();
    
    mPerfusionTimer = 0.0;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::ApplyStepMeasurement(double timeStep, bool forced)
{
  for(unsigned int i=0; i<mMeasurementPoints.size(); i++)
  {
    bool check = mMeasurementPoints[i].CheckMeasurementTime(timeStep, forced);
    if(check)
    {
      double value = 0.0;
      std::vector<DoubleImageType::IndexType> indices = mMeasurementPoints[i].indexList;
      for(unsigned j=0; j<indices.size(); j++) { value += mITKheatMap->GetPixel(indices[j]); }
      mMeasurementPoints[i].SetValue(mMeasurementPoints[i].totalTime, value);
    }
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateMultiMaterialThermalActor::GetPropertyFromMaterial(const G4Material *mat, G4String prop, G4double unit)
{
  G4MaterialPropertiesTable *materialPropertyTable = mat->GetMaterialPropertiesTable();
  if(materialPropertyTable) { return materialPropertyTable->GetConstProperty(prop) * unit; }
  else { return 0.0; }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::ConstructRegionMasks(GateVImageVolume *gateImage)
{
  // This function creates a 'diffusion struct' (see .hh) for each material in the
  // considered voxelised volume. Each struct is composed of:
  // 1. diffusivity/period (constant values)
  // 2. regionOfInterest in the energyMap
  // 3. image mask (1 if corresponding material, 0 otherwise)
  // 4. timer/sigma (variable values)
  //
  // Additionaly, an eV->Kelvin conversion map is created to convert inline the energyMap into heatMap
  // Finally, a perfusionRate map is constructed following the method chosen by the user ('image', 'fixedValue' or 'byMaterial')
  
  // Create an ITK copy of the GATE label image
  GateImage *gateImg = gateImage->GetImage();
  FloatImageType::Pointer originLabelImage = ConvertGateToITKImage_float(gateImg);
  FloatDuplicatorType::Pointer floatDuplicatorFilter = FloatDuplicatorType::New();
  floatDuplicatorFilter->SetInputImage(originLabelImage);
  floatDuplicatorFilter->Update();
  FloatImageType::Pointer labelImage = floatDuplicatorFilter->GetOutput();
  FloatImageType::SizeType labelImageSize = labelImage->GetRequestedRegion().GetSize();
  
  // find the maximum resolution in image that will be use to limit the diffusion to the voxel size
  double resolutionMax = 0.0;
  double voxelVolume = 1.0;
  for (unsigned int i = 0; i < 3; i++)
  {
    double spacing = gateImg->GetVoxelSize()[i];
    voxelVolume = voxelVolume * spacing * mm;
    if (spacing > resolutionMax) { resolutionMax = spacing; }
  }
  resolutionMax = resolutionMax * mm;
  double sigmaMax = 1.0 * resolutionMax;

  // Create a temporary materialToLabel map in order to regroup every voxel with the same material under the same label
  std::map<float, G4Material *> labelToNewMaterial;
  std::map<G4Material *, float> materialToNewLabel;
  std::map<G4Material *, float>::iterator itNewLabel;
  gateImage->BuildLabelToG4MaterialVector(mMaterialList);
  int newLabel = 0;
  
  FloatIteratorType itImage(labelImage, labelImage->GetRequestedRegion());
  for(itImage.GoToBegin(); !itImage.IsAtEnd(); ++itImage)
  {
    // 1. get voxel label and material
    int label = itImage.Get();
    G4Material *mat = mMaterialList[label];

    // 2. check if material already exists, create map entry if not
    itNewLabel = materialToNewLabel.find(mat);
    if(itNewLabel == materialToNewLabel.end())
    {
      materialToNewLabel.insert(std::make_pair(mat, (float)newLabel));      
      labelToNewMaterial.insert(std::make_pair((float)newLabel, mat));      
      newLabel++;
    }
    
    // 3. replace old label with the new one
    itNewLabel = materialToNewLabel.find(mat);
    itImage.Set(itNewLabel->second);
    labelImage->Update();
  }
  
  // Create masks and diffusion structure for each label
  for(itNewLabel = materialToNewLabel.begin(); itNewLabel != materialToNewLabel.end(); ++itNewLabel)
  {
    newLabel = itNewLabel->second;

    // 1. find the ROI (lower and higher indices) corresponding to the label in the image (crop process)
    int lowerBound[3] = {(int)labelImageSize[0]-1,(int)labelImageSize[1]-1,(int)labelImageSize[2]-1};
    int upperBound[3] = {0,0,0};
    for(itImage.GoToBegin(); !itImage.IsAtEnd(); ++itImage)
    {
      if(itImage.Get() == newLabel)
      {
        FloatImageType::IndexType index = itImage.GetIndex();
        if(index[0] < lowerBound[0]) { lowerBound[0] = index[0]; }
        if(index[1] < lowerBound[1]) { lowerBound[1] = index[1]; }
        if(index[2] < lowerBound[2]) { lowerBound[2] = index[2]; }
        if(index[0] > upperBound[0]) { upperBound[0] = index[0]; }
        if(index[1] > upperBound[1]) { upperBound[1] = index[1]; }
        if(index[2] > upperBound[2]) { upperBound[2] = index[2]; }
      }
    }
    
    // 2. slightly increase the ROI in each direction for the diffusion process
    FloatImageType::IndexType start;
    FloatImageType::SizeType size;
    bool useROI = false;
    for(int i=0; i<3; i++)
    {
      lowerBound[i] -= 4; // 4.0 * sigma (gaussian blurring) 
      upperBound[i] += 5; // 1pix + 4.0 * sigma (gaussian bluring)
      if(lowerBound[i] > 0) { useROI = true; }
      else { lowerBound[i] = 0; }
      if(upperBound[i] < (int)labelImageSize[i]) { useROI = true; }
      else { upperBound[i] = labelImageSize[i]; }
      
      start[i] = lowerBound[i];
      size[i] = upperBound[i] - lowerBound[i];
    }

    // 3. create mask and diffusion structure
    FloatImageType::RegionType region;
    region.SetIndex(start);
    region.SetSize(size);
    
    FloatToFloatROIFilterType::Pointer roiFilter = FloatToFloatROIFilterType::New();
    roiFilter->SetInput(labelImage);
    roiFilter->SetRegionOfInterest(region);

    BinaryThresholdFilterType::Pointer binaryThresholdFilter = BinaryThresholdFilterType::New();
    binaryThresholdFilter->SetOutsideValue(float(0.0));
    binaryThresholdFilter->SetInsideValue(float(1.0));
    binaryThresholdFilter->SetInput(roiFilter->GetOutput());
    binaryThresholdFilter->SetLowerThreshold(newLabel-0.1);
    binaryThresholdFilter->SetUpperThreshold(newLabel+0.1);
    binaryThresholdFilter->Update();
    
    FloatImageType::Pointer mask = binaryThresholdFilter->GetOutput();
    mask->DisconnectPipeline();

    double diffusivity = GetPropertyFromMaterial(itNewLabel->first, "DIFFUSIVITY", mm2/s);
    DiffusionStruct newDiffStruct(diffusivity, sigmaMax, useROI, region, mask);
    mMaterialToDiffusionStruct.insert(std::make_pair(itNewLabel->first, newDiffStruct));
  }

  // Define the minimum timeStep for diffusion (highest diffusive material)
//   double diffusivityMin = 1.0e10;
  double diffusivityMax = 0.0;
  std::map<G4Material *, DiffusionStruct>::iterator itMap;  
  for(itMap = mMaterialToDiffusionStruct.begin(); itMap != mMaterialToDiffusionStruct.end(); ++itMap)
  {
//     if(itMap->second.diffusivity < diffusivityMin and itMap->second.diffusivity > 0) { diffusivityMin = itMap->second.diffusivity; }
    if(itMap->second.diffusivity > diffusivityMax) { diffusivityMax = itMap->second.diffusivity; }
  }
  mMinTimeStep = sigmaMax * sigmaMax / (2.0 * diffusivityMax);
//   double durationMin = sigmaMax * sigmaMax / (2.0 * diffusivityMax);
//   double durationMax = sigmaMax * sigmaMax / (2.0 * diffusivityMin);
//   G4cout << "diffMax = " << diffusivityMax / (mm2/s) << " durationMin = " << durationMin / s  << " | diffMin = " << diffusivityMin / (mm2/s) << " durationMax = " << durationMax / s << G4endl;
  
  // debug
  std::map<G4Material *, DiffusionStruct>::iterator itTmp;
  for(itTmp=mMaterialToDiffusionStruct.begin(); itTmp!=mMaterialToDiffusionStruct.end(); ++itTmp)
  {
    G4cout << "mat "<< itTmp->first->GetName() << " | diff = " << itTmp->second.diffusivity / (mm2/s)
           << " mm2.s-1 | period = " << itTmp->second.period / s
           << " s | timer = " << itTmp->second.timer / s << " s " << G4endl;
           
//     std::ostringstream temp;
//     temp << itTmp->first->GetName();
//     SaveITKimage(itTmp->second.mask, "output/itkMask_" + temp.str() + ".mhd");
  }

  // Construct conversion map (eV->Kelvin) with dimensions of heat map (MANDATORY)
  DoubleDuplicatorType::Pointer doubleDuplicatorFilter1 = DoubleDuplicatorType::New();
  doubleDuplicatorFilter1->SetInputImage(mITKheatMap);
  doubleDuplicatorFilter1->Update();
  mITKheatConversionMap = doubleDuplicatorFilter1->GetOutput();
  mITKheatConversionMap->DisconnectPipeline();
  for(itImage.GoToBegin(); !itImage.IsAtEnd(); ++itImage)
  {
    // 1. get voxel label and material
    DoubleImageType::IndexType index = itImage.GetIndex();
    std::map<float, G4Material *>::iterator itLabel = labelToNewMaterial.find(itImage.Get());
    G4Material *mat = itLabel->second;
    
    // 2. get tissue heat capacity ('user' value from 'macro' or 'by material' from 'Material.xml') and tissue density
    double tissueDensity = mat->GetDensity();
    double tissueHeatCapacity;
    if(mUserTissueHeatCapacity > 0.0) { tissueHeatCapacity = mUserTissueHeatCapacity; }
    else { tissueHeatCapacity = GetPropertyFromMaterial(mat, "HEATCAPACITY", joule/(kg*kelvin)); }

    // 3. calculate conversion factor (eV->Kelvin)
    double eVtoDegreeFactor = 1.0 / (tissueDensity * voxelVolume * tissueHeatCapacity);
    mITKheatConversionMap->SetPixel(index, eVtoDegreeFactor);
  }
  
  // Construct perfusionMap (OPTIONAL)
  double mMinPerfusionCoef = 1.0e10;
  double mMaxPerfusionCoef = 0.0;
  if(mIsPerfusionActivated)
  {
    // create empty perfusion map with dimensions of heat map 
    DoubleDuplicatorType::Pointer doubleDuplicatorFilter2 = DoubleDuplicatorType::New();
    doubleDuplicatorFilter2->SetInputImage(mITKheatMap);
    doubleDuplicatorFilter2->Update();
    mITKperfusionRateMap = doubleDuplicatorFilter2->GetOutput();
    
    // - security conditions (have to be set in 'macro')
    if(mUserBloodDensity<=0.0) { GateError("Error: Please, set the 'bloodDensity' in order to use the blood perfusion process."); }
    if(mUserBloodHeatCapacity<=0.0) { GateError("Error: Please, set the 'bloodHeatCapacity' in order to use the blood perfusion process."); }

    // - if perfusion image is provided by the user, read it
    DoubleImageType::Pointer tmpPerfusionRateMap = DoubleImageType::New();
    if(mIsPerfusionByImage)
    {
      DoubleReaderType::Pointer doubleReaderFilter = DoubleReaderType::New();
      doubleReaderFilter->SetFileName(mUserPerfusionImageName);
      doubleReaderFilter->Update();
      tmpPerfusionRateMap = doubleReaderFilter->GetOutput();
    }
    
    for(itImage.GoToBegin(); !itImage.IsAtEnd(); ++itImage)
    {
      // 1. get voxel label and material
      DoubleImageType::IndexType index = itImage.GetIndex();
      std::map<float, G4Material *>::iterator itLabel = labelToNewMaterial.find(itImage.Get());
      G4Material *mat = itLabel->second;

      // 2. get voxel label and material
      double perfusionRate = mUserBloodPerfusionRate;
      if(mIsPerfusionByMaterial) { perfusionRate = GetPropertyFromMaterial(mat, "PERFUSIONRATE", 1./s); }
      else if(mIsPerfusionByImage) { perfusionRate = tmpPerfusionRateMap->GetPixel(index) / s; }

      // 3. get tissue heat capacity ('user' value from 'macro' or 'by material' from 'Material.xml') and tissue density
      double tissueDensity = mat->GetDensity();
      double tissueHeatCapacity;
      if(mUserTissueHeatCapacity > 0.0) { tissueHeatCapacity = mUserTissueHeatCapacity; }
      else { tissueHeatCapacity = GetPropertyFromMaterial(mat, "HEATCAPACITY", joule/(kg*kelvin)); }
      
      // 4. final perfusionRate factor
      double perfusionCoef = perfusionRate * (mUserBloodDensity * mUserBloodHeatCapacity) / (tissueDensity * tissueHeatCapacity);
      mITKperfusionRateMap->SetPixel(index, perfusionCoef);

      // 5. find the min and max perfusionRate values for calculating perfusion time step
      if(perfusionCoef < mMinPerfusionCoef) { mMinPerfusionCoef = perfusionCoef; }
      if(perfusionCoef > mMaxPerfusionCoef) { mMaxPerfusionCoef = perfusionCoef; }
    }

    // define the global timeStep for perfusion (time for 1% heat decrease for the highest perfusionRate)
    mMinPerfusionTimeStep = -log(mPerfusionRatio) / mMaxPerfusionCoef;
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMultiMaterialThermalActor::FloatImageType::Pointer GateMultiMaterialThermalActor::ConvertGateToITKImage_float(GateImage *gateImg)
{
  ImportFilterType::Pointer gateToITKImageFilter = ImportFilterType::New();
  ImportFilterType::SizeType  size;
  double origin[3];
  double spacing[3];
  for (unsigned int i = 0; i < 3; i++)
  {
    size[i] = gateImg->GetResolution()[i];
    spacing[i] = gateImg->GetVoxelSize()[i];
    origin[i] = -gateImg->GetHalfSize()[i] + 0.5 * spacing[i];
  }

  ImportFilterType::IndexType start;
  start.Fill(0);
  ImportFilterType::RegionType region;
  region.SetIndex( start );
  region.SetSize(  size  );
 
  gateToITKImageFilter->SetRegion(region);
  gateToITKImageFilter->SetOrigin(origin);
  gateToITKImageFilter->SetSpacing(spacing);
 
  const unsigned int numberOfPixels =  size[0] * size[1] * size[2];
  const bool importImageFilterWillOwnTheBuffer = false;
  gateToITKImageFilter->SetImportPointer(&*(gateImg->begin()), numberOfPixels, importImageFilterWillOwnTheBuffer);
  gateToITKImageFilter->Update();

  FloatImageType::Pointer output = gateToITKImageFilter->GetOutput();
  return output;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateMultiMaterialThermalActor::DoubleImageType::Pointer GateMultiMaterialThermalActor::ConvertGateToITKImage_double(GateImageDouble *gateImg)
{
  typedef itk::ImportImageFilter<double, 3> DoubleImportFilterType;
  DoubleImportFilterType::Pointer importFilter = DoubleImportFilterType::New();

  DoubleImportFilterType::SizeType  size;
  double origin[3];
  double spacing[3];
  for (unsigned int i = 0; i < 3; i++)
  {
    size[i] = gateImg->GetResolution()[i];
    spacing[i] = gateImg->GetVoxelSize()[i];
    origin[i] = -gateImg->GetHalfSize()[i] + 0.5 * spacing[i];
  }

  DoubleImportFilterType::IndexType start;
  start.Fill(0);
  DoubleImportFilterType::RegionType region;
  region.SetIndex( start );
  region.SetSize(  size  );

  importFilter->SetRegion(region);
  importFilter->SetOrigin(origin);
  importFilter->SetSpacing(spacing);

  const unsigned int numberOfPixels =  size[0] * size[1] * size[2];
  const bool importImageFilterWillOwnTheBuffer = false;
  importFilter->SetImportPointer(&*(gateImg->begin()), numberOfPixels, importImageFilterWillOwnTheBuffer);
  importFilter->Update();
  DoubleImageType::Pointer output = importFilter->GetOutput();
  
  return output;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::ReadMeasurementFile(DoubleImageType::Pointer img)
{
  // Open file
  std::ifstream is;
  OpenFileInput(mMeasurementFilename, is);
  skipComment(is);

  // Use R et/or T ? Time s  | Translation mm | Rotation deg
  double timeUnit=0;
  if (!ReadColNameAndUnit(is, "Time", timeUnit)) {
    GateError("The file '" << mMeasurementFilename << "' need to begin with 'Time'\n");
  }

  // Loop line
  skipComment(is);
  while (is) {
    int label = lrint(ReadDouble(is));
    double timeStep = ReadDouble(is) * timeUnit;
    signed int lx = lrint(ReadDouble(is));
    unsigned int ux = lrint(ReadDouble(is));
    signed int ly = lrint(ReadDouble(is));
    unsigned int uy = lrint(ReadDouble(is));
    signed int lz = lrint(ReadDouble(is));
    unsigned int uz = lrint(ReadDouble(is));
    
    G4cout<<"lx,ux: [" <<lx<<","<<ux<<"] ly,uy: [" <<ly<<","<<uy<<"] lz,uz: [" <<lz<<","<<uz<<"]" << G4endl;
    
    DoubleImageType::SizeType size = img->GetRequestedRegion().GetSize();    
    if(timeStep>0 and lx>-1 and ly>-1 and lz>-1 and ux<size[0] and uy<size[1] and uz<size[2])
    {
      MeasurementStruct newMeasPoint(label, timeStep);
      for(unsigned int i=lx; i<ux+1; i++) {
        for(unsigned int j=ly; j<uy+1; j++) {
          for(unsigned int k=lz; k<uz+1; k++) {
            DoubleImageType::IndexType newIndex;
            newIndex[0] = i;
            newIndex[1] = j;
            newIndex[2] = k;
            newMeasPoint.indexList.push_back(newIndex);
          }
        }        
      }

      mMeasurementPoints.push_back(newMeasPoint);
    }
  
    skipComment(is);
  }
  
  // End
  is.close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::SaveITKimage(FloatImageType::Pointer img, G4String name)
{
  itk::ImageFileWriter<FloatImageType>::Pointer writer = itk::ImageFileWriter<FloatImageType>::New();
  writer->SetFileName(name);
  writer->SetInput(img);
  writer->Update();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActor::SaveITKimage(DoubleImageType::Pointer img, G4String name)
{
  itk::ImageFileWriter<DoubleImageType>::Pointer writer = itk::ImageFileWriter<DoubleImageType>::New();
  writer->SetFileName(name);
  writer->SetInput(img);
  writer->Update();
}
//-----------------------------------------------------------------------------

#endif // end define USE_ITK
