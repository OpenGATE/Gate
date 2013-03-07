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

// G4
#include "G4Event.hh"
#include "G4MaterialTable.hh"
#include "G4ParticleTable.hh"
#include "G4EmCalculator.hh"
#include "G4TransportationManager.hh"

// rtk
#include <rtkThreeDCircularProjectionGeometryXMLFile.h>
#include "rtkConstantImageSource.h"

// itk
#include <itkImportImageFilter.h>
#include <itkTimeProbe.h>
#include <itkDivideImageFilter.h>
#include <itkLogImageFilter.h>
#include <itkMultiplyImageFilter.h>

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
  //   EnableBeginOfEventAction(true);
  //   EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Handling of the interpolation weight in primary: store the weights and
// the material indices in vectors and return nada. The integral is computed in the
// ProjectedValueAccumulation since one has to repeat the same ray cast for each
// and every energy of the primary.
class GateHybridForcedDetectionActor::PrimaryInterpolationWeightMultiplication
{
public:
  PrimaryInterpolationWeightMultiplication() {};
  ~PrimaryInterpolationWeightMultiplication() {};
  bool operator!=( const PrimaryInterpolationWeightMultiplication & ) const {
    return false;
  }
  bool operator==(const PrimaryInterpolationWeightMultiplication & other) const {
    return !( *this != other );
  }

  inline double operator()( const rtk::ThreadIdType threadId,
                            const double stepLengthInVoxel,
                            const double weight,
                            const float *p,
                            const unsigned int i) {
    m_InterpolationWeights[threadId][(unsigned int)(p[i])] += stepLengthInVoxel * weight;
    return 0.;
  }

  std::vector<double>* GetInterpolationWeights() { return m_InterpolationWeights; }

private:
  std::vector<double> m_InterpolationWeights[ITK_MAX_THREADS];
};
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Most of the computation for the primary is done in this functor. After a ray
// has been cast, it loops over the energies, computes the ray line integral for
// that energy and takes the exponential of the opposite and add.
class GateHybridForcedDetectionActor::PrimaryValueAccumulation
{
public:
  typedef itk::Vector<double, 3> VectorType;

  PrimaryValueAccumulation() {};
  ~PrimaryValueAccumulation() {};
  bool operator!=( const PrimaryValueAccumulation & ) const
  {
    return false;
  }
  bool operator==(const PrimaryValueAccumulation & other) const
  {
    return !( *this != other );
  }

  inline double operator()( const rtk::ThreadIdType threadId,
                            double input,
                            const double &itkNotUsed(rayCastValue),
                            const VectorType &stepInMM,
                            const VectorType &source,
                            const VectorType &sourceToPixel,
                            const VectorType &nearestPoint,
                            const VectorType &farthestPoint) const
  {
    double *p = m_MaterialMu->GetPixelContainer()->GetBufferPointer();

    // Multiply interpolation weights by step norm in MM to convert voxel
    // intersection length to MM.
    const double stepInMMNorm = stepInMM.GetNorm();
    for(unsigned int j=0; j<m_InterpolationWeights[threadId].size()-1; j++)
      m_InterpolationWeights[threadId][j] *= stepInMMNorm;

    // The last material is the world material. One must fill the weight with
    // the length from source to nearest point and farthest point to pixel
    // point.
    VectorType worldVector = sourceToPixel - nearestPoint + farthestPoint;
    for(int i=0; i<3; i++)
      worldVector[i] *= m_Spacing[i];
    m_InterpolationWeights[threadId].back() = worldVector.GetNorm();

    // Loops over energy, multiply weights by mu, accumulate using Beer Lambert
    for(unsigned int i=0; i<m_EnergyWeightList->size(); i++) {
      double rayIntegral = 0.;
      for(unsigned int j=0; j<m_InterpolationWeights[threadId].size(); j++){
        rayIntegral += m_InterpolationWeights[threadId][j] * *p++;
      }
      input += vcl_exp(-rayIntegral) * (*m_EnergyWeightList)[i];
    }

    // Reset weights for next ray in thread.
    std::fill(m_InterpolationWeights[threadId].begin(), m_InterpolationWeights[threadId].end(), 0.);
    return input;
  }

  void SetSpacing(const VectorType &_arg){ m_Spacing = _arg; }
  void SetInterpolationWeights(std::vector<double> *_arg){ m_InterpolationWeights = _arg; }
  void SetEnergyWeightList(std::vector<double> *_arg) { m_EnergyWeightList = _arg; }
  void SetMaterialMu(itk::Image<double, 2>::Pointer _arg) { m_MaterialMu = _arg; }
  void Init(unsigned int nthreads) {
    for(unsigned int i=0; i<nthreads; i++) {
      m_InterpolationWeights[i].resize(m_MaterialMu->GetLargestPossibleRegion().GetSize()[0]);
      std::fill(m_InterpolationWeights[i].begin(), m_InterpolationWeights[i].end(), 0.);
    }
  }

private:
  VectorType                      m_Spacing;
  std::vector<double>            *m_InterpolationWeights;
  std::vector<double>            *m_EnergyWeightList;
  itk::Image<double, 2>::Pointer  m_MaterialMu;
  unsigned int nmat;
};
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin of Run
void GateHybridForcedDetectionActor::BeginOfRunAction(const G4Run*r)
{
  GateVActor::BeginOfRunAction(r);

  // Get information on the source
  GateSourceMgr * sm = GateSourceMgr::GetInstance();
  if (sm->GetNumberOfSources() == 0) {
    GateError("No source set. Abort.");
  }
  if (sm->GetNumberOfSources() != 1) {
    GateWarning("Several sources found, we consider the first one.");
  }
  mSource = sm->GetSource(0);

  // Create list of energies
  std::vector<double> energyList;
  std::vector<double> energyWeightList;
  energyList.clear();
  G4String st = mSource->GetEneDist()->GetEnergyDisType();
  if (st == "Mono") {
    energyList.push_back(mSource->GetEneDist()->GetMonoEnergy());
    energyWeightList.push_back(1.);
  }
  else if (st == "User") { // histo
    G4PhysicsOrderedFreeVector h = mSource->GetEneDist()->GetUserDefinedEnergyHisto ();
    double weightSum = 0.;
    for(unsigned int i=0; i<h.GetVectorLength(); i++) {
      double E = h.Energy(i);
      energyList.push_back(E);
      energyWeightList.push_back(h.Value(E));
      weightSum += energyWeightList.back();
    }
    for(unsigned int i=0; i<h.GetVectorLength(); i++)
      energyWeightList[i] /= weightSum;
  }
  else
    GateError("Error, source type is not Mono or User. Abort.");

  // Conversion of CT to ITK and to int values
  // SR: is this a safe cast? Shouldn't we add 0.5? To check with DS
  GateVImageVolume * gate_image_volume = dynamic_cast<GateVImageVolume*>(mVolume);
  GateImage * gate_image = gate_image_volume->GetImage();
  InputImageType::Pointer input = ConvertGateImageToITKImage(gate_image);

  // Create projection image
  mPrimaryImage = CreateVoidProjectionImage();
  InputImageType::Pointer mPrimaryImage2 = CreateVoidProjectionImage();

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

  // Create conversion label to mu
  itk::TimeProbe muProbe;
  muProbe.Start();
  CreateLabelToMuConversion(energyList, gate_image_volume);
  muProbe.Stop();
  G4cout << "Computation of the mu lookup table took "
         << muProbe.GetTotal()
         << ' '
         << muProbe.GetUnit()
         << G4endl;

  // Create primary projector and compute primary
  itk::TimeProbe primaryProbe;
  primaryProbe.Start();
  typedef rtk::JosephForwardProjectionImageFilter< InputImageType,
                                                   InputImageType,
                                                   PrimaryInterpolationWeightMultiplication,
                                                   PrimaryValueAccumulation> JFPType;
    JFPType::Pointer jfp = JFPType::New();
    jfp->InPlaceOn();
    jfp->SetInput(mPrimaryImage);
    jfp->SetInput(1, input );
    jfp->SetGeometry( oneProjGeometry.GetPointer() );
    jfp->GetProjectedValueAccumulation().SetSpacing( input->GetSpacing() );
    jfp->GetProjectedValueAccumulation().SetInterpolationWeights( jfp->GetInterpolationWeightMultiplication().GetInterpolationWeights() );
    jfp->GetProjectedValueAccumulation().SetEnergyWeightList( &energyWeightList );
    jfp->GetProjectedValueAccumulation().SetMaterialMu( mMaterialMu );
    jfp->GetProjectedValueAccumulation().Init( jfp->GetNumberOfThreads() );
    TRY_AND_EXIT_ON_ITK_EXCEPTION(jfp->Update());
    mPrimaryImage = jfp->GetOutput();

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
    flatFieldSource->SetConstant( -1000. );
    // Joseph Forward projector
    JFPType::Pointer jfpFlat = JFPType::New();
    jfpFlat->InPlaceOn();
    jfpFlat->SetInput(mPrimaryImage2);
    jfpFlat->SetInput(1, flatFieldSource->GetOutput() );
    jfpFlat->SetGeometry( oneProjGeometry.GetPointer() );
    jfpFlat->GetProjectedValueAccumulation().SetSpacing( flatFieldSource->GetSpacing() );
    jfpFlat->GetProjectedValueAccumulation().SetInterpolationWeights( jfpFlat->GetInterpolationWeightMultiplication().GetInterpolationWeights() );
    jfpFlat->GetProjectedValueAccumulation().SetEnergyWeightList( &energyWeightList );
    jfpFlat->GetProjectedValueAccumulation().SetMaterialMu( mMaterialMu );
    jfpFlat->GetProjectedValueAccumulation().Init( jfpFlat->GetNumberOfThreads() );
    TRY_AND_EXIT_ON_ITK_EXCEPTION(jfpFlat->Update());
    mFlatFieldImage = jfpFlat->GetOutput();
  }
  primaryProbe.Stop();
  G4cout << "Computation of the primary took "
         << primaryProbe.GetTotal()
         << ' '
         << primaryProbe.GetUnit()
         << G4endl;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback Begin Event
/*void GateHybridForcedDetectionActor::BeginOfEventAction(const G4Event*e)
{
}*/
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
  //DD("GateHybridForcedDetectionActor UserSteppingAction");
  GateVActor::UserSteppingAction(v, step);

  /* Get interaction point from step
     Retrieve : 
     - type of limiting process (Compton Rayleigh Fluorescence)
     - no Fluo yet, wait for bug fix in next G4 release (4.6 ?)
     - coordinate of interaction, convert if needed into world coordinate system
     - Get Energy
     - -> generate adequate forward projections towards detector
  */

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

  // Write the image of primary radiation
  itk::ImageFileWriter<InputImageType>::Pointer imgWriter;
  imgWriter = itk::ImageFileWriter<InputImageType>::New();
  char filename[1024];
  G4int rID = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();
  sprintf(filename, mPrimaryFilename.c_str(), rID);
  imgWriter->SetFileName(filename);
  imgWriter->SetInput(mPrimaryImage);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(imgWriter->Update());

  if(mMaterialMuFilename != "") {
    typedef itk::ImageFileWriter< itk::Image<double, 2> > TwoDWriter;
    TwoDWriter::Pointer w = TwoDWriter::New();
    w->SetInput(mMaterialMu);
    w->SetFileName(mMaterialMuFilename);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(w->Update());
  }

  if(mAttenuationFilename != "") {
    //Writing attenuation image (-log(primaryImage/flatFieldImage))
    itk::DivideImageFilter<InputImageType,InputImageType,InputImageType>::Pointer      divFilter;
    itk::LogImageFilter<InputImageType, InputImageType>::Pointer                       logFilter;
    itk::MultiplyImageFilter< InputImageType, InputImageType, InputImageType>::Pointer mulFilter;
    itk::ImageFileWriter<InputImageType>::Pointer                                      imgAttWriter;

    divFilter  = itk::DivideImageFilter<InputImageType,InputImageType,InputImageType>::New();
    logFilter  = itk::LogImageFilter<InputImageType,InputImageType>::New();
    mulFilter  = itk::MultiplyImageFilter<InputImageType,InputImageType>::New();
    imgAttWriter = itk::ImageFileWriter<InputImageType>::New();

    divFilter->SetInput1(mPrimaryImage);
    divFilter->SetInput2(mFlatFieldImage);
    logFilter->SetInput(divFilter->GetOutput());
    mulFilter->SetInput(logFilter->GetOutput());
    mulFilter->SetConstant(-1.0);
    mulFilter->InPlaceOn();

    //Writing flat field image SR: Do we set an option for this image in main.mac?
    imgAttWriter->SetFileName("output/flatFieldImage.mha");
    imgAttWriter->SetInput(mFlatFieldImage);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgAttWriter->Update());

    //Writing attenuation image
    char filename[1024];
    G4int rID = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();
    sprintf(filename, mAttenuationFilename.c_str(), rID);
    imgAttWriter->SetFileName(filename);
    imgAttWriter->SetInput(mulFilter->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(imgAttWriter->Update());
  }
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
    detectorToWorld = x * detectorToWorld;
  }

  // CT to world
  v = ct;
  phys = v->GetPhysicalVolume();
  G4AffineTransform ctToWorld(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    ctToWorld = x * ctToWorld;
  }

  // Source to world
  G4String volname = src->GetRelativePlacementVolume();
  v = GateObjectStore::GetInstance()->FindVolumeCreator(volname);
  phys = v->GetPhysicalVolume();
  G4AffineTransform sourceToWorld(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    sourceToWorld = x * sourceToWorld;
  }

  // Detector parameters
  G4AffineTransform detectorToCT(detectorToWorld * ctToWorld.Inverse());

  // TODO: check where to get the two directions of the detector.
  // Probably the dimension that has lowest size in one of the three directions. 
  G4ThreeVector du = detectorToCT.TransformAxis(G4ThreeVector(1,0,0));
  G4ThreeVector dv = detectorToCT.TransformAxis(G4ThreeVector(0,1,0));
  G4ThreeVector dp = detectorToCT.TransformPoint(G4ThreeVector(0,0,0));

  // Source (assumed point or focus)
  G4ThreeVector s = src->GetPosDist()->GetCentreCoords(); // point 
  if(src->GetPosDist()->GetPosDisType()!=G4String("Point")) // Focus
    s = src->GetAngDist()->GetFocusPointCopy();
  G4AffineTransform sourceToCT( sourceToWorld * ctToWorld.Inverse());
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
void GateHybridForcedDetectionActor::CreateLabelToMuConversion(const std::vector<double> &Elist,
                                                               GateVImageVolume * gate_image_volume)
{
  // Get image materials + world
  G4EmCalculator * emcalc = new G4EmCalculator;
  std::vector<G4Material*> m;
  gate_image_volume->BuildLabelToG4MaterialVector(m);
  GateVVolume *v = gate_image_volume;
  while (v->GetLogicalVolumeName() != "world_log")
    v = v->GetParentVolume();
  m.push_back(const_cast<G4Material*>(v->GetMaterial()));

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

  itk::Image<double, 2>::RegionType region;
  region.SetSize(0, m.size());
  region.SetSize(1, Elist.size());
  mMaterialMu = itk::Image<double, 2>::New();
  mMaterialMu->SetRegions(region);
  mMaterialMu->Allocate();
  itk::ImageRegionIterator< itk::Image<double, 2> > it(mMaterialMu, region);
  for(unsigned int e=0; e<Elist.size(); e++)
    for(unsigned int i=0; i<m.size(); i++) {
      G4Material * mat = m[i];
      //double d = mat->GetDensity(); // not needed
      double mu = 0;
      for (unsigned int j = 0; j < processNameVector.size(); j++) {
        // Note: the G4EmCalculator retrive the correct G4VProcess
        // (standard, Penelope, Livermore) from the processName.
        double xs =
            emcalc->ComputeCrossSectionPerVolume(Elist[e], "gamma", processNameVector[j], mat->GetName());
        // In (length unit)^{-1} according to
        // http://www.lcsim.org/software/geant4/doxygen/html/classG4EmCalculator.html#a870d5fffaca35f6e2946da432034bd4c
        mu += xs;
      }
      it.Set(mu);
      ++it;
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHybridForcedDetectionActor::InputImageType::Pointer
GateHybridForcedDetectionActor::ConvertGateImageToITKImage(GateImage * gateImg)
{
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

  return import->GetOutput();
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

  return source->GetOutput();
}
//-----------------------------------------------------------------------------

#endif
