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
#include <rtkLookupTableImageFilter.h>

// itk
#include <itkImportImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkExpImageFilter.h>
#include <itkLogImageFilter.h>

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
  else {
    GateError("Error, source type is not Mono or User. Abort.");
  }

  // Conversion of CT to ITK and to int values
  // SR: is this a safe cast? Shouldn't we add 0.5? To check with DS
  GateVImageVolume * gate_image_volume = dynamic_cast<GateVImageVolume*>(mVolume);
  GateImage * gate_image = gate_image_volume->GetImage();
  InputImageType::Pointer input = ConvertGateImageToITKImage(gate_image);
  itk::CastImageFilter<InputImageType, IntegerImageType>::Pointer cast;
  cast = itk::CastImageFilter<InputImageType, IntegerImageType>::New();
  cast->SetInput(input);
  cast->Update();

  // Create projection image
  mPrimaryImage = CreateVoidProjectionImage();

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

  // loop on Energy to create DRR
  for(unsigned int i=0; i<energyList.size(); i++) {
    double E = energyList[i];

    // Create conversion label to mu
    itk::Image<double, 1>::Pointer label2mu;
    CreateLabelToMuConversion(E, gate_image_volume, label2mu);

    // create mu image
    rtk::LookupTableImageFilter<IntegerImageType, DoubleImageType>::Pointer lutFilter;
    lutFilter = rtk::LookupTableImageFilter<IntegerImageType, DoubleImageType>::New();
    lutFilter->SetLookupTable(label2mu);
    lutFilter->SetInput(cast->GetOutput());
    lutFilter->Update();

    // Generate drr
    DoubleImageType::Pointer drr = GenerateDRR(lutFilter->GetOutput(), mPrimaryImage, oneProjGeometry);
//// Debug: write DRR
//typedef itk::ImageFileWriter<DoubleImageType> WriterType;
//WriterType::Pointer writer = WriterType::New();
//std::ostringstream os;
//os << "output/drr-" << E/CLHEP::keV << "-";
//os.fill('0');
//os.width(4);
//os << r->GetRunID() << ".mha";
//writer->SetFileName(os.str());
//writer->SetInput(drr);
//writer->Update();

    // Multiply by -1
    itk::MultiplyImageFilter<DoubleImageType, DoubleImageType>::Pointer opp;
    opp = itk::MultiplyImageFilter<DoubleImageType, DoubleImageType>::New();
    opp->SetInput(drr);
    opp->SetConstant(-1.);
    opp->Update();

    // Take exponential
    itk::ExpImageFilter<DoubleImageType, DoubleImageType>::Pointer exp;
    exp = itk::ExpImageFilter<DoubleImageType, DoubleImageType>::New();
    exp->SetInput(opp->GetOutput());
    exp->Update();

    // Multiply by energy weight
    itk::MultiplyImageFilter<DoubleImageType, DoubleImageType>::Pointer multiply;
    multiply = itk::MultiplyImageFilter<DoubleImageType, DoubleImageType>::New();
    multiply->SetInput(exp->GetOutput());
    multiply->SetConstant(energyWeightList[i]);
    multiply->Update();

    // Add to current image
    itk::AddImageFilter<DoubleImageType, DoubleImageType>::Pointer add;
    add = itk::AddImageFilter<DoubleImageType, DoubleImageType>::New();
    add->SetInput1(mPrimaryImage);
    add->SetInput2(multiply->GetOutput());
    add->Update();
    mPrimaryImage = add->GetOutput();

//// Debug: write mu image
//if(!r->GetRunID()) {
//typedef itk::ImageFileWriter<DoubleImageType> WriterTypeIn;
//WriterTypeIn::Pointer writerin = WriterTypeIn::New();
//std::string name = "output/mu-"+DoubletoString(E/CLHEP::keV)+".mha";
//writerin->SetFileName(name);
//writerin->SetInput(lutFilter->GetOutput());
//writerin->Update();
//}

  }

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

  // Convert primary to line integral
  itk::LogImageFilter<DoubleImageType, DoubleImageType>::Pointer log;
  log = itk::LogImageFilter<DoubleImageType, DoubleImageType>::New();
  log->SetInput(mPrimaryImage);
  log->Update();
  itk::MultiplyImageFilter<DoubleImageType, DoubleImageType>::Pointer mult;
  mult = itk::MultiplyImageFilter<DoubleImageType, DoubleImageType>::New();
  mult->SetInput(log->GetOutput());
  mult->SetConstant(-1.);
  mult->Update();

  // Write the image of primary radiation
  itk::ImageFileWriter<DoubleImageType>::Pointer imgWriter;
  imgWriter = itk::ImageFileWriter<DoubleImageType>::New();
  char filename[1024];
  G4int rID = G4RunManager::GetRunManager()->GetCurrentRun()->GetRunID();
  sprintf(filename, mPrimaryFilename.c_str(), rID);
  imgWriter->SetFileName(filename);
  imgWriter->SetInput(mult->GetOutput());
  imgWriter->Update();
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
GateHybridForcedDetectionActor::DoubleImageType::Pointer
GateHybridForcedDetectionActor::GenerateDRR(const DoubleImageType * input,
                                            const DoubleImageType * projInput,
                                            GeometryType * geometry)
{
  typedef rtk::JosephForwardProjectionImageFilter<DoubleImageType, DoubleImageType> JFPType;
  JFPType::Pointer jfp = JFPType::New();
  jfp->InPlaceOff();
  jfp->SetInput(projInput);
  jfp->SetInput(1, input);
  jfp->SetGeometry(geometry);
  jfp->Update();
  return jfp->GetOutput();  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActor::CreateLabelToMuConversion(const double E, 
                                                               GateVImageVolume * gate_image_volume,
                                                               itk::Image<double, 1>::Pointer & label2mu)
{
  G4EmCalculator * emcalc = new G4EmCalculator;
  std::vector<G4Material*> m;
  gate_image_volume->BuildLabelToG4MaterialVector(m);
  G4String part = "gamma";
  G4String proc_compton = "Compton";
  G4String proc_rayleigh= "Rayleigh"; // FIXME retrieve user process

  itk::Image<double, 1>::RegionType region;
  region.SetSize(0, m.size());
  label2mu = itk::Image<double, 1>::New();
  label2mu->SetRegions(region);
  label2mu->Allocate();
  itk::ImageRegionIterator< itk::Image<double, 1> > it(label2mu, region);
  for(unsigned int i=0; i<m.size(); i++) {
    G4Material * mat = m[i];
    double d = mat->GetDensity();
    //SR: why not looping over the list of processes like Edward does?
    double xs_c = emcalc->ComputeCrossSectionPerVolume(E, part, proc_compton, mat->GetName());
    double xs_r = emcalc->ComputeCrossSectionPerVolume(E, part, proc_rayleigh, mat->GetName());
    // In (length unit)^{-1} according to
    // http://www.lcsim.org/software/geant4/doxygen/html/classG4EmCalculator.html#a870d5fffaca35f6e2946da432034bd4c
    double mu = (xs_c+xs_r);
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
  import->Update();

  return import->GetOutput();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateHybridForcedDetectionActor::DoubleImageType::Pointer
GateHybridForcedDetectionActor::CreateVoidProjectionImage()
{
  mDetector = GateObjectStore::GetInstance()->FindVolumeCreator(mDetectorName);

  DoubleImageType::SizeType size;
  size[0] = GetDetectorResolution()[0];
  size[1] = GetDetectorResolution()[1];
  size[2] = 1;

  DoubleImageType::SpacingType spacing;
  spacing[0] = mDetector->GetHalfDimension(0)*2.0/size[0];
  spacing[1] = mDetector->GetHalfDimension(1)*2.0/size[1];
  spacing[2] = 1.0;

  DoubleImageType::PointType origin;
  origin[0] = -mDetector->GetHalfDimension(0)+0.5*spacing[0];
  origin[1] = -mDetector->GetHalfDimension(1)+0.5*spacing[1];
  origin[2] = 0.0;

  rtk::ConstantImageSource<DoubleImageType>::Pointer source;
  source = rtk::ConstantImageSource<DoubleImageType>::New();
  source->SetSpacing(spacing);
  source->SetOrigin(origin);
  source->SetSize(size);
  source->Update();

  return source->GetOutput();
}
//-----------------------------------------------------------------------------

#endif

