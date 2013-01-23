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
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

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
DD("GateHybridForcedDetectionActor BeginOfRunAction");
  GateVActor::BeginOfRunAction(r);

  // Get information on the attached 3D image
  GateVImageVolume * gate_image_volume = dynamic_cast<GateVImageVolume*>(mVolume);
  GateImage * gate_image = gate_image_volume->GetImage();
  G4ThreeVector gate_size = gate_image->GetResolution();
  G4ThreeVector gate_spacing = gate_image->GetVoxelSize();
  G4ThreeVector gate_origin = gate_image->GetOrigin();  
  InputImageType::SizeType size;
  InputImageType::PointType origin;
  InputImageType::RegionType region;
  InputImageType::SpacingType spacing;
  for(unsigned int i=0; i<3; i++) {
    size[i] = gate_size[i];
    spacing[i] = gate_spacing[i];
    origin[i] = gate_origin[i];
  }
DD(size);
DD(origin);
DD(spacing);
  region.SetSize(size);
  InputImageType::Pointer input = InputImageType::New();
  input->SetRegions(region);
  input->SetSpacing(spacing);
  input->SetOrigin(origin);
  input->Allocate();
DD("allocated");

  // Get information on the detector plane
DD(mDetectorName);
  mDetector = GateObjectStore::GetInstance()->FindVolumeCreator(mDetectorName);

  // Get information on the source
  GateSourceMgr * sm = GateSourceMgr::GetInstance();
  if (sm->GetNumberOfSources() == 0) {
    GateError("No source set. Abort.");
  }
  if (sm->GetNumberOfSources() != 1) {
    GateWarning("Several sources found, we consider the first one.");
  }
  mSource = sm->GetSource(0);
  
  // Create list of mu according to E and materials
  G4String st = mSource->GetEneDist()->GetEnergyDisType();
  if (st == "Mono") { // Mono
    mEnergyList.push_back(mSource->GetEneDist()->GetMonoEnergy());
    std::cout << G4BestUnit(mEnergyList[0], "Energy") << std::endl;
  }
  else if (st == "User") { // histo
    G4PhysicsOrderedFreeVector h = mSource->GetEneDist()->GetUserDefinedEnergyHisto ();
    for(unsigned int i=0; i<h.GetVectorLength(); i++) {
      double E = h.Energy(i);
      mEnergyList.push_back(E);
 //     std::cout << G4BestUnit(E, "Energy") << " value = " << h.Value(E) << std::endl;
    }
  }
  else {
    GateError("Error, source type is not Mono or User. Abort.");
  }

  // Create geometry and param of output image
  PointType primarySourcePosition, detectorPosition;
  VectorType detectorRowVector, detectorColVector;
  ComputeGeometryInfoInImageCoordinateSystem(gate_image_volume,
                                             mDetector,
                                             mSource,
                                             primarySourcePosition,
                                             detectorPosition,
                                             detectorRowVector,
                                             detectorColVector);
  GeometryType::Pointer geometry = GeometryType::New();
DD(primarySourcePosition)
DD(detectorPosition)
DD(detectorRowVector)
DD(detectorColVector)
  geometry->AddReg23Projection(primarySourcePosition,
                               detectorPosition,
                               detectorRowVector,
                               detectorColVector);
DD("done")
// DEBUG write geometry
rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer writer =
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
writer->SetObject(geometry);
writer->SetFilename("bidon.xml");
writer->WriteFile();
return;
  OutputImageType::Pointer output = CreateGeometry(mDetector, mSource, geometry);

  // loop on Energy to create DRR
  for(unsigned int i=0; i<mEnergyList.size(); i++) {
    double E = mEnergyList[i];

    // Create conversion label to mu
    std::vector<double> label2mu;
    CreateLabelToMuConversion(E, gate_image_volume, label2mu);

    // create mu image
    CreateMuImage(label2mu, gate_image, input);
    
    // Debug: write mu image
//    typedef itk::ImageFileWriter<InputImageType> WriterTypeIn;
//    typename WriterTypeIn::Pointer writerin = WriterTypeIn::New();
//    std::string name = "output/mu-"+DoubletoString(E)+".mhd";
//    writerin->SetFileName(name);
//    writerin->SetInput(input);
//    writerin->Update();

    // Generate drr
    output = GenerateDRR(input, output, geometry);

    // (merge) TODO
    

    // Debug: write DRR
//    typedef itk::ImageFileWriter<OutputImageType> WriterType;
//    WriterType::Pointer writer = WriterType::New();
//    name = "output/drr-"+DoubletoString(E)+".mhd";
//    writer->SetFileName(name);
//    writer->SetInput(output);
//    writer->Update();
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
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActor::ResetData() 
{
  // GateVActor::ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateHybridForcedDetectionActor::OutputImageType::Pointer 
GateHybridForcedDetectionActor::CreateGeometry(GateVVolume * detector,
                                               GateVSource * src, 
                                               GeometryType * geometry)
{
  DD("CreateGeometry");
  
  // projection input image
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  //  projInput = ConstantImageSourceType::New();

  OutputImageType::SizeType size;
  size[2] = 1;
  size[0] = GetDetectorResolution()[0];
  size[1] = GetDetectorResolution()[1];
  DD(size);

  OutputImageType::SpacingType spacing;
  spacing[2] = 1.0;
  spacing[0] = detector->GetHalfDimension(0)*2.0/size[0]*mm; //FIXME (in mm ?)
  spacing[1] = detector->GetHalfDimension(1)*2.0/size[1]*mm;
  DD(spacing);

  OutputImageType::PointType origin;
  GateVVolume * v = detector;
  G4ThreeVector du(1,0,0);
  G4ThreeVector dv(0,1,0);
  DD(du);
  DD(dv);
  G4ThreeVector tdu=du;
  G4ThreeVector tdv=dv;
  G4RotationMatrix rotation;
  G4ThreeVector translation;
  while (v->GetLogicalVolumeName() != "world_log") {
    G4VPhysicalVolume * phys = v->GetPhysicalVolume();
    const G4RotationMatrix* ro = phys->GetObjectRotation();
    G4ThreeVector to = phys->GetObjectTranslation();
    rotation = (*ro)*rotation;
    translation = translation + to;
    v = v->GetParentVolume();
  }

  DD(rotation);
  DD(translation);
  tdu = rotation * tdu + translation;
  DD(tdu);
  tdv = rotation * tdv + translation;
  DD(tdv);
  origin[0] = translation[0];
  origin[1] = translation[1];
  origin[2] = translation[2];
  DD(origin);

  // http://hypernews.slac.stanford.edu/HyperNews/geant4/get/geometry/17/1.html
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();  
  // Set values
  projInput->SetOrigin(origin);
  projInput->SetSpacing(spacing);
  projInput->SetSize(size);
  projInput->SetConstant(0.0);
  projInput->Update();

  // source
  DD(src->GetType());
  DD(src->GetName());
  G4ThreeVector c = src->GetPosDist()->GetCentreCoords();
  DD(c);

  // geometry (in mm)
  DD(mm);
  double sid = sqrt(norm(c))*mm; //1000;
  double sdd = sqrt(norm(c-translation))*mm; //1536;
  DD(sid);
  DD(sdd);
  double gantryAngle = 0.0;
  double sx = (size[0] * spacing[0])/2.0;
  double sy = (size[1] * spacing[1])/2.0;
  DD(sx);
  DD(sy);
  geometry->AddProjection(sid, sdd, gantryAngle, -sx, -sy);

  // DEBUG write geometry
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer writer = 
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  writer->SetObject(geometry);
  writer->SetFilename("bidon.xml");
  writer->WriteFile();

  // Return output image
  return projInput->GetOutput();
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
  // To understand the use of GetRotation and GetTranslation check 4.1.4.1 at
  // http://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html

  // Detector to world
  GateVVolume * v = detector;
  G4VPhysicalVolume * phys = v->GetPhysicalVolume();
  G4AffineTransform detectorToWorld(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    detectorToWorld *= detectorToWorld;
  }

  // CT to world
  v = ct;
  phys = v->GetPhysicalVolume();
  G4AffineTransform ctToWorld(phys->GetRotation(), phys->GetTranslation());
  while (v->GetLogicalVolumeName() != "world_log") {
    v = v->GetParentVolume();
    phys = v->GetPhysicalVolume();
    G4AffineTransform x(phys->GetRotation(), phys->GetTranslation());
    ctToWorld *= x;
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
    sourceToWorld *= x;
  }

  // Detector parameters
  G4AffineTransform detectorToCT(ctToWorld.Inverse() * detectorToWorld);

  // TODO: get rot1 and rot2 in beam
  G4ThreeVector du = detectorToCT.TransformAxis(G4ThreeVector(1,0,0));
  G4ThreeVector dv = detectorToCT.TransformAxis(G4ThreeVector(0,1,0));
  G4ThreeVector dp = detectorToCT.TransformPoint(G4ThreeVector(0,0,0));

  // Source (assumed focus)
  G4ThreeVector s = src->GetAngDist()->GetFocusPointCopy();
  G4AffineTransform sourceToCT(ctToWorld.Inverse() * sourceToWorld);
  s = sourceToCT.TransformPoint(s);

  for(int i=0; i<3; i++) {
      detectorRowVector[i] = du[i];
      detectorColVector[i] = dv[i];
      detectorPosition[i] = dp[i];
      primarySourcePosition[i] = s[i];
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActor::CreateMuImage(const std::vector<double> & label2mu,
                                                   const GateImage * gate_image, 
                                                   InputImageType * input)
{
  typedef itk::ImageRegionIterator<InputImageType> IteratorType;
  IteratorType pi(input,input->GetLargestPossibleRegion());
  pi.GoToBegin();
  GateImage::const_iterator data = gate_image->begin();
  while (!pi.IsAtEnd()) {
    double e = label2mu[(int)(*data)];
    pi.Set(e); 
    ++pi;
    ++data;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateHybridForcedDetectionActor::OutputImageType::Pointer 
GateHybridForcedDetectionActor::GenerateDRR(const InputImageType * input, 
                                            const OutputImageType * projInput, 
                                            GeometryType * geometry)
{
  typedef rtk::JosephForwardProjectionImageFilter<InputImageType, OutputImageType> JFPType;
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
                                                               std::vector<double> & label2mu)
{
  G4EmCalculator * emcalc = new G4EmCalculator;
  std::vector<G4Material*> m;
  gate_image_volume->BuildLabelToG4MaterialVector(m);
  G4String part = "gamma";
  G4String proc_compton = "Compton";
  G4String proc_rayleigh= "Rayleigh";
  label2mu.clear();
  label2mu.resize(m.size());
  for(unsigned int i=0; i<m.size(); i++) {
    G4Material * mat = m[i];
    double d = mat->GetDensity();
    //SR: why not looping over the list of processes like Edward does?
    double xs_c = emcalc->ComputeCrossSectionPerVolume(E, part, proc_compton, mat->GetName());
    double xs_r = emcalc->ComputeCrossSectionPerVolume(E, part, proc_rayleigh, mat->GetName());
    double mu = (xs_c+xs_r)/d;
    label2mu[i] = mu;
  }
}
//-----------------------------------------------------------------------------


#endif

