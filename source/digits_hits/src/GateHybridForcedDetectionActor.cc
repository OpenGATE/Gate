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
#include "GateVImageVolume.hh"
#include "GateMaterialMuHandler.hh"

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

  // Get information on the detector plane and source
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
  DD(mSource->GetName());
  
  
  // Create list of mu according to E and materials
  G4String st = mSource->GetEneDist()->GetEnergyDisType();
  DD(st);
  if (st == "Mono") { // Mono
    mEnergyList.push_back(mSource->GetEneDist()->GetMonoEnergy());
    std::cout << G4BestUnit(mEnergyList[0], "Energy") << std::endl;
  }
  else if (st == "User") { // histo
    G4PhysicsOrderedFreeVector h = mSource->GetEneDist()->GetUserDefinedEnergyHisto ();
    for(uint i=0; i<h.GetVectorLength(); i++) {
      mEnergyList.push_back(h.Energy(i));
      std::cout << G4BestUnit(mEnergyList[i], "Energy") << std::endl;
    }
  }
  else {
    GateError("Error, source type is not Mono or User. Abort.");
  }



  // Create geometry and param of output image 
  InputImageType::Pointer input = InputImageType::New();
  GeometryType::Pointer geometry = GeometryType::New(); 
  OutputImageType::Pointer output = CreateGeometry(mDetector, mSource, geometry);

  // loop on Energy to create DRR
  DD(mEnergyList.size());
  for(uint i=0; i<mEnergyList.size(); i++) {
    DD(i);
    double E = mEnergyList[i];
    std::cout << G4BestUnit(E, "Energy");

    // Create conversion label to mu
    std::vector<double> label2mu;
    CreateLabelToMuConversion(E, label2mu);

    // create mu image
    CreateMuImage(label2mu, gate_image, input);
    
    // launch drr
    GenerateDRR(input, output, geometry, output);

    // (merge) TODO

    // Debug: write DRR
    typedef itk::ImageFileWriter<OutputImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    std::string name = "output/drr-"+DoubletoString(E)+".mhd";
    writer->SetFileName(name);
    writer->SetInput(output);
    writer->Update();
  }


  DD("OLD");

  /*
  // FIXME  - test RTK here
  typename InputImageType::SizeType size;
  typename InputImageType::PointType origin;
  typename InputImageType::RegionType region;
  typename InputImageType::SpacingType spacing;
  for(uint i=0; i<3; i++) {
    size[i] = gate_size[i];
    spacing[i] = gate_spacing[i];
    origin[i] = gate_origin[i];
  }
  DD(size);
  DD(origin);
  DD(spacing);
  region.SetSize(size);
  input->SetRegions(region);
  input->SetSpacing(spacing);
  input->SetOrigin(origin);
  input->Allocate();
  DD("allocated");
  
  // copy values
  G4EmCalculator * emcalc = new G4EmCalculator;
  std::vector<G4Material*> m;
  gate_image_volume->BuildLabelToG4MaterialVector(m);
  double E = 60*keV;
  G4String part = "gamma";
  G4String proc_compton = "Compton";
  G4String proc_rayleigh= "Rayleigh";
  DD(E);
  DD(m.size());
  //  GateMaterialMuHandler * muHandler = new GateMaterialMuHandler(m.size());
  std::vector<double> ed;
  for(uint i=0; i<m.size(); i++) {
    // DD(i);
    G4Material * mat = m[i];
    // DD(mat->GetName());
    //muHandler->AddMaterial(mat);
    //    double d = mat->GetDensity();
    double d = mat->GetDensity();
    //DD(d/(g/cm3));
    double xs_c = emcalc->ComputeCrossSectionPerVolume(E, part, proc_compton, mat->GetName());
    double xs_r = emcalc->ComputeCrossSectionPerVolume(E, part, proc_rayleigh, mat->GetName());
    double mu = (xs_c+xs_r)/d;
    // DD(mu);
    ed.push_back(mu);
  }
  DD("done");
  
  typedef itk::ImageRegionIterator<InputImageType> IteratorType;
  IteratorType pi(input,input->GetLargestPossibleRegion());
  pi.GoToBegin();
  GateImage::const_iterator data = gate_image->begin();
  while (!pi.IsAtEnd()) {
    double e = ed[(int)(*data)];
    pi.Set(e); // FIXME --> to change 
    ++pi;
    ++data;
  }

  // debug write image
  typedef itk::ImageFileWriter<OutputImageType> InputWriterType;
  typename InputWriterType::Pointer iwriter = InputWriterType::New();
  iwriter->SetFileName("titi.mhd");
  iwriter->SetInput(input);
  iwriter->Update();
  DD("done write titi input");

  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  const ConstantImageSourceType::Pointer projInput = ConstantImageSourceType::New();
  size[2] = 1;
  size[0] = size[1] = 128;
  spacing[0] = spacing[1] = 3;
  // projInput->SetOrigin(origin);
  projInput->SetSpacing(spacing);
  projInput->SetSize(size);
  projInput->SetConstant(0.);
  projInput->Update();

  // geometry
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  double sid = 1000;
  double sdd = 1536;
  double gantryAngle = 0.0;
  double sx = (size[0] * spacing[0])/2.0;
  double sy = (size[1] * spacing[1])/2.0;
  DD(sx);
  DD(sy);
  geometry->AddProjection(sid, sdd, gantryAngle, -sx, -sy);
  //  cout <<geometry;

  // projection
  typedef rtk::JosephForwardProjectionImageFilter<InputImageType, OutputImageType> JFPType;
  JFPType::Pointer jfp = JFPType::New();
  jfp->InPlaceOff();
  jfp->SetInput(projInput->GetOutput()); // output //FIXME
  jfp->SetInput(1, input); // input
  jfp->SetGeometry(geometry);
  DD("START");
  jfp->Update();
  DD("done");
  output = jfp->GetOutput();
  
  typedef itk::ImageFileWriter<OutputImageType> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetFileName("output/toto.mhd");
  writer->SetInput(output);
  writer->Update();
  */

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

  typename OutputImageType::SizeType size;
  size[2] = 1;
  size[0] = GetDetectorResolution()[0];
  size[1] = GetDetectorResolution()[1];
  DD(size);

  typename OutputImageType::SpacingType spacing;
  spacing[2] = 1.0;
  spacing[0] = detector->GetHalfDimension(0)*2.0/size[0]*mm; //FIXME (in mm ?)
  spacing[1] = detector->GetHalfDimension(1)*2.0/size[1]*mm;
  DD(spacing);

  typename OutputImageType::PointType origin;
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
void GateHybridForcedDetectionActor::CreateMuImage(const std::vector<double> & label2mu, 
                                                   const GateImage * gate_image, 
                                                   InputImageType * input)
{
  DD("CreateMuImage");
  
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
void GateHybridForcedDetectionActor::GenerateDRR(const InputImageType * input, 
                                                 const OutputImageType * projInput, 
                                                 GeometryType * geometry,
                                                 OutputImageType * output)
{
  DD("GenerateDRR");
  
  typedef rtk::JosephForwardProjectionImageFilter<InputImageType, OutputImageType> JFPType;
  JFPType::Pointer jfp = JFPType::New();
  jfp->InPlaceOff();
  jfp->SetInput(projInput);   // output
  jfp->SetInput(1, input); // input
  jfp->SetGeometry(geometry);
  DD("START");
  jfp->Update();
  DD("done");
  output = jfp->GetOutput();  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateHybridForcedDetectionActor::CreateLabelToMuConversion(const double E, 
                                                               std::vector<double> & label2mu)
{

}
//-----------------------------------------------------------------------------


#endif

