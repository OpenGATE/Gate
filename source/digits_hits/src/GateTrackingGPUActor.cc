/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATETRACKINGGPUACTOR_CC
#define GATETRACKINGGPUACTOR_CC

#include "GateTrackingGPUActor.hh"
#include "GateTrackingGPUActorIO.hh"
#include "GateMiscFunctions.hh"
#include "GateRunManager.hh"
#include "GateVImageVolume.hh"
#include "GateRandomEngine.hh"
#include "GateApplicationMgr.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"

#include <sys/time.h>

//-----------------------------------------------------------------------------
GateTrackingGPUActor::GateTrackingGPUActor(G4String name, G4int depth):
  GateVActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateTrackingGPUActor() -- begin"<<G4endl);
  gpu_input = 0;
  gpu_output = 0;
  mCudaDeviceID = 0;
  max_buffer_size = 5;
  pMessenger = new GateTrackingGPUActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateTrackingGPUActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateTrackingGPUActor::~GateTrackingGPUActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateTrackingGPUActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateTrackingGPUActor -- Construct - begin" << G4endl);
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  ResetData();
  GateMessageDec("Actor", 4, "GateTrackingGPUActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateTrackingGPUActor::SaveData() {
  //FIXME
  DD("SaveData");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActor::SetCudaDeviceID(int n) {
  mCudaDeviceID = n;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActor::SetGPUBufferSize(int n) {
  max_buffer_size = n;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActor::ResetData() {
  DD("ResetData");

  GateTrackingGPUActorInput_delete(gpu_input);
  gpu_input = GateTrackingGPUActorInput_new();

  GateVImageVolume * im = dynamic_cast<GateVImageVolume*>(mVolume);
  G4ThreeVector s = im->GetImage()->GetResolution();
  gpu_input->phantom_size_x = s.x();
  gpu_input->phantom_size_y = s.y();
  gpu_input->phantom_size_z = s.z();

  G4ThreeVector spacing = im->GetImage()->GetVoxelSize();
  gpu_input->phantom_spacing_x = spacing.x();
  gpu_input->phantom_spacing_y = spacing.y();
  gpu_input->phantom_spacing_z = spacing.z();
  
  gpu_input->cudaDeviceID = mCudaDeviceID;
  DD(gpu_input->phantom_size_x);
  DD(gpu_input->phantom_size_y);
  DD(gpu_input->phantom_size_z);
  DD(mCudaDeviceID);

  // Data in unsigned int
  DD(im->GetImage()->end()-im->GetImage()->begin());
  std::vector<float>::const_iterator iter = im->GetImage()->begin();
  while (iter != im->GetImage()->end()) {
    gpu_input->phantom_material_data.push_back((unsigned short int)*iter);
    ++iter;
  }
  DD(gpu_input->phantom_material_data.size());
  
  GateTrackingGPUActorOutput_delete(gpu_output);
  gpu_output = GateTrackingGPUActorOutput_new();
  DD("end");

  DD(max_buffer_size);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActor::BeginOfRunAction(const G4Run *)
{
  DD("GateTrackingGPUActor::BeginOfRunAction");
  // Set materials
  GateVImageVolume * im = dynamic_cast<GateVImageVolume*>(mVolume);
  GateTrackingGPUActorInput_Init_Materials(gpu_input, im);
  // Get number of particles
  GateApplicationMgr * a = GateApplicationMgr::GetInstance();
  DD(a->IsTotalAmountOfPrimariesModeEnabled());
  DD(a->GetRequestedAmountOfPrimariesPerRun());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActor::UserSteppingAction(const GateVVolume * /*v*/, 
                                              const G4Step * step)
{
  GateDebugMessage("Actor", 4, "GateTrackingGPUActor -- UserSteppingAction" << G4endl);
  
  // Check if we are on the boundary
  G4StepPoint * preStep = step->GetPreStepPoint();
  //  G4StepPoint * postStep = step->GetPostStepPoint();
  if (preStep->GetStepStatus() != fGeomBoundary) { 
    // This is not the first step in the volume 
    step->GetTrack()->SetTrackStatus( fStopAndKill ); // FIXME : one step more to remove.
    return;
  }

  // STEP1 ---------------------------------
  // Store a particle
  GateTrackingGPUActorParticle p;
  // DD(step->GetTrack()->GetDefinition()->GetParticleName());
  if (step->GetTrack()->GetDefinition() == G4Gamma::Gamma()) p.type = 0;
  if (step->GetTrack()->GetDefinition() == G4Electron::Electron()) p.type = 1;

  // We dont store e- yet
  if (p.type == 1) {
    // DD("stop because e-");
    return;
  }

  p.E = preStep->GetKineticEnergy()/MeV;
  p.eventID = GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
  p.trackID = step->GetTrack()->GetTrackID();
  p.t = preStep->GetGlobalTime();

  int h = preStep->GetTouchable()->GetHistory()->GetDepth ();
  // DD(h);
  const G4AffineTransform transformation = 
    preStep->GetTouchable()->GetHistory()->GetTransform(h-3);

  //FIXME : store transformation with phantom centre and explain why -3

  //  preStep->GetTouchable()->GetHistory()->GetTopTransform().Inverse();  
  // DD(transformation.NetTranslation());

  G4ThreeVector zero;
  // DD(transformation.TransformPoint(zero));

  G4ThreeVector localPosition = transformation.TransformPoint(preStep->GetPosition());
  // DD(preStep->GetPosition());
  //  DD(localPosition);
  
  p.px = localPosition.x();
  p.py = localPosition.y();
  p.pz = localPosition.z();

  // FIXME Apply rotation to momentum

  p.dx = preStep->GetMomentumDirection().x();
  p.dy = preStep->GetMomentumDirection().y();
  p.dz = preStep->GetMomentumDirection().z();
  //  GateTrackingGPUActorParticle_Print(p);

  gpu_input->particles.push_back(p); // FIXME SLOW 
  // DD(gpu_input->particles.size());

  // We kill the particle without mercy
  step->GetTrack()->SetTrackStatus( fStopAndKill );

  // STEP2 if enough particles in the buffer, start the gpu tracking
  if (gpu_input->particles.size() == max_buffer_size) {
    
    // DD(max_buffer_size);
    gpu_input->seed = static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());
    // DD(gpu_input->seed);
    GateTrackingGPUActorTrack(gpu_input, gpu_output); // FIXME

    // STEP3 get particles from gpu and create tracks
    DD(gpu_output->particles.size());
    GateTrackingGPUActorOutput::ParticlesList::const_iterator 
      iter = gpu_output->particles.begin();
    while (iter != gpu_output->particles.end()) {
      CreateNewParticle(*iter);
      ++iter;
    }
    
    static G4EventManager * em = G4EventManager::GetEventManager();
    G4StackManager * sm = em->GetStackManager(); 
    DD(sm->GetNTotalTrack());

    // Free output
    gpu_output->particles.clear();
    gpu_input->particles.clear();
  }


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActor::CreateNewParticle(const GateTrackingGPUActorParticle & p) 
{
  // DD("CreateNewParticle");
  G4ThreeVector dir(p.dx, p.dy, p.dz);
  // DD(dir);
  // // dir.setX(dir.x()*p.E);
  // // dir.setY(dir.y()*p.E);
  // // dir.setZ(dir.z()*p.E);
  // // DD(dir);
  dir /= dir.mag();
  // dir.setX(dir.x()/m);
  // dir.setY(dir.y()/m);
  // dir.setZ(dir.z()/m);
  // DD(dir);
  G4ThreeVector position(p.px*mm, p.py*mm, p.pz*mm);
  // DD(position);
  // DD(G4BestUnit(p.E, "Energy"));

  G4DynamicParticle * dp = new G4DynamicParticle(G4Gamma::Gamma(), dir, p.E*MeV);
  double time = p.t;
  G4Track * newTrack = new G4Track(dp, time, position);

  // DD(newTrack->GetMomentumDirection().mag2());

  //FIXME
  static long trackid=0;
  newTrack->SetTrackID(p.trackID+trackid);
  ++trackid;
  newTrack->SetParentID(666);//p.eventID);
  // SetTrackID ; SetParentID ; 
  
  // Insert
  // DD("insert");
  static G4EventManager * em = G4EventManager::GetEventManager();
  G4StackManager * sm = em->GetStackManager(); 
  sm->PushOneTrack(newTrack);
  // // DD("end");


}
//-----------------------------------------------------------------------------



#endif /* end #define GATETRACKINGGPUACTOR_CC */
