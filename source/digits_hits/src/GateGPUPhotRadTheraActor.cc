/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateGPUPhotRadTheraActor_CC
#define GateGPUPhotRadTheraActor_CC

#include "GateGPUPhotRadTheraActor.hh"
#include "GateMiscFunctions.hh"
#include "GateRunManager.hh"
#include "GateVImageVolume.hh"
#include "GateRandomEngine.hh"
#include "GateApplicationMgr.hh"

#include "G4Gamma.hh"
#include "G4Electron.hh"

#include <sys/time.h>

//-----------------------------------------------------------------------------
GateGPUPhotRadTheraActor::GateGPUPhotRadTheraActor(G4String name, G4int depth):
  GateVActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateGPUPhotRadTheraActor() -- begin"<<G4endl);
  mGPUDeviceID = 0;
  gpu_input = 0;
  max_buffer_size = 5;
  pMessenger = new GateGPUPhotRadTheraActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateGPUPhotRadTheraActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateGPUPhotRadTheraActor::~GateGPUPhotRadTheraActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateGPUPhotRadTheraActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateGPUPhotRadTheraActor -- Construct - begin" << G4endl);
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  ResetData();
  GateMessageDec("Actor", 4, "GateGPUPhotRadTheraActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------

/*
//-----------------------------------------------------------------------------
/// Save data
void GateGPUPhotRadTheraActor::SaveData() {
  //FIXME
  DD("SaveData");

}
//-----------------------------------------------------------------------------
*/

//-----------------------------------------------------------------------------
void GateGPUPhotRadTheraActor::SetGPUDeviceID(int n) {
  mGPUDeviceID = n;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUPhotRadTheraActor::SetGPUBufferSize(int n) {
  max_buffer_size = n;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUPhotRadTheraActor::ResetData() {
  DD("ResetData");

  GateGPUIO_Input_delete(gpu_input);
  gpu_input = GateGPUIO_Input_new();

  GateVImageVolume * im = dynamic_cast<GateVImageVolume*>(mVolume);
  G4ThreeVector s = im->GetImage()->GetResolution();
  gpu_input->phantom_size_x = s.x();
  gpu_input->phantom_size_y = s.y();
  gpu_input->phantom_size_z = s.z();

  G4ThreeVector spacing = im->GetImage()->GetVoxelSize();
  gpu_input->phantom_spacing_x = spacing.x();
  gpu_input->phantom_spacing_y = spacing.y();
  gpu_input->phantom_spacing_z = spacing.z();
  
  gpu_input->cudaDeviceID = mGPUDeviceID;
  DD(gpu_input->phantom_size_x);
  DD(gpu_input->phantom_size_y);
  DD(gpu_input->phantom_size_z);
  DD(mGPUDeviceID);

  // Data in unsigned int
  DD(im->GetImage()->end()-im->GetImage()->begin());
  std::vector<float>::const_iterator iter = im->GetImage()->begin();
  while (iter != im->GetImage()->end()) {
    gpu_input->phantom_material_data.push_back((unsigned short int)*iter);
    ++iter;
  }
  DD(gpu_input->phantom_material_data.size());
  
  DD("end");

  DD(max_buffer_size);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUPhotRadTheraActor::BeginOfRunAction(const G4Run *)
{
    DD("GateGPUPhotRadTheraActor::BeginOfRunAction");
    // Set materials
    GateVImageVolume * im = dynamic_cast<GateVImageVolume*>(mVolume);
    std::vector<G4Material*> m;
    im->BuildLabelToG4MaterialVector(m);
    G4String name = im->GetObjectName();
    GateGPUIO_Input_Init_Materials(gpu_input, m, name);

    // Get number of particles
    GateApplicationMgr * a = GateApplicationMgr::GetInstance();
    DD(a->IsTotalAmountOfPrimariesModeEnabled());
    DD(a->GetRequestedAmountOfPrimariesPerRun());

    // Init PRNG
    ct_photons = 0;
    unsigned int seed = 
            static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());
    DD(seed);
    srand(seed);

    // Init phantom size
    half_phan_size_x = gpu_input->phantom_size_x * gpu_input->phantom_spacing_x * 0.5f;
    half_phan_size_y = gpu_input->phantom_size_y * gpu_input->phantom_spacing_y * 0.5f;
    half_phan_size_z = gpu_input->phantom_size_z * gpu_input->phantom_spacing_z * 0.5f;

#ifdef GATE_USE_GPU
    // Init GPU' stuff
    GPU_GatePhotRadThera_init(gpu_input, gpu_dosemap, gpu_materials, gpu_phantom,
                              gpu_photons, gpu_electrons, 
                              cpu_photons, max_buffer_size, seed);
    DD(max_buffer_size);
#endif    
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUPhotRadTheraActor::EndOfRunAction(const G4Run *)
{
    // Remaining particles?
    if (ct_photons != 0) {
#ifdef GATE_USE_GPU
        GPU_GatePhotRadThera(gpu_dosemap, gpu_materials, gpu_phantom,
                             gpu_photons, gpu_electrons, cpu_photons, ct_photons);
#endif    
    }

    // Export dosemap & shutdown the GPU
#ifdef GATE_USE_GPU
    GPU_GatePhotRadThera_end(gpu_dosemap, gpu_materials, gpu_phantom, gpu_photons, gpu_electrons,
                             cpu_photons);
#endif

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
#define EPS 1.0e-03f
void GateGPUPhotRadTheraActor::UserSteppingAction(const GateVVolume * /*v*/, 
                                                  const G4Step * step)
{
    GateDebugMessage("Actor", 4, "GateGPUPhotRadTheraActor -- UserSteppingAction" << G4endl);

    // Check if we are on the boundary
    G4StepPoint * preStep = step->GetPreStepPoint();
    //  G4StepPoint * postStep = step->GetPostStepPoint();
    if (preStep->GetStepStatus() != fGeomBoundary) { 
        // This is not the first step in the volume 
        step->GetTrack()->SetTrackStatus( fStopAndKill ); // FIXME : one step more to remove.
        return;
    }

    if (step->GetTrack()->GetDefinition() != G4Gamma::Gamma()) return;       // G4_gamma
    
    //FIXME : store transformation with phantom centre and explain why -3
    int h = preStep->GetTouchable()->GetHistory()->GetDepth ();
    const G4AffineTransform transformation = 
        preStep->GetTouchable()->GetHistory()->GetTransform(h-3);
    
    G4ThreeVector localPosition = transformation.TransformPoint(preStep->GetPosition());
    
    // STEP1 ---------------------------------
    // Store a photon
    cpu_photons.E[ct_photons] = preStep->GetKineticEnergy()/MeV;
    cpu_photons.eventID[ct_photons] = 
                          GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    cpu_photons.trackID[ct_photons] = step->GetTrack()->GetTrackID();
    cpu_photons.t[ct_photons] = preStep->GetGlobalTime();
    cpu_photons.type[ct_photons] = 22; // G4_gamma

    // Need to change the world frame
    cpu_photons.px[ct_photons] = localPosition.x() + half_phan_size_x;
    cpu_photons.py[ct_photons] = localPosition.y() + half_phan_size_y;
    cpu_photons.pz[ct_photons] = localPosition.z() + half_phan_size_z;

    // If the particle is just on the volume boundary, push it inside the volume
    //  (this fix a bug during the navigation)
    if (cpu_photons.px[ct_photons]== 2.0f*half_phan_size_x) cpu_photons.px[ct_photons] -= EPS;
    if (cpu_photons.py[ct_photons]== 2.0f*half_phan_size_y) cpu_photons.py[ct_photons] -= EPS;
    if (cpu_photons.pz[ct_photons]== 2.0f*half_phan_size_z) cpu_photons.pz[ct_photons] -= EPS;

    // FIXME Apply rotation to momentum
    cpu_photons.dx[ct_photons] = preStep->GetMomentumDirection().x();
    cpu_photons.dy[ct_photons] = preStep->GetMomentumDirection().y();
    cpu_photons.dz[ct_photons] = preStep->GetMomentumDirection().z();

    cpu_photons.endsimu[ct_photons] = 0;
    cpu_photons.active[ct_photons] = 1;
    cpu_photons.seed[ct_photons] = rand();
    
    // STEP2 ---------------------------------
    // We kill the particle without mercy
    step->GetTrack()->SetTrackStatus( fStopAndKill );

    // STEP3 ---------------------------------
    // if enough particles in the buffer, start the gpu tracking
    ct_photons++;
    if (ct_photons == max_buffer_size) {
#ifdef GATE_USE_GPU
        GPU_GatePhotRadThera(gpu_dosemap, gpu_materials, gpu_phantom,
                             gpu_photons, gpu_electrons, cpu_photons, ct_photons);
#endif    
        ct_photons = 0;

    }

}
#undef EPS
//-----------------------------------------------------------------------------


#endif /* end #define GateGPUPhotRadTheraActor_CC */
