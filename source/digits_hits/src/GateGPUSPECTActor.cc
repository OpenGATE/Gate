/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEGPUSPECTACTOR_CC
#define GATEGPUSPECTACTOR_CC

#include "GateGPUSPECTActor.hh"

#include "GateMiscFunctions.hh"
#include "GateRunManager.hh"
//#include "GateVImageVolume.hh"
#include "GateVVolume.hh"
#include "GateBox.hh"
#include "GateRandomEngine.hh"
#include "GateApplicationMgr.hh"

#include "GateDetectorConstruction.hh"
#include "GateMaterialDatabase.hh"

#include <G4VoxelLimits.hh>

#include "G4Gamma.hh"
#include "G4Electron.hh"

#include <sys/time.h>

//-----------------------------------------------------------------------------
GateGPUSPECTActor::GateGPUSPECTActor(G4String name, G4int depth):
  GateVActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateGPUSPECTActor() -- begin"<<G4endl);
  gpu_input = 0;
  mGPUDeviceID = 0;
  max_buffer_size = 5;
  mHoleHexaHeight = 0.0;
  mHoleHexaRadius = 0.0;
  mHoleHexaRotAxis = G4ThreeVector(0, 0, 0);
  mHoleHexaRotAngle = 0.0;
  mHoleHexaMat = G4String("");
  mCubArrayRepNumX = 1;
  mCubArrayRepNumY = 1;
  mCubArrayRepNumZ = 1;
  mCubArrayRepVecX = 0.0;
  mCubArrayRepVecY = 0.0;
  mCubArrayRepVecZ = 0.0;
  mLinearRepNum = 1;
  mLinearRepVecX = 0.0;
  mLinearRepVecY = 0.0;
  mLinearRepVecZ = 0.0;
  pMessenger = new GateGPUSPECTActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GateGPUSPECTActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateGPUSPECTActor::~GateGPUSPECTActor()  {
  delete pMessenger;
  GateGPUCollimIO_Input_delete(gpu_input);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateGPUSPECTActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateGPUSPECTActor -- Construct - begin" << G4endl);
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);

  ResetData();
  GateMessageDec("Actor", 4, "GateGPUSPECTActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GateGPUSPECTActor::SaveData() {
  //FIXME
  DD("SaveData");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetGPUDeviceID(G4int n) {
  mGPUDeviceID = n;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetGPUBufferSize(G4int n) {
  max_buffer_size = n;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetHoleHexaHeight(G4double d) {
  mHoleHexaHeight = d;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetHoleHexaRadius(G4double d) {
  mHoleHexaRadius = d;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetHoleHexaRotAxis(G4ThreeVector v) {
  mHoleHexaRotAxis = v;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetHoleHexaRotAngle(G4double d) {
  mHoleHexaRotAngle = d;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetHoleHexaMaterial(G4String m) {
  mHoleHexaMat = m;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetCubArrayRepNumX(G4int n) {
  mCubArrayRepNumX = n;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetCubArrayRepNumY(G4int n) {
  mCubArrayRepNumY = n;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetCubArrayRepNumZ(G4int n) {
  mCubArrayRepNumZ = n;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetCubArrayRepVec(G4ThreeVector v) {
  mCubArrayRepVecX = v[0];
  mCubArrayRepVecY = v[1];
  mCubArrayRepVecZ = v[2];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetLinearRepNum(G4int n) {
  mLinearRepNum = n;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::SetLinearRepVec(G4ThreeVector v) {
  mLinearRepVecX = v[0];
  mLinearRepVecY = v[1];
  mLinearRepVecZ = v[2];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::ResetData() {
  DD("ResetData");

  GateGPUCollimIO_Input_delete(gpu_input);
  gpu_input = GateGPUCollimIO_Input_new();

  GateBox * vol = dynamic_cast<GateBox*>(mVolume);

  // Set materials
  std::vector<G4Material*> m;
  m.push_back((G4Material*)vol->GetMaterial());

  G4Material* colli_mat = GateDetectorConstruction::GetGateDetectorConstruction()
  								->mMaterialDatabase.GetMaterial(mHoleHexaMat);

  m.push_back((G4Material*)colli_mat);

  G4String name = vol->GetObjectName();
  //G4String name = "SPECThead";
  GateGPUCollimIO_Input_Init_Materials(gpu_input, m, name);

  gpu_input->size_x = vol->GetBoxXLength();
  gpu_input->size_y = vol->GetBoxYLength();
  gpu_input->size_z = vol->GetBoxZLength();

  gpu_input->HexaRadius = mHoleHexaRadius;
  gpu_input->HexaHeight = mHoleHexaHeight;

  gpu_input->CubRepNumY = mCubArrayRepNumY;
  gpu_input->CubRepNumZ = mCubArrayRepNumZ;

  gpu_input->CubRepVecX = mCubArrayRepVecX;
  gpu_input->CubRepVecY = mCubArrayRepVecY;
  gpu_input->CubRepVecZ = mCubArrayRepVecZ;

  gpu_input->LinRepVecX = mLinearRepVecX;
  gpu_input->LinRepVecY = mLinearRepVecY;
  gpu_input->LinRepVecZ = mLinearRepVecZ;

  gpu_input->cudaDeviceID = mGPUDeviceID;

  DD(mGPUDeviceID);

  DD(gpu_input->HexaRadius);
  DD(gpu_input->HexaHeight);

  DD(gpu_input->CubRepNumY);
  DD(gpu_input->CubRepNumZ);

  DD(gpu_input->CubRepVecX);
  DD(gpu_input->CubRepVecY);
  DD(gpu_input->CubRepVecZ);

  DD(gpu_input->LinRepVecX);
  DD(gpu_input->LinRepVecY);
  DD(gpu_input->LinRepVecZ);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUSPECTActor::BeginOfRunAction(const G4Run *)
{
  DD("GateGPUSPECTActor::BeginOfRunAction");


  // Get number of particles
  GateApplicationMgr * a = GateApplicationMgr::GetInstance();
  DD(a->IsTotalAmountOfPrimariesModeEnabled());
  DD(a->GetRequestedAmountOfPrimariesPerRun());

  // Init PRNG
  ct_photons = 0;
  unsigned int seed =
    static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());
  //DD(seed);
  srand(seed);

#ifdef GATE_USE_GPU
  int size_center = (mCubArrayRepNumY * mCubArrayRepNumZ) + ((mCubArrayRepNumY - 1) * (mCubArrayRepNumZ - 1));

  // Init GPU' stuff
  //GPU_GateTransTomo_init(gpu_input, gpu_materials, gpu_phantom,
  //                       gpu_photons, cpu_photons, max_buffer_size, seed);

  GPU_GateSPECT_init(gpu_input, gpu_collim, cpu_centerOfHexagons, gpu_centerOfHexagons,
                     gpu_photons, cpu_photons, gpu_materials, max_buffer_size, size_center, seed);

  //DD(max_buffer_size);
#endif

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUSPECTActor::EndOfRunAction(const G4Run *)
{
    // Remaining particles?
    if (ct_photons != 0) {

#ifdef GATE_USE_GPU
        GPU_GateSPECT(gpu_collim, cpu_centerOfHexagons, gpu_centerOfHexagons, gpu_photons,
        			   cpu_photons, gpu_materials, ct_photons);
#endif

        // G4
        static G4EventManager * em = G4EventManager::GetEventManager();
        G4StackManager * sm = em->GetStackManager();

        unsigned int ct = 0;
        while (ct < ct_photons) {
          if (cpu_photons.active[ct]) {

            // Create new particle
            G4ThreeVector dir(cpu_photons.dx[ct],
                              cpu_photons.dy[ct], cpu_photons.dz[ct]);
            dir /= dir.mag();
            G4ThreeVector position(cpu_photons.px[ct]*mm,
                                   cpu_photons.py[ct]*mm,
                                   cpu_photons.pz[ct]*mm);

            G4DynamicParticle * dp = new G4DynamicParticle(G4Gamma::Gamma(), dir,
                                                           cpu_photons.E[ct]*MeV);
            double time = cpu_photons.t[ct];
            G4Track * newTrack = new G4Track(dp, time, position);

            // FIXME
            static long trackid=0;
            newTrack->SetTrackID(cpu_photons.trackID[ct]+trackid);
            ++trackid;
            newTrack->SetParentID(666);

            // Insert
            sm->PushOneTrack(newTrack);
          }
          ct++;

        }
    }

    // Shutdown the GPU
#ifdef GATE_USE_GPU
    //GPU_GateTransTomo_end(gpu_materials, gpu_phantom, gpu_photons, cpu_photons);
    GPU_GateSPECT_end(gpu_centerOfHexagons, gpu_photons, cpu_photons, gpu_materials);
#endif

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#define EPS 1.0e-03f
void GateGPUSPECTActor::UserSteppingAction(const GateVVolume * /*v*/,
                                              const G4Step * step)
{
  GateDebugMessage("Actor", 4, "GateGPUSPECTActor -- UserSteppingAction" << G4endl);

  //DD("GateGPUSPECTActor::UserSteppingAction");

  // Check if we are on the boundary
  G4StepPoint * preStep = step->GetPreStepPoint();

  //DD(preStep->GetPosition());

  //  G4StepPoint * postStep = step->GetPostStepPoint();
  if (preStep->GetStepStatus() != fGeomBoundary) {
    // This is not the first step in the volume
    	step->GetTrack()->SetTrackStatus( fStopAndKill ); // FIXME : one step more to remove.
    	return;
  }

  if (step->GetTrack()->GetDefinition() != G4Gamma::Gamma()) return;

  // Compute local position in the collimator space
  int h = preStep->GetTouchable()->GetHistory()->GetDepth();
  G4ThreeVector localPosition = preStep->GetTouchable()->GetHistory()
  		->GetTransform(h).TransformPoint(preStep->GetPosition());

  // STEP1 ---------------------------------

  // Store a photon
  cpu_photons.E[ct_photons] = preStep->GetKineticEnergy()/MeV;
  cpu_photons.eventID[ct_photons] =
                        GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
  cpu_photons.trackID[ct_photons] = step->GetTrack()->GetTrackID();
  cpu_photons.t[ct_photons] = preStep->GetGlobalTime();
  cpu_photons.type[ct_photons] = 22; // G4_gamma

  cpu_photons.px[ct_photons] = localPosition.x() + EPS;
  cpu_photons.py[ct_photons] = localPosition.y();
  cpu_photons.pz[ct_photons] = localPosition.z();

  // Need to change the world frame
  /*cpu_photons.px[ct_photons] = localPosition.x() + half_phan_size_x;
  cpu_photons.py[ct_photons] = localPosition.y() + half_phan_size_y;
  cpu_photons.pz[ct_photons] = localPosition.z() + half_phan_size_z;

  // If the particle is just on the volume boundary, push it inside the volume
  //  (this fix a bug during the navigation)
  if (cpu_photons.px[ct_photons]== 2.0f*half_phan_size_x) cpu_photons.px[ct_photons] -= EPS;
  if (cpu_photons.py[ct_photons]== 2.0f*half_phan_size_y) cpu_photons.py[ct_photons] -= EPS;
  if (cpu_photons.pz[ct_photons]== 2.0f*half_phan_size_z) cpu_photons.pz[ct_photons] -= EPS;*/

  // Compute local direction in the collimator space
  G4ThreeVector localMomentum = preStep->GetTouchable()->GetHistory()
  		->GetTransform(h).TransformAxis(preStep->GetMomentumDirection());

  cpu_photons.dx[ct_photons] = localMomentum.x();
  cpu_photons.dy[ct_photons] = localMomentum.y();
  cpu_photons.dz[ct_photons] = localMomentum.z();

  /*DD(cpu_photons.dx[ct_photons]);
  DD(cpu_photons.dy[ct_photons]);
  DD(cpu_photons.dz[ct_photons]);*/

  cpu_photons.endsimu[ct_photons] = 0;
  cpu_photons.active[ct_photons] = 1;
  cpu_photons.seed[ct_photons] = rand();

  // STEP2 ---------------------------------

  // We kill the particle without mercy
  step->GetTrack()->SetTrackStatus( fStopAndKill );
  ct_photons++;

  // STEP3 ---------------------------------

  // if enough particles in the buffer, start the gpu tracking
  if (ct_photons == max_buffer_size) {

#ifdef GATE_USE_GPU

    GPU_GateSPECT(gpu_collim, cpu_centerOfHexagons, gpu_centerOfHexagons, gpu_photons,
    				cpu_photons, gpu_materials, ct_photons);

#endif

    // G4
    static G4EventManager * em = G4EventManager::GetEventManager();
    G4StackManager * sm = em->GetStackManager();

    ct_photons = 0;
    while (ct_photons < max_buffer_size) {
      if (cpu_photons.active[ct_photons]) {

        // Create new particle
        G4ThreeVector dir(cpu_photons.dx[ct_photons],
                          cpu_photons.dy[ct_photons], cpu_photons.dz[ct_photons]);
        dir /= dir.mag();

		G4ThreeVector position(cpu_photons.px[ct_photons],
                               cpu_photons.py[ct_photons],
                               cpu_photons.pz[ct_photons]);

        // FIXME : Need a complete transformation with phantom centre
        //         (see below when photons are stored)
        /*G4ThreeVector position(cpu_photons.px[ct_photons] - half_phan_size_x,
                               cpu_photons.py[ct_photons] - half_phan_size_y,
                               cpu_photons.pz[ct_photons] - half_phan_size_z);*/

        G4DynamicParticle * dp = new G4DynamicParticle(G4Gamma::Gamma(), dir,
                                                       cpu_photons.E[ct_photons]*MeV);
        double time = cpu_photons.t[ct_photons];
        G4Track * newTrack = new G4Track(dp, time, position);

        // FIXME
        static long trackid=0;
        newTrack->SetTrackID(cpu_photons.trackID[ct_photons]+trackid);
        ++trackid;
        newTrack->SetParentID(666);

        // Insert
        sm->PushOneTrack(newTrack);
      }
      ct_photons++;

    } // while

    ct_photons = 0;
  } // if

}
#undef EPS
//-----------------------------------------------------------------------------

#endif /* end #define GATEGPUSPECTACTOR_CC */
