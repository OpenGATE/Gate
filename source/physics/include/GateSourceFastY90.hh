#include "GateConfiguration.h"

#ifndef GATEFASTY90SOURCE_HH
#define GATEFASTY90SOURCE_HH

#include "G4Event.hh"
#include "globals.hh"
#include "G4VPrimaryGenerator.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryVertex.hh"
#include "G4ParticleMomentum.hh"

#include "GateVSource.hh"
#include "GateSourceFastY90Messenger.hh"

//#include "GateRunManager.hh"

#include "GateUserActions.hh"

class GateSourceFastY90 : public GateVSource
{
public:
  GateSourceFastY90(G4String name);
  ~GateSourceFastY90();

  G4int GeneratePrimaries(G4Event *event);
  void GeneratePrimaryVertex(G4Event* event);

  void SetMinEnergy(G4double energy)
  {
    mMinEnergy = energy;
    CalculateEnergyTable();
  }
  G4double GetMinEnergy() {return mMinEnergy;}

  void SetPositronProbability(G4double probability) {mPosProb = probability;}

  void LoadVoxelizedPhantom(G4String filename);
  void SetPhantomPosition(G4ThreeVector pos);

  G4double GetNextTime( G4double timeStart );

protected:
  G4double mMinEnergy;  // minimum energy below which photons won't be generated

  G4double mBremProb;   // probability of a brem photon above minEnergy
  G4double mPosProb;    // probability of producing a positron
  G4double mGammaProb;  // probability of emitting a 1.76 MeV gamma (TODO: not yet implemented)

  static const G4double mEnergyTable[200];     // energy probability table in 10 keV steps
  static const G4double mRangeTable[100][120]; // probability table of range in 0.1 mm increments
  static const G4double mAngleTable[100][180]; // probability table of angles in 1 degree increments
  static const G4double mPositronEnergyTable[738];

  G4double *mCumulativeEnergyTable; // cumulative histogram of energy probability
  G4double **mCumulativeRangeTable; // cumulative histogram of energy probability
  G4double **mCumulativeAngleTable; // cumulative probability table of angle

  G4ParticleDefinition* pGammaParticleDefinition;
  G4ParticleDefinition* pPositronParticleDefinition;

  void CalculateEnergyTable(); // create the energy table

  G4double GetBremsstrahlungEnergy();
  G4double GetRange(G4double energy);   // distance from source to point of bremsstrahlung emission
  G4double GetAngle(G4double energy);   //
  G4ThreeVector PerturbVector(G4ThreeVector original, G4double theta);
  G4double GetPositronEnergy();

};

#endif
