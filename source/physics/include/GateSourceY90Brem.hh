#include "GateConfiguration.h"

#ifndef GATEY90BREMSOURCE_HH
#define GATEY90BREMSOURCE_HH

#include "G4Event.hh"
#include "globals.hh"
#include "G4VPrimaryGenerator.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4PrimaryVertex.hh"
#include "G4ParticleMomentum.hh"

#include <iomanip>
#include <vector>

#include "GateVSource.hh"
#include "GateSourceY90BremMessenger.hh"

//#include "GateRunManager.hh"

#include "GateUserActions.hh"

class GateSourceY90Brem : public GateVSource
{
public:
  GateSourceY90Brem(G4String name);
  ~GateSourceY90Brem();

//  void Initialize() {};

  G4int GeneratePrimaries(G4Event *event);
  void GeneratePrimaryVertex(G4Event* event);

  void SetMinEnergy(G4double energy) {mMinEnergy = energy;}
  G4float GetMinEnergy() {return mMinEnergy;}

  G4double GetNextTime( G4double timeStart );

protected:
  G4double mMinEnergy;  // minimum energy below which photons won't be generated TODO: not implemented
  G4double mBremProb;   // probability of a brem photon above minEnergy
  G4double mPosProb; // probability of producing a positron

  static const G4double mEnergyTable[200];  // energy probability table in 10 keV steps
  static const G4double mRangeTable[100][120]; // probability table of range in 0.1 mm increments
  static const G4double mAngleTable[100][180]; // probability table of angles in 1 degree increments

  G4double *mCumulativeEnergyTable; // cumulative histogram of energy probability
  G4double **mCumulativeRangeTable; // cumulative histogram of energy probability
  G4double **mCumulativeAngleTable; // cumulative probability table of angle in 1 degree increments

  G4ParticleDefinition* pGammaParticleDefinition;
  G4ParticleDefinition* pPositronParticleDefinition;

  G4double GetEnergy();
  G4double GetRange(G4double energy);
  G4double GetAngle(G4double energy);
  G4ThreeVector PerturbVector(G4ThreeVector original, G4double theta);
};

#endif
