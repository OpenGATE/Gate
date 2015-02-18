/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GateBinaryCascade_hh
#define GateBinaryCascade_hh

#include "G4VIntraNuclearTransportModel.hh"
#include "G4ReactionProductVector.hh"
#include "G4KineticTrackVector.hh"
#include "G4ListOfCollisions.hh"
#include "G4V3DNucleus.hh"
#include "G4Fancy3DNucleus.hh"
#include "G4Fragment.hh"
#include "G4VFieldPropagation.hh"
#include "G4VScatterer.hh"
#include "G4LorentzVector.hh"
#include "G4LorentzRotation.hh"

#include "G4BCDecay.hh"
#include "G4BCLateParticle.hh"
#include "G4BCAction.hh"

class G4CollisionManager;

class G4Track;
class G4KineticTrack;
class G43DNucleus;
class G4Scatterer;

class GateBinaryCascade : public G4VIntraNuclearTransportModel
{
public:

  GateBinaryCascade();
  GateBinaryCascade(const GateBinaryCascade & right);

  virtual ~GateBinaryCascade();

  const GateBinaryCascade& operator=(GateBinaryCascade & right);
  G4int operator==(GateBinaryCascade& right) {return (this == &right);}
  G4int operator!=(GateBinaryCascade& right) {return (this != &right);}

  G4HadFinalState* ApplyYourself(const G4HadProjectile& aTrack, 
                                              G4Nucleus& theNucleus);
  virtual G4ReactionProductVector * Propagate(G4KineticTrackVector *,
					      G4V3DNucleus *);

private:

  G4int GetTotalCharge(std::vector<G4KineticTrack *> & aV)
  {
    G4int result = 0;
    std::vector<G4KineticTrack *>::iterator i;
    for(i = aV.begin(); i != aV.end(); ++i)
    {
       result += G4lrint((*i)->GetDefinition()->GetPDGCharge());
    }
    return result;
  }
  G4int GetTotalBaryonCharge(std::vector<G4KineticTrack *> & aV)
  {
    G4int result = 0;
    std::vector<G4KineticTrack *>::iterator i;
    for(i = aV.begin(); i != aV.end(); ++i)
    {
       if ( (*i)->GetDefinition()->GetBaryonNumber() != 0 ){ 
             result += G4lrint((*i)->GetDefinition()->GetPDGCharge());
       }
    }
    return result;
  }
  void PrintWelcomeMessage();
  void BuildTargetList();
  void FindCollisions(G4KineticTrackVector *);
  void FindDecayCollision(G4KineticTrack *);
  void FindLateParticleCollision(G4KineticTrack *);
  G4bool ApplyCollision(G4CollisionInitialState *);
  G4bool Capture(G4bool verbose=false);
  G4bool Absorb();
  G4bool CheckPauliPrinciple(G4KineticTrackVector *);
  G4double GetExcitationEnergy();
  G4bool CheckDecay(G4KineticTrackVector *);
  void CorrectFinalPandE();
  void UpdateTracksAndCollisions(G4KineticTrackVector * oldSecondaries,
				 G4KineticTrackVector * oldTarget,
				 G4KineticTrackVector * newSecondaries);
  G4bool DoTimeStep(G4double timeStep);
  G4KineticTrackVector* CorrectBarionsOnBoundary(G4KineticTrackVector *in, 
                                               G4KineticTrackVector *out);
  G4Fragment * FindFragments();
  void StepParticlesOut();
  G4LorentzVector GetFinal4Momentum();
  G4LorentzVector GetFinalNucleusMomentum();
  G4ReactionProductVector * Propagate1H1(G4KineticTrackVector *,
					 G4V3DNucleus *);
  G4double GetIonMass(G4int Z, G4int A);
  
// utility methods
  G4ThreeVector GetSpherePoint(G4double r, const G4LorentzVector & momentumdirection);
  void ClearAndDestroy(G4KineticTrackVector * ktv);
  void ClearAndDestroy(G4ReactionProductVector * rpv);

// for debugging purpose
  void PrintKTVector(G4KineticTrackVector * ktv, std::string comment=std::string(""));
  void PrintKTVector(G4KineticTrack* kt, std::string comment=std::string(""));
  void DebugApplyCollision(G4CollisionInitialState * collision, 
                           G4KineticTrackVector *products);
  void DebugEpConservation(const G4HadProjectile & aTrack, G4ReactionProductVector* products);			   
			   
private:
  G4KineticTrackVector theProjectileList;
  G4KineticTrackVector theTargetList;
  G4KineticTrackVector theSecondaryList;
  G4KineticTrackVector theCapturedList;
  G4KineticTrackVector theFinalState;

  G4ExcitationHandler * theExcitationHandler;
  G4CollisionManager * theCollisionMgr;
  
  G4Scatterer * theH1Scatterer;

  std::vector<G4BCAction *> theImR;
  G4BCDecay * theDecay;
  G4BCLateParticle * theLateParticle;
  G4VFieldPropagation * thePropagator;
  G4double theCurrentTime;
  G4double theBCminP;
  G4double theCutOnP;
  G4double theCutOnPAbsorb;
  G4LorentzVector theInitial4Mom;
  G4int currentA, currentZ;
  G4double massInNucleus;
  G4double currentInitialEnergy;   // for debugging
  G4LorentzRotation precompoundLorentzboost;
  G4double theOuterRadius;
  G4bool thePrimaryEscape;
  const G4ParticleDefinition * thePrimaryType;
  G4ThreeVector theMomentumTransfer;
  
  

};

#endif




