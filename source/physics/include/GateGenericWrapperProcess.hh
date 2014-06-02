/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GenericWrapperProcess
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEGENERICWRAPPERPROCESS_HH
#define GATEGENERICWRAPPERPROCESS_HH


#include "G4WrapperProcess.hh"
#include "GateFilterManager.hh"

#include "G4EmCalculator.hh"

class G4MaterialCutsCouple;

class GenericWrapperProcess : public G4WrapperProcess
{
public:
  GenericWrapperProcess(G4String t);
  virtual ~GenericWrapperProcess();

  virtual G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& step);
  virtual G4double PostStepGetPhysicalInteractionLength(
                             const G4Track& track,
                             G4double   previousStepSize,
                             G4ForceCondition* condition
                            ) ;  

  void SetSplitFactor(G4double);
  void SetCSEFactor(G4double);
  void SetIsActive(G4bool);
  void SetKeepSec(G4bool b){mkeepSec=b;}
 // G4bool AddFilter(G4String filterType );

   G4bool GetIsActive();
   G4int GetFactor();
   G4int GetNSecondaries();

  GateFilterManager * GetFilterManagerPrimary(){return pFilterManagerPrimary;}
  GateFilterManager * GetFilterManagerSecondary(){return pFilterManagerSecondary;}

  void IncFilterManagerSecondary(){mNFilterSecondary++;}
  void IncFilterManagerPrimary(){mNFilterPrimary++;}

  void SetGenericWrapperProcess(G4String t);
  //G4String GetName(){return name;}

  void Initialisation(G4String particleName);

private:
  GateFilterManager * pFilterManagerPrimary;
  GateFilterManager * pFilterManagerSecondary;

  G4EmCalculator * emcalc;

  bool mCSEnhancement;
  bool mSplitting;
  bool mInitCS;
  bool mRR;

  std::map<G4String,std::vector<double> > theListOfBranchRatioFactor;

  G4int mSplitFactor;
  G4int mNSecondaries;
  G4bool mActive;
  G4double mWeight;
  G4int mNFilterPrimary;
  G4int mNFilterSecondary;
  //G4String name;

  G4double mCSEFactor;
  G4bool mkeepSec;

  int mNbins; //number of bins of the table of cross section ratios in function of energy
  double mEneMax; //maximum energy of the table of cross section ratios in function of energy

};

#endif
