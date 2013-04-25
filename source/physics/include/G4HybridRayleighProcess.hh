/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  G4HybridRayleighProcess
  \author fabien.baldacci@creatis.insa-lyon.fr
*/


#ifndef G4HYBRIDRAYLEIGHPROCESS_HH
#define G4HYBRIDRAYLEIGHPROCESS_HH


#include "G4WrapperProcess.hh"
#include "G4VEmProcess.hh"

class G4HybridRayleighProcess : public G4WrapperProcess
{
public:
  G4HybridRayleighProcess();
  ~G4HybridRayleighProcess();
  G4VEmProcess* GetEmProcess();
  G4VParticleChange* PostStepDoIt(const G4Track& track, const G4Step& step);
  void SetInVolume() {mCurrentSplit = mInVolumeSplit;}
  void SetOutVolume() {mCurrentSplit = mOutVolumeSplit;}
  void SetInVolumeSplit(int splitFactor) {mInVolumeSplit = splitFactor;}
  void SetOutVolumeSplit(int splitFactor) {mOutVolumeSplit = splitFactor;}
  int GetInVolumeSplitFactor() {return mInVolumeSplit;}
  int GetOutVolumeSplitFactor() {return mOutVolumeSplit;}
  
private:
  int mInVolumeSplit;
  int mOutVolumeSplit;
  int mCurrentSplit;
};

#endif
