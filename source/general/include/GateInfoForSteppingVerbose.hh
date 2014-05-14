/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateSteppingVerbose
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
          david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEInfoSteppingVerbose_hh
#define GATEInfoSteppingVerbose_hh 1

class GateInfoForSteppingVerbose  {
public:   
  GateInfoForSteppingVerbose(){mEnergy=0.;mTime=0.;mVolume="";mProcess="";mParticle="";}
  ~GateInfoForSteppingVerbose(){}

  void SetEnergy(double energy){mEnergy=energy;}
  double GetEnergy(){return mEnergy;}

  void SetTime(double time){mTime=time;}
  double GetTime(){return mTime;}
  void AddTime(double time){mTime+=time;}

  void SetVolume(G4String volume){mVolume=volume;}
  G4String GetVolume(){return mVolume;}

  void SetProcess(G4String process){mProcess=process;}
  G4String GetProcess(){return mProcess;}

  void SetParticle(G4String particle){mParticle=particle;}
  G4String GetParticle(){return mParticle;}

  void Clear(){mEnergy=0.;mTime=0.;mVolume="";mProcess="";mParticle="";}

protected:
  double mEnergy;
  double mTime;
  G4String mVolume;
  G4String mProcess;
  G4String mParticle;
};

#endif
