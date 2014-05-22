/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSimplifiedDecayTransition_H
#define GateSimplifiedDecayTransition_H 1
#include "math.h"
#include <limits>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>

#include "Randomize.hh"

using namespace std;

class GateSimplifiedDecay;
typedef pair<string,double> psd;


class GateSimplifiedDecayTransition {

  friend class GateSimplifiedDecay;

 public:

  GateSimplifiedDecayTransition(int cs, int ns, double pr,  mem_fun_t<psd,GateSimplifiedDecayTransition> act, double en=0, double ampl=0, double norm=0, int Z=0):
    currentState(cs),
    nextState(ns),
    probability(pr),
    action(act),
    energy(en),
    amplitude(ampl),
    normalisationFactor(norm),
    atomicNumber(Z){;}

  ~GateSimplifiedDecayTransition(){;}

  inline psd issueGamma(){
    return psd("gamma",energy);
  }
  inline psd issuePositron(){
    //    return  psd("e+",majoredHitAndMiss());
    return  psd("e+",simpleHitAndMiss());
  }
  inline psd issueNone(){
    return  psd("none",0);
  }



  void print(){
    cout
	 << currentState << ", "
	 <<  nextState << ", "
	 <<  probability << ", "
	 <<  energy << ", "
	 <<  amplitude << ", "
	 <<  normalisationFactor << ", "
	 <<  atomicNumber
	 <<  endl;
  }

  GateSimplifiedDecayTransition* sample(int n){
    for (int i=0; i<n; i++) cout << majoredHitAndMiss() << endl;
    return this;
  }


 private:

  double fermiFunction(double eKin);
  double simpleHitAndMiss();
  double majoredHitAndMiss();

  inline double majoringFunction(double x){
    return amplitude*sin(Pi*x/energy);
  }
  inline double majoringInverseCDF(double x){
    return energy/Pi*acos(- Pi*x/(energy * amplitude));
  }
  inline double CDFRandom(){
    return majoringInverseCDF( energy * amplitude/Pi * (2*G4UniformRand()-1) );
  }

 private:
  int    sequenceNumber;
  int    currentState;
  int    nextState;
  double probability;
  mem_fun_t<psd,GateSimplifiedDecayTransition> action;
  double energy;                                    //  Maximum energy (inMeV)
  double amplitude;                                 //  Majoring function amplitude (for positrons)
  double normalisationFactor;                       //  normalisation factor for PDF (for positrons)
  int    atomicNumber;

  static  double kAlpha;
  static  double kAlphaSquared;
  static  double Pi;
  static  double E;

};



#endif
