/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#ifndef GateSimplifiedDecayTransition_H
#define GateSimplifiedDecayTransition_H 1
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <functional>
#include "Randomize.hh"
#include <G4PhysicalConstants.hh>

class GateSimplifiedDecay;
typedef std::pair<std::string,double> psd;


class GateSimplifiedDecayTransition {

  friend class GateSimplifiedDecay;

 public:

  GateSimplifiedDecayTransition(int cs, int ns, double pr,  std::function<psd(GateSimplifiedDecayTransition*)> act, double en=0, double ampl=0, double norm=0, int Z=0):
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
    std::cout
	 << currentState << ", "
	 <<  nextState << ", "
	 <<  probability << ", "
	 <<  energy << ", "
	 <<  amplitude << ", "
	 <<  normalisationFactor << ", "
	 <<  atomicNumber
	 <<  std::endl;
  }

  GateSimplifiedDecayTransition* sample(int n){
    for (int i=0; i<n; i++) std::cout << majoredHitAndMiss() << std::endl;
    return this;
  }


 private:

  double fermiFunction(double eKin);
  double simpleHitAndMiss();
  double majoredHitAndMiss();

  inline double majoringFunction(double x){
    return amplitude*sin(pi*x/energy);
  }
  inline double majoringInverseCDF(double x){
    return energy/pi*acos(-pi*x/(energy * amplitude));
  }
  inline double CDFRandom(){
    return majoringInverseCDF( energy * amplitude/pi * (2*G4UniformRand()-1) );
  }

 private:
  int    sequenceNumber;
  int    currentState;
  int    nextState;
  double probability;
  std::function<psd(GateSimplifiedDecayTransition*)> action;
  double energy;                                    //  Maximum energy (inMeV)
  double amplitude;                                 //  Majoring function amplitude (for positrons)
  double normalisationFactor;                       //  normalisation factor for PDF (for positrons)
  int    atomicNumber;

  static  double kAlphaSquared;

};



#endif
