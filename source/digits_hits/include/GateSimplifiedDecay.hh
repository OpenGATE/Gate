/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GateSimplifiedDecay_H
#define GateSimplifiedDecay_H 1

#include <vector>
#include "GateSimplifiedDecayTransition.hh"
#include "Randomize.hh"

class GateSimplifiedDecay{

public:

  GateSimplifiedDecay():transitionVector(new vector<GateSimplifiedDecayTransition*>){;}

  ~GateSimplifiedDecay(){
    vector<GateSimplifiedDecayTransition*>::iterator i = this->transitionVector->begin();
    for( ; i != this->transitionVector->end(); ++i ) {
      delete *i;
    }
    transitionVector->clear();
    delete transitionVector;
  }

  void print(){
    for_each(  transitionVector->begin(),  transitionVector->end(), mem_fun( &GateSimplifiedDecayTransition::print)  );
  }

  void sample(int n, int k){
    (*transitionVector)[k]->sample(n);
  }

  inline void addTransition(GateSimplifiedDecayTransition* t ){
    transitionVector->push_back(t );
  }

  vector<psd>* doDecay( vector<psd>* );

private:
  vector<GateSimplifiedDecayTransition*>* transitionVector;

};




#endif
