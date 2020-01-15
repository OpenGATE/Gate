/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GateSimplifiedDecay_H
#define GateSimplifiedDecay_H 1

// for_each
#include <algorithm>
// mem_fun
#include <functional>
#include <vector>
#include "GateSimplifiedDecayTransition.hh"

class GateSimplifiedDecay{

public:

  GateSimplifiedDecay():transitionVector(new std::vector<GateSimplifiedDecayTransition*>){;}

  ~GateSimplifiedDecay(){
    std::vector<GateSimplifiedDecayTransition*>::iterator i = this->transitionVector->begin();
    for( ; i != this->transitionVector->end(); ++i ) {
      delete *i;
    }
    transitionVector->clear();
    delete transitionVector;
  }

  void print(){
    std::for_each(  transitionVector->begin(),  transitionVector->end(), std::mem_fn( &GateSimplifiedDecayTransition::print)  );
  }

  void sample(int n, int k){
    (*transitionVector)[k]->sample(n);
  }

  inline void addTransition(GateSimplifiedDecayTransition* t ){
    transitionVector->push_back(t );
  }

  std::vector<psd>* doDecay( std::vector<psd>* );

private:
  std::vector<GateSimplifiedDecayTransition*>* transitionVector;

};




#endif
