/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateSimplifiedDecay.hh"

//----------------------------------------------------------------------------------------
//  doDecay: "decays" the isotope according to the simplified scheme.
//  The method is implemented as a state machine (or automaton).
//----------------------------------------------------------------------------------------
vector<psd>* GateSimplifiedDecay::doDecay(vector<psd>* vp){
  int currentState(0);
  int currentTransition(0);

  while (currentState>=0){

    // Find the first transition for the current state
    while ( (*transitionVector)[currentTransition]->currentState < currentState )  currentTransition++;

    // Select a transition according to probability
    double  x ( G4UniformRand() ) ;
    while(  (*transitionVector)[currentTransition]->probability < x )  currentTransition++;

    // Perform action associated with this transition
    psd part(   (*transitionVector)[currentTransition]->action( (*transitionVector)[currentTransition] ) );
    if (part.first != "none") vp->push_back( part  );

    //  ... and move on to the next state
    currentState =  (*transitionVector)[currentTransition]->nextState;

  }

  return vp;
}
//----------------------------------------------------------------------------------------
