/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEFAKEPHYSICSLIST_HH
#define GATEFAKEPHYSICSLIST_HH


/*
 * \file  GateFakePhysicsList.hh
 * \brief fGate Fake Physicslist class for development
 */

#include "G4VUserPhysicsList.hh"
#include "globals.hh"

class GateFakePhysicsList : public G4VUserPhysicsList {
  
  public: 
    // Constructor
  GateFakePhysicsList():G4VUserPhysicsList() {}
  
    // Destructor
    virtual ~GateFakePhysicsList() {}
  
    // Construct process, particules and cuts
    void ConstructProcess();
    void ConstructParticle();
    void SetCuts();

  protected:
    // these methods Construct particles 
    void ConstructBosons();
    void ConstructLeptons();

  protected:
    // these methods Construct physics processes and register them
    void ConstructGeneral();
    void ConstructEM();

};
  
#endif /* end #define GATEFAKEPHYSICSLIST_HH */


//-----------------------------------------------------------------------------
// EOF
//-----------------------------------------------------------------------------
