/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEFAKERUNMANAGER_HH
#define GATEFAKERUNMANAGER_HH



/*
 * \file GateFakeRunManager.hh
 * \brief fGate Fake RunManager class for development
 */

#include "G4RunManager.hh"

class GateFakeRunManager : public G4RunManager {
  
public: 
  // Constructor
  GateFakeRunManager():G4RunManager() {}
  
  // Destructor
  virtual ~GateFakeRunManager() {}
  
}; // end class
  
#endif /* end #define GATEFAKERUNMANAGER_HH */

//-----------------------------------------------------------------------------
// EOF
//-----------------------------------------------------------------------------
