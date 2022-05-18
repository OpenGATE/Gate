/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATESINGLETONDEBUGPOSITRONANNIHILATION_HH
#define GATESINGLETONDEBUGPOSITRONANNIHILATION_HH

#include "GateSingletonDebugPositronAnnihilationMessenger.hh"

class GateSingletonDebugPositronAnnihilation
{
public:
  GateSingletonDebugPositronAnnihilation();
  ~GateSingletonDebugPositronAnnihilation() { }
	
  static GateSingletonDebugPositronAnnihilation* GetInstance();

  bool GetDebugFlag();
  void SetDebugFlag(const bool aFlag) { mDebugFlag = aFlag;};

private:
  bool mDebugFlag;
  static GateSingletonDebugPositronAnnihilation *fInstance;
  GateSingletonDebugPositronAnnihilationMessenger* mMessenger;

};

#endif
