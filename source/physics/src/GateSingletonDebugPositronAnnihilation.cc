/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateSingletonDebugPositronAnnihilation.hh"
#include "GateSingletonDebugPositronAnnihilationMessenger.hh"

GateSingletonDebugPositronAnnihilation *GateSingletonDebugPositronAnnihilation::fInstance = nullptr;

GateSingletonDebugPositronAnnihilation *GateSingletonDebugPositronAnnihilation::GetInstance() {
    if (fInstance == nullptr) fInstance = new GateSingletonDebugPositronAnnihilation();
    return fInstance;
}

GateSingletonDebugPositronAnnihilation::GateSingletonDebugPositronAnnihilation() {
    mMessenger = new GateSingletonDebugPositronAnnihilationMessenger();
    mDebugFlag = false;
    mOutputFile = "output/data.bin";
}

bool GateSingletonDebugPositronAnnihilation::GetDebugFlag(){
    return mDebugFlag;
}

std::string GateSingletonDebugPositronAnnihilation::GetOutputFile(){
    return mOutputFile;
}

