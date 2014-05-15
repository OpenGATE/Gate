/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEVOLUMEMANAGER_HH
#define GATEVOLUMEMANAGER_HH 1

#include "G4String.hh"
#include <vector>
#include <map>

#include "GateVVolume.hh"

class GateVolumeManager
{
   public :
    virtual ~GateVolumeManager(){ singleton_VolumeManager = 0;}	

   static GateVolumeManager *GetInstance()
  {   
    if (singleton_VolumeManager == 0)
    {
      singleton_VolumeManager = new GateVolumeManager;
    }
    
    return singleton_VolumeManager;
  };
   
   typedef GateVVolume *(*maker_volume)(const G4String& itsName, G4bool acceptsChildren, G4int depth);
   std::map<G4String,maker_volume> theListOfVolumePrototypes;
   
   private :
   GateVolumeManager();
   static GateVolumeManager *singleton_VolumeManager;
   

};
   
#endif
