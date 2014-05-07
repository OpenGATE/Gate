/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateGPUParticle.hh"
#include "GateMessageManager.hh"
#include "GatePhysicsList.hh"
#include "G4UnitsTable.hh"
#include <iostream>
#include <cstring>

GateGPUParticle* GateGPUParticle_new( G4int bufferParticleEntry,
    G4int cudaDevice )
{
    GateGPUParticle *p = new GateGPUParticle;
    // Allocating each buffer
    p->E = new float[ bufferParticleEntry ];
    p->dx = new float[ bufferParticleEntry ];
    p->dy = new float[ bufferParticleEntry ];
    p->dz = new float[ bufferParticleEntry ];
    p->px = new float[ bufferParticleEntry ];
    p->py = new float[ bufferParticleEntry ];
    p->pz = new float[ bufferParticleEntry ];
    p->t = new float[ bufferParticleEntry ];
    p->parentID = new int[ bufferParticleEntry ];
    p->eventID = new int[ bufferParticleEntry ];
    p->trackID = new int[ bufferParticleEntry ];
    p->type = new int[ bufferParticleEntry ];
    p->size = 0;
    p->cudaDeviceID = cudaDevice;
    return p;
}

void GateGPUParticle_delete( GateGPUParticle *p )
{
    if( p )
    {
        delete[] p->E;
        delete[] p->dx;
        delete[] p->dy;
        delete[] p->dz;
        delete[] p->px;
        delete[] p->py;
        delete[] p->pz;
        delete[] p->t;
        delete[] p->parentID;
        delete[] p->eventID;
        delete[] p->trackID;
        delete[] p->type;
        p->size = 0;
        p = NULL;
    }
}

void GateGPUParticle_Print( const GateGPUParticle* p, int id )
{
    if( p->type[ id ] == 0 )
        std::cout << "type = gamma" << std::endl;
    if( p->type[ id ] == 1 )
        std::cout << "type = e-" << std::endl;
    std::cout << "E= " << G4BestUnit( p->E[ id ], "Energy") << std::endl;
    std::cout << "parent id = " << p->parentID[ id ] << std::endl;
    std::cout << "event id = " << p->eventID[ id ]  << std::endl;
    std::cout << "track id = " << p->trackID[ id ]  << std::endl;
    std::cout << "t = " << G4BestUnit(p->t[ id ], "Time")  << std::endl;
    std::cout << "position = " << p->px[ id ] << " " << p->py[ id ] << " " << p->pz[ id ] << " mm" << std::endl;
    std::cout << "dir = " << p->dx[ id ] << " " << p->dy[ id ] << " " << p->dz[ id ] << std::endl;
}
