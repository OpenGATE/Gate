/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATECPUPARTICLE_H
#define GATECPUPARTICLE_H

struct GateCPUParticle {
    float          *E;  // MeV
    float          *dx; // direction (unary)
    float          *dy; // direction (unary)
    float          *dz; // direction (unary)
    float          *px; // mm position
    float          *py; // mm position
    float          *pz; // mm position
    float          *t;  // ns time
    int            *parentID; // parent ID of the particle
    int            *eventID; // event ID of the particle
    int            *trackID; // track ID of the particle
    int            *type; // gamma = 0 ; e- = 1
    float          *hole;
    unsigned int    size;
};

GateCPUParticle* GateCPUParticle_new( int bufferParticleEntry );
void GateCPUParticle_delete( GateCPUParticle *input );
void GateCPUParticle_Print( GateCPUParticle const* p, int id );

#endif
