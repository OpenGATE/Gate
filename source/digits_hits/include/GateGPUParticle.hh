/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGPUPARTICLE_H
#define GATEGPUPARTICLE_H

struct GateGPUParticle {
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
    unsigned int    size;
    int             cudaDeviceID;
};

GateGPUParticle* GateGPUParticle_new( int bufferParticleEntry,
    int cudaDevice );
void GateGPUParticle_delete( GateGPUParticle *input );
void GateGPUParticle_Print( GateGPUParticle const* p, int id );

#endif
