/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateTrackingGPUActorIO.hh"
#include "GateRandomEngine.hh"
#include "GateMessageManager.hh"
#include "GatePhysicsList.hh"

#include "G4UnitsTable.hh"

#include <iostream>
#include <cassert>
#include <cfloat>
using std::cout;
using std::endl;

//-----------------------------------------------------------------------------
GateTrackingGPUActorInput* GateTrackingGPUActorInput_new()
{
  DD("GateTrackingGPUActorInput_new");
  GateTrackingGPUActorInput* input = new GateTrackingGPUActorInput;
  input->particles.clear();
  input->phantom_material_data.clear();

  input->phantom_size_x = -1;
  input->phantom_size_y = -1;
  input->phantom_size_z = -1;
  input->phantom_spacing_x = -1*mm/mm;
  input->phantom_spacing_y = -1*mm/mm;
  input->phantom_spacing_z = -1*mm/mm;

  input->cudaDeviceID = 0;

  return input;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActorInput_delete(GateTrackingGPUActorInput*input)
{
  DD("GateTrackingGPUActorInput_delete");
  if (input != 0) {
    input->particles.clear();
    input->phantom_material_data.clear();
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActorInput_Init_Materials(GateTrackingGPUActorInput * input, 
                                              GateVImageVolume * v)
{
  DD("GateTrackingGPUActorInput_Init_Materials");
  
  std::vector<G4Material*> m;
  v->BuildLabelToG4MaterialVector(m);
  DD(m.size());
  unsigned int n = m.size();
  input->nb_materials = n;

  // Number of elements per material
  // Index to access material mixture 
  input->mat_nb_elements = new unsigned short int[n];
  input->mat_index = new unsigned short int[n];
  int k=0;
  for(unsigned int i=0; i<n; i++) {
    //DD(m[i]->GetName());
    input->mat_nb_elements[i] = m[i]->GetNumberOfElements();
    if (i == 0) input->mat_index[i] = 0;
    else  input->mat_index[i] = input->mat_nb_elements[i]+input->mat_index[i-1];
    //GateTrackingGPUActorInput_Print_mat(input, i);
    //DD(input->mat_nb_elements[i]);
    //DD(input->mat_index[i]);
    k += input->mat_nb_elements[i];
  }

  // Mixture of each material
  // Atomic number density of each element of materials (Avo*density*MassFraction / Ai)
  //DD(k);
  input->nb_elements_total = k;
  input->mat_mixture = new unsigned short int[k];
  input->mat_atom_num_dens = new float[k];
  int p = 0;
  double avo = CLHEP::Avogadro;
  for(unsigned int i=0; i<n; i++) {
    //DD(m[i]->GetName());
    for(unsigned int e=0; e<input->mat_nb_elements[i]; e++) {
      input->mat_mixture[p] = m[i]->GetElement(e)->GetZ();
      //DD(input->mat_mixture[p]);
      input->mat_atom_num_dens[p] = avo * m[i]->GetDensity() * m[i]->GetFractionVector()[e] /m[i]->GetElement(e)->GetA();
      //DD(input->mat_atom_num_dens[p]);
      p++;
    }
  }  
  //DD(p);

  // Total number of atoms per volume (sum{mat_atom_num_dens_i})
  // Total number of electrons per volume (sum{mat_atom_num_dens_i*Z_i})
  input->mat_nb_atoms_per_vol = new float[n];
  input->mat_nb_electrons_per_vol = new float[n];
  p=0;
  for(unsigned int i=0; i<n; i++) {
    //DD(m[i]->GetName());
    input->mat_nb_atoms_per_vol[i] = 0;
    input->mat_nb_electrons_per_vol[i] = 0;
    for(unsigned int e=0; e<input->mat_nb_elements[i]; e++) {
      input->mat_nb_atoms_per_vol[i] += input->mat_atom_num_dens[p];
      input->mat_nb_electrons_per_vol[i] += input->mat_atom_num_dens[p] * m[i]->GetElement(e)->GetZ();
      p++;
    }
    //DD(input->mat_nb_atoms_per_vol[i]);
    //DD(input->mat_nb_electrons_per_vol[i]);
  }

  // Electron energy threshold (eIonisation)
  input->electron_cut_energy = new float[n];
  input->electron_max_energy = new float[n];
  input->electron_mean_excitation_energy = new float[n];
  input->fX0 = new float[n];
  input->fX1 = new float[n];
  input->fD0 = new float[n];
  input->fC = new float[n];
  input->fA = new float[n];
  input->fM = new float[n];

  GatePhysicsList * pl = GatePhysicsList::GetInstance();
  std::string name = v->GetObjectName();
  //DD(name);
  G4EmCalculator * calculator = new G4EmCalculator;
  for(unsigned int i=0; i<n; i++) {
    //DD(m[i]->GetName());
    double range = pl->GetMapOfRegionCuts()[name].electronCut;
    //DD(range/mm);
    input->electron_cut_energy[i] =
      calculator->ComputeEnergyCutFromRangeCut(range, G4Electron::Electron(), m[i]);
    //DD(input->electron_cut_energy[i]/keV);
    input->electron_max_energy[i] = FLT_MAX;
    //DD(input->electron_max_energy[i]);
    input->electron_mean_excitation_energy[i] = m[i]->GetIonisation()->GetMeanExcitationEnergy();
    //DD(input->electron_mean_excitation_energy[i]);

    input->fX0[i] = m[i]->GetIonisation()->GetX0density();
    input->fX1[i] = m[i]->GetIonisation()->GetX1density();
    input->fD0[i] = m[i]->GetIonisation()->GetD0density();
    input->fC[i] = m[i]->GetIonisation()->GetCdensity();
    input->fA[i] = m[i]->GetIonisation()->GetAdensity();
    input->fM[i] = m[i]->GetIonisation()->GetMdensity();
    /*DD(input->fX0[i]);
    DD(input->fX1[i]);
    DD(input->fD0[i]);
    DD(input->fC[i]);
    DD(input->fA[i]);
    DD(input->fM[i]);*/
  }
    
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateTrackingGPUActorOutput* GateTrackingGPUActorOutput_new()
{
  DD("GateTrackingGPUActorOutput_new");
  GateTrackingGPUActorOutput* output = new GateTrackingGPUActorOutput;
  output->particles.clear();
  return output;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActorOutput_delete(GateTrackingGPUActorOutput*output)
{
  DD("GateTrackingGPUActorOutput_delete");
  if (output != 0) {
    output->particles.clear();
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActorParticle_Print(const GateTrackingGPUActorParticle & p)
{
  if (p.type == 0) std::cout << "type = gamma" << std::endl;
  if (p.type == 1) std::cout << "type = e-" << std::endl;
  std::cout << "E = " << G4BestUnit(p.E, "Energy") << std::endl;
  std::cout << "event id = " << p.eventID  << std::endl;
  std::cout << "track id = " << p.trackID  << std::endl;
  std::cout << "t = " << G4BestUnit(p.t, "Time")  << std::endl;
  std::cout << "position = " << p.px << " " << p.py << " " << p.pz << " mm" << std::endl;
  std::cout << "dir = " << p.dx << " " << p.dy << " " << p.dz << std::endl;
}
//-----------------------------------------------------------------------------


#ifndef GATE_USE_CUDA
void GateTrackingGPUActorTrack(const GateTrackingGPUActorInput * input, 
                               GateTrackingGPUActorOutput * output)
{

  // FAKE TRACKING
  GateTrackingGPUActorInput::ParticlesList::const_iterator 
      iter = input->particles.begin();
    while (iter != input->particles.end()) {
      GateTrackingGPUActorParticle p = *iter;
      p.E = p.E/2.0;
      // p.px += 30*cm;
      // p.py += 30*cm;
      // p.pz += 30*cm;
      output->particles.push_back(p);
      ++iter;
    }
    //GateError("Gate is compiled without CUDA enabled. You cannot use 'GateTrackingGPUActor'.");
}
#endif



