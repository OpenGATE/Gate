/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateGPUCollimIO.hh"
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
GateGPUCollimIO_Input* GateGPUCollimIO_Input_new()
{
  DD("GateGPUCollimIO_Input_new");
  GateGPUCollimIO_Input* input = new GateGPUCollimIO_Input;
  input->particles.clear();
  input->HexaRadius = 0.0;
  input->HexaHeight = 0.0;
  input->CubRepNumY = 0;
  input->CubRepNumZ = 0;
  input->CubRepVecX = 0.0;
  input->CubRepVecY = 0.0;
  input->CubRepVecZ = 0.0;
  input->LinRepVecX = 0.0;
  input->LinRepVecY = 0.0;
  input->LinRepVecZ = 0.0;

  //input->collim_material_data.clear();
  /*input->phantom_material_data.clear();
    input->phantom_size_x = -1;
    input->phantom_size_y = -1;
    input->phantom_size_z = -1;
    input->phantom_spacing_x = -1*mm/mm;
    input->phantom_spacing_y = -1*mm/mm;
    input->phantom_spacing_z = -1*mm/mm;*/
  input->seed = 0;
  input->firstInitialID = 0;
  input->cudaDeviceID = 0;
  input->nb_events = 10000;
  return input;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUCollimIO_Input_delete(GateGPUCollimIO_Input * input)
{
  DD("GateGPUCollimIO_Input_delete");
  if (input != 0) {
    input->particles.clear();
    //input->collim_material_data.clear();
    //input->phantom_material_data.clear();
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUCollimIO_Input_Init_Materials(GateGPUCollimIO_Input * input,
                                          std::vector<G4Material*> & m,
                                          G4String & name)
{
  DD("GateGPUCollimIO_Input_Init_Materials");

  // std::vector<G4Material*> m;
  // v->BuildLabelToG4MaterialVector(m);
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
    else  input->mat_index[i] = input->mat_nb_elements[i-1]+input->mat_index[i-1];
    //GateGPUCollimIO_Input_Print_mat(input, i);
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
      DD(input->mat_mixture[p]);
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
  input->rad_length = new float[n];
  input->fX0 = new float[n];
  input->fX1 = new float[n];
  input->fD0 = new float[n];
  input->fC = new float[n];
  input->fA = new float[n];
  input->fM = new float[n];

  GatePhysicsList * pl = GatePhysicsList::GetInstance();
  // std::string name = v->GetObjectName();
  //DD(name);
  G4EmCalculator * calculator = new G4EmCalculator;
  for(unsigned int i=0; i<n; i++) {
    //DD(m[i]->GetName());
    //DD(name);
    double range = pl->GetMapOfRegionCuts()[name].electronCut;
    //DD(range/mm);
    input->electron_cut_energy[i] =
      calculator->ComputeEnergyCutFromRangeCut(range, G4Electron::Electron(), m[i]);
    //DD(input->electron_cut_energy[i]/keV);
    input->electron_max_energy[i] = FLT_MAX;
    //DD(input->electron_max_energy[i]);
    input->electron_mean_excitation_energy[i] = m[i]->GetIonisation()->GetMeanExcitationEnergy();
    //DD(input->electron_mean_excitation_energy[i]);

    input->rad_length[i] = m[i]->GetRadlen();
    //DD(m[i]->GetRadlen());

    input->fX0[i] = m[i]->GetIonisation()->GetX0density();
    input->fX1[i] = m[i]->GetIonisation()->GetX1density();
    input->fD0[i] = m[i]->GetIonisation()->GetD0density();
    input->fC[i] = m[i]->GetIonisation()->GetCdensity();
    input->fA[i] = m[i]->GetIonisation()->GetAdensity();
    input->fM[i] = m[i]->GetIonisation()->GetMdensity();
    DD(input->fX0[i]);
    DD(input->fX1[i]);
    DD(input->fD0[i]);
    DD(input->fC[i]);
    DD(input->fA[i]);
    DD(input->fM[i]);
  }

}
//-----------------------------------------------------------------------------


//----------------------------------------------------------
/*struct ActivityMaterialTuple
  {
  unsigned int index;
  float activity;
  };
  //----------------------------------------------------------


  //----------------------------------------------------------
  struct ActivityMaterialTupleStrictWeakOrdering
  {
  bool operator()(const ActivityMaterialTuple& a, const ActivityMaterialTuple& b)
  {
  return a.activity < b.activity;
  }
  };
  typedef std::vector<ActivityMaterialTuple> ActivityMaterialTuplesVector;
  //----------------------------------------------------------


  //----------------------------------------------------------
  void GateGPUCollimIO_Input_parse_activities(const ActivityMap& activities,
  GateGPUCollimIO_Input * input)
  {
  DD("GateGPUCollimIO_Input_parse_activities");
  assert(input);
  assert(input->activity_data.empty());
  assert(input->activity_index.empty());

  assert(input->phantom_size_x > 0);
  assert(input->phantom_size_y > 0);
  assert(input->phantom_size_z > 0);

  ActivityMaterialTuplesVector tuples;
  double total_activity = 0;
  { // fill tuples structure
  for (ActivityMap::const_iterator iter = activities.begin(); iter != activities.end(); iter++)
  {
  const int ii = iter->first[0];
  const int jj = iter->first[1];
  const int kk = iter->first[2];

  assert(ii >= 0);
  assert(jj >= 0);
  assert(kk >= 0);
  assert(ii < input->phantom_size_x);
  assert(jj < input->phantom_size_y);
  assert(kk < input->phantom_size_z);

  const int index = ii + jj*input->phantom_size_x + kk*input->phantom_size_y*input->phantom_size_x;

  assert(index >= 0);
  assert(index < input->phantom_size_x*input->phantom_size_y*input->phantom_size_z);

  ActivityMaterialTuple tuple;
  tuple.index = index;
  tuple.activity = iter->second;
  tuples.push_back(tuple);
  total_activity += tuple.activity; // in GBq - JB
  }
  }

  input->tot_activity = total_activity;

  { // sort tuples by activities
  std::sort(tuples.begin(),tuples.end(),ActivityMaterialTupleStrictWeakOrdering());
  }

  { // allocate and fill gpu input structure
  double cumulated_activity = 0;
  for (ActivityMaterialTuplesVector::const_iterator iter = tuples.begin(); iter != tuples.end(); iter++)
  {
  cumulated_activity += iter->activity;
  input->activity_data.push_back(cumulated_activity/total_activity);
  input->activity_index.push_back(iter->index);
  }
  }
  }*/
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateGPUCollimIO_Output* GateGPUCollimIO_Output_new()
{
  DD("GateGPUCollimIO_Output_new");
  GateGPUCollimIO_Output* output = new GateGPUCollimIO_Output;
  output->particles.clear();
  return output;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUCollimIO_Output_delete(GateGPUCollimIO_Output * output)
{
  DD("GateGPUCollimIO_Output_delete");
  if (output != 0) {
    output->particles.clear();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateGPUCollimIO_Particle_Print(const GateGPUCollimIO_Particle & p)
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

/*
  #ifndef GATE_USE_GPU
  void GateGPUCollimIOTrack(const GateGPUCollimIO_Input * input,
  GateGPUCollimIO_Output * output)
  {

  // FAKE TRACKING
  GateGPUCollimIO_Input::ParticlesList::const_iterator
  iter = input->particles.begin();
  while (iter != input->particles.end()) {
  GateGPUCollimIO_Particle p = *iter;
  p.E = p.E/2.0;
  // p.px += 30*cm;
  // p.py += 30*cm;
  // p.pz += 30*cm;
  output->particles.push_back(p);
  ++iter;
  }
  //GateError("Gate is compiled without CUDA enabled. You cannot use 'GateGPUCollimIO'.");
  }
  #endif
*/
