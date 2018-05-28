/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GateKermaFactorHandler :
  \brief
*/

#include "GateKermaFactorHandler.hh"
#include "GateMiscFunctions.hh"

#include <G4PhysicalConstants.hh>
#include <G4UnitsTable.hh>

using namespace CLHEP;

//-----------------------------------------------------------------------------
GateKermaFactorHandler::GateKermaFactorHandler()
{
  m_energy       = 0.;
  m_cubicVolume  = 0.;
  m_distance     = 0.;
  m_kerma_factor = 0.;

  mKFExtrapolation             = false;
  mKFDA                        = false;
  mKermaEquivalentFactor       = false;
  mPhotonKermaEquivalentFactor = false;

  kfTable.clear();
  MuEnTable.clear();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateKermaFactorHandler::SetEnergy(double eEnergy)
{
  m_energy = eEnergy;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateKermaFactorHandler::SetCubicVolume(double eCubicVolume)
{
  m_cubicVolume = eCubicVolume;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateKermaFactorHandler::SetDistance(double eDistance)
{
  m_distance = eDistance;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateKermaFactorHandler::SetMaterial(const G4Material* eMaterial)
{
  GateMessage("Actor", 10, "Material: " << eMaterial->GetName() << Gateendl);
  m_material = eMaterial;

  const G4String name(m_material->GetName());

  if (name == "G4_H" ||
      name == "Hydrogen")
  {
    kfTable = kerma_factor_table_hydrogen;
    MuEnTable = MuEnTableHydrogen;
  }
  else if (name == "G4_TISSUE_SOFT_ICRU-4" ||
           name == "Soft_Tissue_ICRU")
  {
    if (mKermaEquivalentFactor)
      //kfTable = kerma_equivalent_factor_ICRU33_Soft_Tissue; // Sv.m²
      kfTable = kerma_equivalent_factor_ICRU33_Soft_Tissue_ICRP60; // Sv.m²
    else
      kfTable = kerma_factor_ICRU33_Soft_Tissue; // Gy.m²
    if (mPhotonKermaEquivalentFactor)
      MuEnTable = KEF_ICRUSoftTissue; // Sv.m²
    else
      MuEnTable = MuEn_ICRU_Soft_Tissue_NIST; // cm²/g
  }
  else if (name == "G4_MUSCLE_STRIATED_ICRU" ||
           name == "Muscle_Skeletal_ICRP_23")
  {
    if (mKFDA)
      kfTable = kerma_factor_table_muscle_DA; // Gy.m²
    else
      kfTable = kerma_factor_muscle_tableau;

    MuEnTable = MuEnMuscleTable;
  }
  else if (name == "G4_LUNG_ICRP" ||
           name == "Lung_ICRP_23")
  {
    kfTable = kerma_factor_Lung_tableau;
    MuEnTable = MuEnLungTable;
  }
  else if (name == "Griffith_Lung_ICRU")
  {
    kfTable = kerma_factor_Lung_Griffith_tableau;
    MuEnTable = MuEn_ICRU44_Lung_Griffith_Table;
  }
  else if (name == "G4_BONE_CORTICAL_ICRP" ||
           name == "Cortical_Bone_ICRP_23")
  {
    kfTable = kerma_factor_Cortical_Bone_tableau;
    MuEnTable = MuEnCorticalBoneTable;
  }
  else if (name == "Urinary_bladder_empty_Adult_ICRU46")
  {
    kfTable   = kerma_factor_Urinary_bladder_empty_Adult_ICRU46_tableau;
    MuEnTable = MuEn_Urinary_bladder_empty_Adult_ICRU46_tableau;
  }
  else if (name == "Skin_Adult_ICRU46")
  {
    kfTable   = kerma_factor_Skin_Adult_ICRU46_tableau;
    MuEnTable = MuEn_Skin_Adult_ICRU46_tableau;
  }
  else if (name == "Adipose_tissue_Adult_1_ICRU46")
  {
    kfTable   = kerma_factor_Adipose_tissue_Adult_1_ICRU46_tableau;
    MuEnTable = MuEn_Adipose_tissue_Adult_1_ICRU46_tableau;
  }
  else if (name == "Testis_Adult_ICRU46")
  {
    kfTable   = kerma_factor_Testis_Adult_ICRU46_tableau;
    MuEnTable = MuEn_Testis_Adult_ICRU46_tableau;
  }
  else if (name == "Lymph_Adult_ICRU46")
  {
    kfTable   = kerma_factor_Lymph_Adult_ICRU46_ICRU46_tableau;
    MuEnTable = MuEn_Lymph_Adult_ICRU46_ICRU46_tableau;
  }
  else if (name == "Blood_Adult_ICRU46")
  {
    kfTable   = kerma_factor_Blood_Adult_ICRU46_tableau;
    MuEnTable = MuEn_Blood_Adult_ICRU46_tableau;
  }
  else if (name == "Skeleton_cartilage_Adult_ICRU46")
  {
    kfTable   = kerma_factor_Skeleton_cartilage_Adult_ICRU46_tableau;
    MuEnTable = MuEn_Skeleton_cartilage_Adult_ICRU46_tableau;
  }
  else if (name == "Skeleton_spongiosa_Adult_ICRU46")
  {
    kfTable   = kerma_factor_Skeleton_spongiosa_Adult_ICRU46_tableau;
    MuEnTable = MuEn_Skeleton_spongiosa_Adult_ICRU46_tableau;
  }
  else if (name == "GI_track_intestine_Adult_ICRU46")
  {
    kfTable   = kerma_factor_GI_track_intestine_Adult_ICRU46_tableau;
    MuEnTable = MuEn_GI_track_intestine_Adult_ICRU46_tableau;
  }
  else if (name == "Air_ICRU" ||
           name == "G4_AIR"   ||
           name == "Air")
  {
    kfTable = kerma_factor_Air_ICRU_tableau;
    MuEnTable = MuEn_ICRU44_Air_Table;
  }
  else if (name == "G4_Galactic" ||
           name == "Vacuum")
  {
    kfTable = kerma_factor_Vacuum_tableau;
    MuEnTable = MuEnVacuumTable;
  }
  else
  {
    GateWarning("Material " << name << " not supported ! Cannot compute dose for this material." << Gateendl);

    kfTable = kerma_factor_Vacuum_tableau;
    MuEnTable = MuEnVacuumTable;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetPhotonFactor(const G4Material* eMaterial)
{
  GateMessage("Actor", 15, "Material: " << eMaterial->GetName() << Gateendl);

  double factor(0.);
  const double* FractionMass(eMaterial->GetFractionVector());

  for(unsigned int i=0; i<eMaterial->GetNumberOfElements(); i++)
  {
    if(eMaterial->GetElement(i)->GetSymbol() == "H")
    {
	    factor = 6.022e23 * FractionMass[i] / 1.008;
	    factor = 1.602e-6 * 3.32e-25 * factor * 2.2 * 1e-8;
      GateMessage("Actor", 10, "Photon correction factor: " << factor << Gateendl);
      return factor;
    }
  }

  return factor;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetKermaFactor(double eEnergy)
{
  if (kfTable.size() == 0)
  {
    GateError("GateKermaFactorHandler -- GetKermaFactor: Kerma Factor table is empty !" << Gateendl);
    exit(EXIT_FAILURE);
  }

  std::vector<double> energyTable;

  if      (kfTable.size() == energy_tableau.size() && !mKermaEquivalentFactor)
    energyTable = energy_tableau; // MeV
  else if (kfTable.size() == energy_table_DA.size() && !mKermaEquivalentFactor)
    energyTable = energy_table_DA; // MeV
  else if (kfTable.size() == energy_table_KermaEquivalentFactor.size() && mKermaEquivalentFactor)
    energyTable = energy_table_KermaEquivalentFactor; // MeV
  else if (kfTable.size() == energy_table_KermaEquivalentFactor_ICRP60.size() && mKermaEquivalentFactor)
    energyTable = energy_table_KermaEquivalentFactor_ICRP60; // MeV
  else
  {
    GateError("GateKermaFactorHandler -- GetKermaFactor: Cannot find an energy table with a good size !" << Gateendl);
    exit(EXIT_FAILURE);
  }

  // KF EXTRAPOLATION /////////////////////////////////////////////////////////
  const double extrapEnergyThreshold = 0.025 * eV;
  //const double extrapEnergyThreshold = 1. * eV;

  if (mKFExtrapolation && eEnergy <= extrapEnergyThreshold)
  {
    // FINDING TABLE ENTRY > EXTRAPENERGYTHRESHOLD ////////////////////////////
    size_t entry = 0;
    for(size_t i = 0; i < energyTable.size(); i++)
      if (entry == 0 && energyTable[i] * MeV >= extrapEnergyThreshold)
        entry = i;
    ///////////////////////////////////////////////////////////////////////////

    GateMessage("Actor", 10, "[GateKermaFactorHandler::" << __FUNCTION__ << "] First energyTable entry > extrapEnergyThreshold: " << entry << " (energyTable[" << entry << "]: " << energyTable[entry] * MeV / eV << " eV, extrapEnergyThreshold: " << extrapEnergyThreshold / eV << " eV)" << Gateendl);

    //OLD//const double extrapolatedKF = 7.011e-21 * std::pow(eEnergy/MeV, -0.466);
    //OLD//const double extrapolatedKF = kfTable[0] * sqrt(energyTable[0] / (eEnergy / MeV));
    const double extrapolatedKF = kfTable[entry] * sqrt(energyTable[entry] / (eEnergy / MeV));

    GateMessage("Actor", 10, "[GateKermaFactorHandler::" << __FUNCTION__ << "] ===> Doing Kerma Factor Extrapolation ! (Energy: " << eEnergy / eV << " eV, ExtrapolatedKF: " << extrapolatedKF << ")" << Gateendl);

    return extrapolatedKF;
  }
  /////////////////////////////////////////////////////////////////////////////
  else if (eEnergy/MeV < energyTable[0]) // && eEnergy/MeV >= 1e-10)
  {
    GateMessage("Actor", 10, "[GateKermaFactorHandler::" << __FUNCTION__ << "] Neutron energy (" << eEnergy/MeV << " MeV) is inferior to minimum energy of kfTable (" << energyTable[0] << " MeV) !" << Gateendl);
    GateMessage("Actor", 10, "[GateKermaFactorHandler::" << __FUNCTION__ << "] ===> Returning " << energyTable[0] << " MeV Kerma Factor ! (" << kfTable[0] << ")" << Gateendl);

    return kfTable[0];
  }

  for (size_t i=1; i<energyTable.size(); i++)
    if (eEnergy/MeV >= energyTable[i-1] && eEnergy/MeV < energyTable[i])
    {
      const double s_diff_energy = (eEnergy/MeV) - (energyTable[i-1]);
      const double b_diff_energy = energyTable[i] - energyTable[i-1];
      const double diff_kerma_factor = kfTable[i] - kfTable[i-1];

      return ((s_diff_energy * diff_kerma_factor) / b_diff_energy) + kfTable[i-1];
    }

  return 0.;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetFlux()
{
  return m_distance / m_cubicVolume / m * m3;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetDose()
{
  return GetKermaFactor(m_energy) * m_distance / m_cubicVolume / m * m3;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetDoseCorrected()
{
  if(m_energy <= 0.025*eV)
    return (GetKermaFactor(m_energy) + GetPhotonFactor(m_material)) * m_distance / m_cubicVolume /m*m3;

  return GetKermaFactor(m_energy) * m_distance / m_cubicVolume /m*m3;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetDoseCorrectedTLE()
{
  double dose = m_energy * GetMuEnOverRho() * m_distance / m_cubicVolume / gray;

  if (mPhotonKermaEquivalentFactor)
    dose = GetMuEnOverRho() * m_distance / m_cubicVolume / m * m3;

  GateMessage("Actor", 10, "GetDoseCorrectedTLE dose: " << dose << " Gy" << Gateendl);
  return dose;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetMuEnOverRho()
{
  if (MuEnTable.size() == 0)
    GateError("MuEn table is empty !" << Gateendl);

  std::vector<double> enTableTLE = energyTableTLE;

  double unitCoef = cm2 / g;

  if (MuEnTable.size() == energyTableTLE.size())
  {
    unitCoef = cm2 / g;
    enTableTLE = energyTableTLE;
  }
  else if (MuEnTable.size() == energyTableTLE_ICRU44.size())
  {
    unitCoef = m2 / kg;
    enTableTLE = energyTableTLE_ICRU44;
  }
  else if (MuEnTable.size() == energyTableTLE_NIST.size())
  {
    unitCoef = cm2 / g;
    enTableTLE = energyTableTLE_NIST;
  }
  else if (MuEnTable.size() == energyTableKEF.size())
  {
    unitCoef = 1.; // Sv.m²
    enTableTLE = energyTableKEF; //MeV
  }
  else
  {
    GateError("GateKermaFactorHandler -- GetMuEnOverRho: Cannot find an energy table with a good size !" << Gateendl);
    exit(EXIT_FAILURE);
  }

  if (m_energy/MeV < enTableTLE[0])
  {
    if (mPhotonKermaEquivalentFactor)
      return MuEnTable[0];
    else
      return 0.;
  }

  for (size_t i=1; i<enTableTLE.size(); i++)
    if (m_energy/MeV >= enTableTLE[i-1] &&
        m_energy/MeV <  enTableTLE[i])
    {
      const double s_diff_energy = (m_energy/MeV) - (enTableTLE[i-1]);
      const double b_diff_energy = enTableTLE[i] - enTableTLE[i-1];
      const double diff_MuEn = MuEnTable[i] - MuEnTable[i-1];

      GateMessage("Actor", 5,  "GateKermaFactorHandler -- GetMuEnOverRho:" << Gateendl
           << " Material name   = " << m_material->GetName() << Gateendl
           << " Photon energy   = " << G4BestUnit(m_energy, "Energy") << Gateendl
           << " enTableTLE[i-1] = " << G4BestUnit(enTableTLE[i-1], "Energy") << Gateendl
           << " enTableTLE[i]   = " << G4BestUnit(enTableTLE[i], "Energy") << Gateendl
           << " enTableTLE size = " << enTableTLE.size() << Gateendl);

      return (((s_diff_energy * diff_MuEn) / b_diff_energy) + MuEnTable[i-1]) * unitCoef;
    }

  return 0.;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TGraph* GateKermaFactorHandler::GetKermaFactorGraph()
{
  GateMessage("Actor", 5, "GateKermaFactorHandler -- Begin of GetKermaFactorGraph\n");
  TGraph* g = new TGraph();
  g->SetTitle((m_material->GetName()+";Neutron energy [MeV];Kerma factor [Gy*m^{2}/neutron]").c_str());
  g->SetMarkerStyle(kFullCircle);
  unsigned n(0);
  for (double i= 1e-10; i <= 2.9e1; i=i*10)
  {
    GateMessage("Actor", 5, "i = " << i << "\n");
    g->SetPoint(n,i,GetKermaFactor(i*MeV));
    n++;
  }
  GateMessage("Actor", 5, "GateKermaFactorHandler -- End of GetKermaFactorGraph\n");
  return g;
}
//-----------------------------------------------------------------------------
