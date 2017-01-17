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

  if (name == "G4_MUSCLE_STRIATED_ICRU" ||
      name == "Muscle_Skeletal_ICRP_23")
  {
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
    GateError("Kerma Factor table is empty !" << Gateendl);

  if (eEnergy/MeV < energy_tableau[0] && eEnergy/MeV >= 1e-10)
    return 7.011e-21 * std::pow(eEnergy/MeV, -0.466);

  for (size_t i=1; i<energy_tableau.size(); i++)
    if (eEnergy/MeV >= energy_tableau[i-1] && eEnergy/MeV < energy_tableau[i])
    {
      const double s_diff_energy = (eEnergy/MeV) - (energy_tableau[i-1]);
      const double b_diff_energy = energy_tableau[i] - energy_tableau[i-1];
      const double diff_kerma_factor = kfTable[i] - kfTable[i-1];

      return ((s_diff_energy * diff_kerma_factor) / b_diff_energy) + kfTable[i-1];
    }

  return 0.;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetDose()
{
  return GetKermaFactor(m_energy) * m_distance / m_cubicVolume /m*m3;
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
  const double dose = m_energy * GetMuEnOverRho() * m_distance / m_cubicVolume / gray;
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

  if (MuEnTable.size() == energyTableTLE_ICRU44.size())
    enTableTLE = energyTableTLE_ICRU44;

  if (m_energy/MeV < enTableTLE[0])
    return 0.;

  for (size_t i=1; i<enTableTLE.size(); i++)
    if (m_energy/MeV >= enTableTLE[i-1] &&
        m_energy/MeV <  enTableTLE[i])
    {
      const double s_diff_energy = (m_energy/MeV) - (enTableTLE[i-1]);
      const double b_diff_energy = enTableTLE[i] - enTableTLE[i-1];
      const double diff_MuEn = MuEnTable[i] - MuEnTable[i-1];

      return (((s_diff_energy * diff_MuEn) / b_diff_energy) + MuEnTable[i-1]) * cm2 / g;
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
