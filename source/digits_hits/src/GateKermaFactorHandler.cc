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
{}
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
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetPhotonFactor(const G4Material* eMaterial)
{
  GateMessage("Actor", 10, "Material: " << eMaterial->GetName() << Gateendl);

  double factor(0.);
  const double* FractionMass(eMaterial->GetFractionVector());

  for(unsigned int i=0; i<eMaterial->GetNumberOfElements(); i++)
  {
    if(eMaterial->GetElement(i)->GetSymbol() == "H")
    {
	    factor = 6.022e23 * FractionMass[i] / 1.008;
	    factor = 1.602e-6 * 3.32e-25 * factor * 2.2 * 1e-8;
      GateMessage("Actor", 9, "Photon correction factor: " << factor << Gateendl);
      return factor;
    }
  }

  return factor;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetKermaFactor(double eEnergy)
{
  if (eEnergy/MeV < energy_tableau[0] && eEnergy/MeV >= 1e-10)
    return 7.011e-21 * std::pow(eEnergy/MeV, -0.466);

  for (size_t i=1; i<energy_tableau.size(); i++)
    if (eEnergy/MeV >= energy_tableau[i-1] && eEnergy/MeV < energy_tableau[i])
    {
      const double s_diff_energy = (eEnergy/MeV) - (energy_tableau[i-1]);
      const double b_diff_energy = energy_tableau[i] - energy_tableau[i-1];
      const double diff_kerma_factor = kerma_factor_muscle_tableau[i] - kerma_factor_muscle_tableau[i-1];

      return ((s_diff_energy * diff_kerma_factor) / b_diff_energy) + kerma_factor_muscle_tableau[i-1];
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
