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

#include <G4PhysicalConstants.hh>
#include <G4UnitsTable.hh>

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>

using namespace std;
using namespace CLHEP;

//-----------------------------------------------------------------------------
GateKermaFactorHandler::GateKermaFactorHandler()
{
  m_h = 6.022E23*0.102/1.008;
  //cout << "m_h= " << m_h << endl;
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
  m_material = eMaterial;

  int nb_of_elements(eMaterial->GetNumberOfElements());
  const double* FractionMass(eMaterial->GetFractionVector());

  for(int i=0; i<nb_of_elements; i++)
    if(eMaterial->GetElement(i)->GetZ() == 1 && eMaterial->GetElement(i)->GetA() == 1)
    {
      //cout << "FractionMass= " << FractionMass[i] << endl;
	    m_h = 6.022E23*FractionMass[i]/1.008;
      G4cout << "m_h= " << m_h << G4endl;
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateKermaFactorHandler::GetKermaFactor(double eEnergy)
{
  double kerma_factor = 0;

  if (eEnergy/MeV < energy_tableau[0])
    return kerma_factor_muscle_tableau[0];

  for (size_t i=1; i<energy_tableau.size(); i++)
  {
    if (eEnergy/MeV >= energy_tableau[i-1] && eEnergy/MeV < energy_tableau[i])
    {
      double s_diff_energy = (eEnergy/MeV)-(energy_tableau[i-1]);
      double b_diff_energy = energy_tableau[i]-energy_tableau[i-1];
      double diff_kerma_factor = kerma_factor_muscle_tableau[i]-kerma_factor_muscle_tableau[i-1];

      kerma_factor = ((s_diff_energy * diff_kerma_factor)/b_diff_energy) + kerma_factor_muscle_tableau[i-1];
      return kerma_factor;
    }
  }

return kerma_factor;
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
    return (GetKermaFactor(m_energy)+7.13E-16) * m_distance / m_cubicVolume /m*m3;

  return GetKermaFactor(m_energy) * m_distance / m_cubicVolume /m*m3;
}
//-----------------------------------------------------------------------------
