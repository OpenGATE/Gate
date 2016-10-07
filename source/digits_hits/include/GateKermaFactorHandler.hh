/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*!
  \class  GateKermaFactorHandler
  \authors: Halima Elazhar (halima.elazhar@ihpc.cnrs.fr)
            Thomas Deschler (thomas.deschler@iphc.cnrs.fr)
*/

#ifndef GATEKERMAFACTORHANDLER_HH
#define GATEKERMAFACTORHANDLER_HH

#include "GateKermaFactorDatabase.hh"

#include <G4Material.hh>

class GateKermaFactorHandler
{
  public:
    GateKermaFactorHandler();
    ~GateKermaFactorHandler() {};

    void SetEnergy(double);
    void SetCubicVolume(double);
    void SetDistance(double);
    void SetMaterial(const G4Material*);

    double GetDoseCorrected();
    double GetDose();

  private:
    double GetKermaFactor(double);

    double m_energy;
    double m_cubicVolume;
    double m_distance;
    double m_kerma_factor;
    double m_h;

    const G4Material* m_material;
};
#endif
