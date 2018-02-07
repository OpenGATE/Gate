/*----------------------
 Copyright (C): OpenGATE Collaboration

 This software is distributed under the terms
 of the GNU Lesser General  Public Licence (LGPL)
 See LICENSE.md for further details
 ------*/

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT

#include "G4SystemOfUnits.hh"

#include "GateARFTable.hh"
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <utility>
#include "TROOT.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH2D.h"
#include "TH1D.h"
#include "TMath.h"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

GateARFTable::GateARFTable(const G4String & aName) :
    _ArfTableName(aName)
  {
  _ArfTableIndex = 0; /* the index of the AERF Table because many are defined for one simulation */
  _ArfTableVector = 0; /* contains the probability for each cos(theta)_k, tan(phi)_k' as a single linearized index */
  _DrfTableVector = 0;
  _NumberOfCosTheta = 2048;
  _DrfTableDimensionX = 1280;
  _DrfTableDimensionY = 1280;
  _DrfBinSize = 0.0442 * cm;
  _NumberOfTanPhi = 512; /* the number of discretized values of tan(phi) */
  _TotalNumberbOfThetaPhi = _NumberOfCosTheta * _NumberOfTanPhi;
  mBinnedPhotonCounter = 0;
  mPhiCounts = 0;
  mStep1 = 0.010 / (_NumberOfCosTheta * 0.5);
  mStep2 = 0.040 / _NumberOfTanPhi;
  mStep3 = 0.20 / (_NumberOfTanPhi * 0.5);
  mStep4 = 0.55 / (_NumberOfTanPhi * 0.5);
  mTanPhiStep = 1.0 / (_NumberOfTanPhi * 0.5);
  mBase1 = 1. + mStep1 * 0.5;
  mBase2 = 0.990005; /* 1024 */
  mBase3 = 0.950005; /* 1536 */
  mBase4 = 0.750005; /* 1792 */
  /*
   variables for energy resolution
   If mEnergyResolution>0, used for energy resolution ~ square_root (photon energy)
   static double dEWindowWidth, mEnergyResolution = 10.0, dResoAtErg = 140.5;
   if mEnergyResolution<0, energy resolution linearly depends on photon energy
   for max E > 240 keV: Siemens E.CAM, dConstant=0.971162, dLinear=0.0555509
   for max E < 240 keV: Siemens E.CAM, dConstant=1.9335, dLinear=0.0383678
   OLD(for Siemens E.CAM, dConstant=-0.226416, dLinear=0.0592695)
   */
  mEnergyResolution = 0.095; /* it is a percentage */
  mEnergyReference = 140.5 * keV;
  mConstantTerm = 1.9335 * keV;
  mLinearTerm = 0.0383678;
  SetNoPrimary();
  _EnergyLow = 0.0; /* the left end energy of the energy window */
  _EnergyHigh = 0.0;
  _EnergyLowUser = 0.0; /* window energy specified by the use */
  _EnergyHighUser = 0.0;
  _ThetaVector = 0;
  _CosThetaVector = 0;
  _TanPhiVector = 0;
  mPhi = 0;
  _AverageNumberOfPixels = 5;
  _DistanceSourceToImage = 36.05 * cm; /* distance from source to the detector projection plane which is the middle of the detector's depth */
  }

GateARFTable::~GateARFTable()
  {
  if (_ArfTableVector != 0)
    {
    delete[] _ArfTableVector;
    }
  if (_CosThetaVector != 0)
    {
    delete[] _CosThetaVector;
    }
  if (_TanPhiVector != 0)
    {
    delete[] _TanPhiVector;
    }
  if (mPhi != 0)
    {
    delete[] mPhi;
    }
  }

void GateARFTable::InitializeCosTheta()
  {
  G4double tmp = 1.0 + mStep1;
  for (G4int cosThetaI = 0; cosThetaI < _NumberOfCosTheta; cosThetaI++)
    {
    /* cosTheta 1.0 --> 0.20 */
    if (cosThetaI <= G4int(_NumberOfCosTheta * 0.5))
      {
      tmp -= mStep1;
      _CosThetaVector[cosThetaI] = tmp;
      _ThetaVector[cosThetaI] = acos(tmp);
      }
    if ((cosThetaI > G4int(_NumberOfCosTheta * 0.5)) && (cosThetaI < 1537))
      {
      tmp -= mStep2;
      _CosThetaVector[cosThetaI] = tmp;
      _ThetaVector[cosThetaI] = acos(tmp);
      }
    if ((cosThetaI > 1536) && (cosThetaI < 1793))
      {
      tmp -= mStep3;
      _CosThetaVector[cosThetaI] = tmp;
      _ThetaVector[cosThetaI] = acos(tmp);
      }
    if (cosThetaI > 1792)
      {
      tmp -= mStep4;
      _CosThetaVector[cosThetaI] = tmp;
      _ThetaVector[cosThetaI] = acos(tmp);
      }
    _CosThetaIVector[2047 - cosThetaI] = _CosThetaVector[cosThetaI];
    }
  }

void GateARFTable::InitializePhi()
  {
  if (_TanPhiVector != 0)
    {
    delete[] _TanPhiVector;
    }
  _TanPhiVector = new G4double[_NumberOfTanPhi];
  if (mPhi != 0)
    {
    delete[] mPhi;
    }
  mPhi = new G4double[_NumberOfTanPhi];
  G4double tmp = -mTanPhiStep;
  G4double temporaryPhi = 1.;
  for (G4int phi = 0; phi < _NumberOfTanPhi; phi++)
    { /*  Phi, 0 --> 180 */
    if (phi < G4int(_NumberOfTanPhi * 0.5))
      tmp += mTanPhiStep;
    if (phi == G4int(_NumberOfTanPhi * 0.5))
      {
      tmp = 1.0;
      }
    if (phi > G4int(_NumberOfTanPhi * 0.5))
      {
      tmp -= mTanPhiStep;
      temporaryPhi += mTanPhiStep;
      }
    _TanPhiVector[phi] = tmp;
    if (phi <= G4int(_NumberOfTanPhi * 0.5))
      {
      mPhi[phi] = atan(_TanPhiVector[phi]);
      }
    else
      {
      mPhi[phi] = atan(temporaryPhi);
      }
    }

  }

void GateARFTable::Initialize(const G4double & energyLow, const G4double & energyHigh)
  {
  _EnergyLowUser = energyLow; /* window energy specified by the user */
  _EnergyHighUser = energyHigh;
  if (_CosThetaVector != 0)
    {
    delete[] _CosThetaVector;
    }
  _CosThetaVector = new G4double[_NumberOfCosTheta];
  _CosThetaIVector = new G4double[_NumberOfCosTheta];
  _ThetaVector = new G4double[_NumberOfCosTheta];
  G4cout.precision(10);

  InitializeCosTheta();
  InitializePhi();

  if (_ArfTableVector != 0)
    {
    delete[] _ArfTableVector;
    }
  _ArfTableVector = new G4double[_TotalNumberbOfThetaPhi];
  _DrfTableVector = new G4double[_DrfTableDimensionX * _DrfTableDimensionY];
  _LowX = -.5 * G4double(_DrfTableDimensionX) * _DrfBinSize;
  _LowY = -.5 * G4double(_DrfTableDimensionY) * _DrfBinSize;
  for (G4int i = 0; i < _TotalNumberbOfThetaPhi; i++)
    {
    _ArfTableVector[i] = 0.;
    }
  for (G4int i = 0; i < _DrfTableDimensionX * _DrfTableDimensionY; i++)
    {
    _DrfTableVector[i] = 0.;
    }
  G4cout << " Initialized ARF Table " << GetName() << Gateendl;
  }

G4double GateARFTable::RetrieveProbability(const G4double & x, const G4double & y)
  {
  G4int theta = 0;
  G4int phi = 0;
  if (GetIndexes(x, y, theta, phi) == 1)
    {
    return _ArfTableVector[theta + phi * _NumberOfCosTheta];
    }
  return 0.;
  }

void GateARFTable::SetEnergyReso(const G4double & aE)
  {
  mEnergyResolution = aE;
  }

void GateARFTable::SetERef(const G4double & aE)
  {
  mEnergyReference = aE;
  }

G4int GateARFTable::GetIndexes(const G4double & x, const G4double & y, G4int& theta, G4int& phi)
  {
  G4double cosTheta = sqrt(1. - x * x - y * y);
  if (cosTheta < 0.2)
    {
    return 0;
    }
  if (cosTheta - 0.99 > 0.)
    {
    theta = G4int((1.0 - cosTheta) * (_NumberOfCosTheta * 0.5) / 0.010);
    }
  else if (cosTheta - 0.95 > 0.)
    {
    theta = G4int((0.99 - cosTheta) * _NumberOfTanPhi / 0.040) + G4int(_NumberOfCosTheta * 0.5);
    }
  else if (cosTheta - 0.75 > 0.)
    {
    theta = G4int((0.95 - cosTheta) * (_NumberOfTanPhi * 0.5) / 0.20) + 1536;
    }
  else
    {
    theta = G4int((0.75 - cosTheta) * (_NumberOfTanPhi * 0.5) / 0.55) + 1792;
    }

  if (theta > 0 && (_CosThetaVector[theta] - cosTheta < 0.))
    {
    theta--;
    }
  if (theta < 2047 && (_CosThetaVector[theta + 1] - cosTheta > 0.))
    {
    theta++;
    }
  if (theta == 0)
    {
    phi = 0;
    return 1;
    }
  if (fabs(x) <= 1.e-8)
    {
    phi = 511;
    return 1;
    }
  G4double tanPhi = y / x;
  if (tanPhi < 0.0)
    {
    tanPhi *= -1.0;
    }

  if (tanPhi - 1.00 < 0.)
    {
    phi = G4int(tanPhi / mTanPhiStep + 0.5);
    if (tanPhi - _TanPhiVector[phi] < 0.)
      {
      phi--;
      }
    }
  else
    {
    tanPhi = fabs(x / y);
    G4int tanPhiIndex = G4int(tanPhi / mTanPhiStep + 0.5);
    if (tanPhi - _TanPhiVector[tanPhiIndex] < 0.)
      {
      tanPhiIndex--;
      }
    phi = 511 - tanPhiIndex;
    }
  return 1;
  }

G4double GateARFTable::computeARFfromDRF(const G4double & xI,
                                         const G4double & yJ,
                                         const G4double & cosTheta)
  {
  G4int index0 = 0;
  G4int i = 0;
  G4int j = 0;
  G4double sum = 0.0;
  G4double smallPixX = 0;
  G4double smallPixY = 0;
  G4double lengthSqr = 0;
  G4double refRadius = ((G4double) _AverageNumberOfPixels + 0.5) * _DrfBinSize;
  G4double refRadiusSqr = refRadius * refRadius;
  G4int numSmallPixSummed = 0;
  G4int centerI = (G4int) (_DrfTableDimensionX / 2);
  G4int centerJ = (G4int) (_DrfTableDimensionY / 2);
  i = centerI + G4int(xI / _DrfBinSize + 0.5); /* for 1023x1023. Center is (511,511) */
  j = centerJ + G4int(yJ / _DrfBinSize + 0.5);
  /* get totally (2iAvgPixNum+1 x 2iAvgPixNum+1) matrix */
  for (G4int n = j - _AverageNumberOfPixels; n < j + _AverageNumberOfPixels + 1; n++)
    {
    for (G4int m = i - _AverageNumberOfPixels; m < i + _AverageNumberOfPixels + 1; m++)
      {
      index0 = m + n * _DrfTableDimensionX;
      /* loops k and l are used to check the 10x10 small elements in each big element (m, n). All elements with the distance
       to the photon point (dTmpi, dTmpj) smaller than the user specified radius are used to do average */
      for (G4int k = 0; k < 10; k++)
        {
        smallPixX = (G4double(m - centerI) + (G4double(k) - 4.5) / 10.0) * _DrfBinSize;
        for (G4int l = 0; l < 10; l++)
          {
          /* dSmallPixX and dSmallPixY are the absolute values for that the small element in unit of cm */
          smallPixY = (G4double(n - centerJ) + (G4double(l) - 4.5) / 10.0) * _DrfBinSize;
          /* distance in unit of cm */
          lengthSqr = (smallPixX - xI) * (smallPixX - xI) + (smallPixY - yJ) * (smallPixY - yJ);
          if (lengthSqr < refRadiusSqr)
            {
            sum = sum + _DrfTableVector[index0];
            numSmallPixSummed++;
            }
          } /* end of loop l */
        } /* end of loop k */
      } /* end of loop m */
    } /* end of loop n */

  //sum /= 100.0;
  //G4double size = mDrfBinSize * mDrfBinSize * G4double(numSmallPixSummed) / 100.0;
  G4double size = _DrfBinSize * _DrfBinSize * G4double(numSmallPixSummed);
  G4double factor = _DistanceSourceToImage * _DistanceSourceToImage
                    / (size * cosTheta * cosTheta * cosTheta);
  factor *= (4.0 * pi / mTotalNumberOfPhotons);
  sum *= factor;
  return sum;

  }

G4int GateARFTable::GetOneDimensionIndex(G4int x, G4int y)
  {
  return x + y * _DrfTableDimensionX;
  }

void GateARFTable::convertDRF2ARF()
  {

  G4int index1 = 0;
  G4int index2 = 0;
  G4int index3 = 0;
  G4int index4 = 0;
  /* prepare the DRF table before processing */
  for (G4int j = 0; j < _DrfTableDimensionY; j++)
    {
    for (G4int i = 0; i < _DrfTableDimensionX; i++)
      {
      index1 = GetOneDimensionIndex(i, j);
      index2 = GetOneDimensionIndex(_DrfTableDimensionX - i - 1, j);
      index3 = GetOneDimensionIndex(_DrfTableDimensionX - i - 1, _DrfTableDimensionY - j - 1);
      index4 = GetOneDimensionIndex(i, _DrfTableDimensionY - j - 1);
      _DrfTableVector[index1] = 0.25
                                * (_DrfTableVector[index1]
                                   + _DrfTableVector[index2]
                                   + _DrfTableVector[index3]
                                   + _DrfTableVector[index4]);
      }
    }
  G4double cosPhi = 0;
  G4double sinPhi = 0;
  G4double halfTableRangeInCmX = (_DrfTableDimensionX * 0.5 - _AverageNumberOfPixels - 2.0)
                                 * _DrfBinSize;
  G4double halfTableRangeInCmY = (_DrfTableDimensionY * 0.5 - _AverageNumberOfPixels - 2.0)
                                 * _DrfBinSize;
  G4double xI = 0;
  G4double yJ = 0;
  G4int index = 0;
  G4double radius = 0;
  for (G4int phiIndex = 0; phiIndex < _NumberOfTanPhi; phiIndex++)
    {
    if (phiIndex == 0)
      {
      sinPhi = 0.0;
      cosPhi = 1.0;
      }
    else if (phiIndex < G4int(_NumberOfTanPhi * 0.5))
      {
      /* in fact, this is the real tan(PHI) */
      cosPhi = 1.0 / sqrt(1.0 + _TanPhiVector[phiIndex] * _TanPhiVector[phiIndex]);
      sinPhi = cosPhi * _TanPhiVector[phiIndex];

      }
    else if (phiIndex == G4int(_NumberOfTanPhi * 0.5))
      {
      /* in fact, from 256-511, the actualy PHI value is for ctg, not for tan */
      sinPhi = sqrt(2.0) * 0.5;
      cosPhi = sinPhi;
      }
    else
      {
      /* the value of dTanPhi is actually the value of ctan of the same angle */
      sinPhi = 1.0
               / sqrt(1.0
                      + _TanPhiVector[_NumberOfTanPhi - phiIndex]
                        * _TanPhiVector[_NumberOfTanPhi - phiIndex]);
      cosPhi = sinPhi * _TanPhiVector[_NumberOfTanPhi - phiIndex];
      }
    for (G4int thetaIndex = 0; thetaIndex < _NumberOfCosTheta; thetaIndex++)
      {
      /* x is cos(Theta), 1.0 --> 0.20 */
      index = thetaIndex + phiIndex * _NumberOfCosTheta;
      if (thetaIndex == 0)
        {
        xI = 0.0;
        yJ = 0.0;
        }
      else
        {
        radius = _DistanceSourceToImage
                 * sqrt(1.0 / (_CosThetaVector[thetaIndex] * _CosThetaVector[thetaIndex]) - 1.0);
        xI = radius * cosPhi;
        yJ = radius * sinPhi;
        }
      if ((xI >= halfTableRangeInCmX) || (yJ >= halfTableRangeInCmY))
        {
        _ArfTableVector[index] = 0.0;
        }
      else
        {
        _ArfTableVector[index] = computeARFfromDRF(xI, yJ, _CosThetaVector[thetaIndex]);
        }
      }
    }

  G4String arfDrfTableBinName = GetName() + "_ARFfromDRFTable.bin";
  size_t tableBufferSize = _TotalNumberbOfThetaPhi * sizeof(G4double);
  std::ofstream outputTableBin(arfDrfTableBinName.c_str(), std::ios::out | std::ios::binary);
  outputTableBin.write((const char*) (_ArfTableVector), tableBufferSize);
  outputTableBin.close();
  }

void GateARFTable::FillDRFTable(const G4double & meanE, const G4double & X, const G4double & Y)
  {
  if (X - _LowX < 0. || X + _LowX > 0.)
    {
    return;
    }
  if (Y - _LowY < 0. || Y + _LowY > 0.)
    {
    return;
    }

  G4int xIndex = G4int((X - _LowX) / _DrfBinSize);
  G4int yIndex = G4int((Y - _LowY) / _DrfBinSize);
  if (xIndex > _DrfTableDimensionX - 1 || (xIndex < 0))
    {
    return;
    }
  if (yIndex > _DrfTableDimensionY - 1 || (yIndex < 0))
    {
    return;
    }
  G4int index = GetOneDimensionIndex(xIndex, yIndex);
  mBinnedPhotonCounter++;
  /*
   erf(x)=1/sqrt(PI) * integral(-x,x) of exp(-x^2)
   =2/sqrt(PI) * integral(0,x) of exp(-x^2)
   assume Gaussian exp(-x^2), where x^2 = (u/sig)^2
   sig = fwhm / (2.0*sqrt(log(2.0)))
   EnergyResolution (ErgReso) is considered 0.0 if -10^(-6)<ErgReso<10^(-6)
   */
  G4double fwhm = 0;
  G4double sigma = 0;
  G4double result = 0.;
  if (mEnergyResolution > 0.000001)
    {
    /* energy resolution (in %) ~ sqrt(E), relative to mEnergyResolution@dResoAtErg kev*/
    fwhm = mEnergyResolution * sqrt(mEnergyReference * meanE);
    sigma = fwhm / (2.0 * sqrt(log(2.0)));
    result = (TMath::Erf((_EnergyHighUser - meanE) / sigma)
              - TMath::Erf((_EnergyLowUser - meanE) / sigma))
             * 0.5;
    }
  else if (mEnergyResolution < -0.000001)
    {
    /* Gaussian width of photopeaks linearly depends on photon energy */
    sigma = mConstantTerm + mLinearTerm * meanE;
    result = (TMath::Erf((_EnergyHighUser - meanE) / sigma)
              - TMath::Erf((_EnergyLowUser - meanE) / sigma))
             * 0.5;
    }
  else if (meanE >= _EnergyLowUser && meanE < _EnergyHighUser)
    {
    /* perfect energy resolution*/
    result = 1.0;
    }
  _DrfTableVector[index] += fabs(result);
  }

void GateARFTable::Describe()
  {
  G4cout << "===== Description of the ARF Table named " << GetName() << " ======\n";
  G4cout << "      Index                  " << GetIndex() << Gateendl;
  G4cout << "      Number of Theta Angles " << GetNbofTheta() << Gateendl;
  G4cout << "      Number of Phi Angles   " << GetNbofPhi() << Gateendl;
  G4cout << "      Total number of values " << GetTotalNb() << Gateendl;
  G4cout << "      Energy Window                 (keV)["
         << GetElow() / keV
         << " , "
         << GetEhigh() / keV
         << "]\n";
  G4cout << "      User Specified Energy Window  (keV)["
         << _EnergyLowUser / keV
         << " , "
         << _EnergyHighUser / keV
         << "]\n";
  G4cout << "      Is Primary             " << GetPrimary() << Gateendl;
  G4cout << "      Energy Resolution " << mEnergyResolution << Gateendl;
  G4cout << "      Energy Of Reference " << mEnergyReference << Gateendl;
  G4cout << "      Energy Deposition Threshold " << _EnergyLowUser / keV << " keV\n";
  G4cout << "      Energy Deposition Uphold " << _EnergyHighUser / keV << " keV\n";

  G4cout << "      Total Number of Binned Photons (not necessarily detected in the energy window chosen) "
         << mBinnedPhotonCounter
         << Gateendl;
  }

void GateARFTable::GetARFAsBinaryBuffer(G4double*& tableBuffer)
  {
  tableBuffer[0] = G4double(GetElow());
  tableBuffer[1] = G4double(GetEhigh());
  tableBuffer[2] = G4double(GetEnergyReso());
  tableBuffer[3] = G4double(GetERef());
  tableBuffer[4] = G4double(GetEWlow());
  tableBuffer[5] = G4double(GetEWhigh());
  for (int i = 0; i < GetTotalNb(); i++)
    {
    tableBuffer[i + 6] = _ArfTableVector[i];
    }
  }

void GateARFTable::FillTableFromBuffer(G4double*& tableBuffer)
  {
  _EnergyLow = tableBuffer[0];
  _EnergyHigh = tableBuffer[1];
  mEnergyResolution = tableBuffer[2];
  mEnergyReference = tableBuffer[3];
  _EnergyLowUser = tableBuffer[4];
  _EnergyHighUser = tableBuffer[5];
  for (int i = 0; i < GetTotalNb(); i++)
    {
    _ArfTableVector[i] = tableBuffer[i + 6];
    }
  }

#endif
